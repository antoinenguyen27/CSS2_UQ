"""
1D CNN with HALT-style uncertainty features, unweighted BCE loss, and Brier/F1 evaluation.
Filters parse failures, uses 25 features per token, and trains with standard BCE.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score, f1_score
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ====================
# Feature extraction helpers (identical to HALT description)
# ====================

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs))

def extract_features_per_step(top20_logprobs, selected_logprob):
    """
    Returns a list of 25 features for one token step.
    Features: avg_logprob, rank_proxy, h_overall, h_alts, delta_h_dec (placeholder),
              then the 20 top‑20 logprobs.
    """
    top20 = np.array(top20_logprobs, dtype=np.float64)
    avg_logprob = np.mean(top20)
    rank_proxy = 1 + np.sum(top20 > selected_logprob)
    probs = softmax(top20)
    h_overall = entropy(probs)
    # Alternatives entropy
    idx_selected = np.argmin(np.abs(top20 - selected_logprob))
    probs_alts = np.delete(probs, idx_selected)
    probs_alts = probs_alts / probs_alts.sum()
    h_alts = entropy(probs_alts)
    delta_h_dec = 0.0  # placeholder, will be filled after whole sequence
    features = [avg_logprob, rank_proxy, h_overall, h_alts, delta_h_dec] + top20.tolist()
    return features

def compute_delta_entropy_sequence(features_list):
    """
    Computes binary decision entropy per step and replaces the placeholder delta_h_dec
    (index 4) with the difference from the previous step.
    """
    h_dec_vals = []
    for step_feat in features_list:
        top20 = step_feat[5:]
        probs = softmax(np.array(top20))
        sorted_probs = np.sort(probs)[::-1]
        p_selected = sorted_probs[0]
        p_comp = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        p = p_selected / (p_selected + p_comp + 1e-12)
        p = np.clip(p, 1e-12, 1-1e-12)
        h_dec = -p * np.log(p) - (1-p) * np.log(1-p)
        h_dec_vals.append(h_dec)
    deltas = [0.0]
    for t in range(1, len(h_dec_vals)):
        deltas.append(h_dec_vals[t] - h_dec_vals[t-1])
    for i, delta in enumerate(deltas):
        features_list[i][4] = delta
    return features_list

# ====================
# Dataset that creates (seq_len, 25) tensors
# ====================
class HALT1DCNNDataset(Dataset):
    def __init__(self, df, max_seq_len=192, pad_value=0.0):
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.features = []
        self.labels = []
        self.lengths = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 25 features"):
            top20_seq = row['top20_token_logprobs']
            sampled_seq = row['sampled_token_logprobs']
            if top20_seq is None or sampled_seq is None:
                continue
            if not isinstance(top20_seq, np.ndarray) or not isinstance(sampled_seq, np.ndarray):
                continue
            min_len = min(len(top20_seq), len(sampled_seq))
            if min_len == 0:
                continue
            top20_seq = top20_seq[:min_len]
            sampled_seq = sampled_seq[:min_len]

            step_features = []
            for t in range(min_len):
                feats = extract_features_per_step(top20_seq[t], sampled_seq[t])
                step_features.append(feats)

            step_features = compute_delta_entropy_sequence(step_features)
            seq_feats = np.array(step_features, dtype=np.float32)   # (T, 25)

            # Truncate or pad to max_seq_len
            if seq_feats.shape[0] > max_seq_len:
                seq_feats = seq_feats[:max_seq_len]
                seq_len = max_seq_len
            else:
                seq_len = seq_feats.shape[0]
                pad_len = max_seq_len - seq_len
                pad_array = np.full((pad_len, 25), self.pad_value, dtype=np.float32)
                seq_feats = np.vstack([seq_feats, pad_array])

            self.features.append(torch.tensor(seq_feats, dtype=torch.float32))
            self.labels.append(int(row['is_correct']))
            self.lengths.append(seq_len)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.lengths[idx]

# ====================
# 1D CNN model (adapted for 25 input channels)
# ====================
class LogProbCNN1D_25(nn.Module):
    def __init__(self, input_channels=25,
                 conv_channels=[64, 128, 128],
                 kernel_sizes=[3, 5, 7],
                 dropout=0.3):
        super().__init__()
        self.input_channels = input_channels
        self.input_conv = nn.Conv1d(input_channels, conv_channels[0], kernel_size=1)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = conv_channels[0]
        for out_ch, k in zip(conv_channels[1:], kernel_sizes):
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, 25) -> (batch, 25, seq_len)
        x = x.permute(0, 2, 1)
        if mask is not None:
            mask_float = mask.float().unsqueeze(1)
            x = x * mask_float + (-1e9) * (1 - mask_float)
        x = F.relu(self.input_conv(x))
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x))
            x = pool(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        logit = self.fc(x).squeeze(-1)
        return logit

# ====================
# Training loop with unweighted BCE, gradient clipping, and F1 logging
# ====================
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Unweighted BCE loss
    criterion = nn.BCEWithLogitsLoss()

    best_val_brier = float('inf')
    patience_counter = 0
    patience = 10
    history = {'train_loss': [], 'train_f1': [], 'train_brier': [],
               'val_loss': [], 'val_f1': [], 'val_brier': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch_x, batch_y, lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            lengths = lengths.to(device)
            # Create mask for real tokens
            mask = torch.arange(batch_x.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            optimizer.zero_grad()
            logits = model(batch_x, mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits)
            train_preds.extend(probs.detach().cpu().numpy())
            train_labels.extend(batch_y.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_brier = brier_score_loss(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds > 0.5)

        # Validation
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_brier = brier_score_loss(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds > 0.5)

        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_brier'].append(train_brier)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_brier'].append(val_brier)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Brier: {train_brier:.4f}")
        print(f"                     Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Brier: {val_brier:.4f}")

        scheduler.step(val_brier)

        if val_brier < best_val_brier:
            best_val_brier = val_brier
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_25.pt')
            print(f"  -> New best model! Val Brier: {val_brier:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_cnn_25.pt'))
    return model, history

def evaluate_model(model, loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            lengths = lengths.to(device)
            mask = torch.arange(batch_x.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            logits = model(batch_x, mask)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# ====================
# Baseline (unchanged)
# ====================
def baseline_avg_logprob(df):
    avg_logprobs = []
    labels = []
    for idx, row in df.iterrows():
        logprobs = row['sampled_token_logprobs']
        if logprobs is None:
            continue
        if isinstance(logprobs, np.ndarray):
            logprobs = logprobs.astype(np.float32)
            logprobs = np.nan_to_num(logprobs, nan=-100.0)
            avg = np.mean(logprobs)
            avg_logprobs.append(avg)
            labels.append(int(row['is_correct']))
    if len(avg_logprobs) == 0:
        return 1.0, 0.5, np.array([])
    avg_logprobs = np.array(avg_logprobs)
    labels = np.array(labels)
    min_val, max_val = avg_logprobs.min(), avg_logprobs.max()
    if max_val > min_val:
        pred_probs = (avg_logprobs - min_val) / (max_val - min_val)
    else:
        pred_probs = np.ones_like(avg_logprobs) * 0.5
    brier = brier_score_loss(labels, pred_probs)
    acc = accuracy_score(labels, pred_probs > 0.5)
    return brier, acc, pred_probs

# ====================
# Data loader (same as your original)
# ====================
def load_mmlu_trace_dataset():
    print("Loading local MMLU dataset...")

    path = Path("local_download_mmlu_rerun/examples.parquet") 

    if not path.exists():
        raise RuntimeError(f"File not found: {path}")

    table = pq.read_table(path)
    df = table.to_pandas()

    print(f"Loaded {len(df)} examples from local file")
    return df

# ====================
# Main execution
# ====================
def main():
    print("="*60)
    print("Loading MMLU Trace Dataset")
    print("="*60)
    df = load_mmlu_trace_dataset()
    print(f"\nTotal examples: {len(df)}")

    # Filter parse failures
    if 'parse_success' in df.columns:
        parse_success_count = df['parse_success'].sum()
        print(f"Parse successes: {parse_success_count} ({parse_success_count/len(df)*100:.1f}%)")
        df = df[df['parse_success'] == True].reset_index(drop=True)
        print(f"Filtered to parsed examples only: {len(df)} examples")
        print(f"Correct answers in parsed examples: {df['is_correct'].mean()*100:.1f}%")
    if len(df) == 0:
        print("ERROR: No successfully parsed examples found!")
        return

    # Train/val/test split (stratified)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_correct'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_correct'])

    max_seq_len = 192
    train_dataset = HALT1DCNNDataset(train_df, max_seq_len=max_seq_len)
    val_dataset = HALT1DCNNDataset(val_df, max_seq_len=max_seq_len)
    test_dataset = HALT1DCNNDataset(test_df, max_seq_len=max_seq_len)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    model = LogProbCNN1D_25(input_channels=25, conv_channels=[64,128,128],
                            kernel_sizes=[3,5,7], dropout=0.3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        model, history = train_model(model, train_loader, val_loader, epochs=50,
                                    lr=1e-3, device=device)
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test evaluation (unweighted BCE)
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    test_brier = brier_score_loss(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds > 0.5)
    test_f1 = f1_score(test_labels, test_preds > 0.5)

    print("\n" + "="*60)
    print("Test Set Evaluation (1D CNN with 25 features, unweighted BCE)")
    print("="*60)
    print(f"Brier Score: {test_brier:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

    # Baseline comparison
    baseline_brier, baseline_acc, _ = baseline_avg_logprob(test_df)
    print("\nBaseline (average logprob):")
    print(f"Brier Score: {baseline_brier:.4f}")
    print(f"Accuracy: {baseline_acc:.4f}")

    if baseline_brier > 0:
        improvement = (baseline_brier - test_brier) / baseline_brier * 100
        print(f"\nImprovement over baseline: {improvement:.1f}% lower Brier")

if __name__ == "__main__":
    main()