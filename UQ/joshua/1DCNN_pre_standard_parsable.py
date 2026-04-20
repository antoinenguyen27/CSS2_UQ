"""
1D CNN on token log-probabilities for MMLU correctness prediction.
Filters out parse failures to evaluate only on valid model outputs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import argparse

# ====================
# 1. Load Dataset - Direct Parquet Loading
# ====================
def load_mmlu_trace_dataset():
    """
    Load the examples.parquet file directly from Hugging Face cache.
    """
    print("Loading MMLU Trace Dataset from antoine3/CSS2_UQ...")
    
    # Find the cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    dataset_dir = cache_dir / "datasets--antoine3--CSS2_UQ"
    
    if dataset_dir.exists():
        snapshots = list(dataset_dir.glob("snapshots/*"))
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_ctime)
            print(f"Found cache at: {latest_snapshot}")
            
            examples_files = list(latest_snapshot.glob("*.parquet"))
            examples_files = [f for f in examples_files if 'examples' in str(f).lower() and 'token' not in str(f).lower()]
            
            if examples_files:
                print(f"Loading {examples_files[0].name}...")
                table = pq.read_table(examples_files[0])
                df = table.to_pandas()
                print(f"Loaded {len(df)} examples")
                return df
    
    raise RuntimeError("Could not load dataset from cache")

# ====================
# 2. Diagnostic Checks
# ====================
def diagnostic_checks(df):
    """Run diagnostics to understand if results are plausible."""
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS")
    print("="*60)
    
    # 1. Class distribution
    correct_count = df['is_correct'].sum()
    total = len(df)
    print(f"\n1. Class Distribution (Parsed Examples Only):")
    print(f"   Correct: {correct_count} ({correct_count/total:.1%})")
    print(f"   Incorrect: {total-correct_count} ({(total-correct_count)/total:.1%})")
    
    # 2. Majority class baseline
    majority_acc = max(correct_count/total, (total-correct_count)/total)
    print(f"\n2. Majority Class Baseline:")
    print(f"   Always predict majority class: {majority_acc:.1%} accuracy")
    
    # 3. Check if logprobs are correlated with correctness
    correct_logprobs = []
    incorrect_logprobs = []
    
    for idx, row in df.iterrows():
        logprobs = row['sampled_token_logprobs']
        if isinstance(logprobs, np.ndarray):
            avg = np.mean(logprobs)
            if row['is_correct']:
                correct_logprobs.append(avg)
            else:
                incorrect_logprobs.append(avg)
    
    if correct_logprobs and incorrect_logprobs:
        print(f"\n3. Average Logprob by Class:")
        print(f"   Correct examples mean logprob: {np.mean(correct_logprobs):.4f}")
        print(f"   Incorrect examples mean logprob: {np.mean(incorrect_logprobs):.4f}")
        print(f"   Difference: {np.mean(correct_logprobs) - np.mean(incorrect_logprobs):.4f}")
    
    # 4. Logistic regression baseline on average logprob
    print(f"\n4. Logistic Regression Baseline (on avg logprob):")
    # Prepare data
    X = []
    y = []
    for idx, row in df.iterrows():
        logprobs = row['sampled_token_logprobs']
        if isinstance(logprobs, np.ndarray) and len(logprobs) > 0:
            X.append(np.mean(logprobs))
            y.append(row['is_correct'])
    
    if len(X) > 0:
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train logistic regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        y_pred = lr.predict(X_test)
        
        lr_brier = brier_score_loss(y_test, y_pred_proba)
        lr_acc = accuracy_score(y_test, y_pred)
        
        print(f"   Accuracy: {lr_acc:.4f}")
        print(f"   Brier Score: {lr_brier:.4f}")
        
        # Compare with majority baseline
        majority_acc_baseline = max(y_test.mean(), 1 - y_test.mean())
        print(f"   Majority class baseline on test set: {majority_acc_baseline:.4f}")
    
    return {
        'majority_acc': majority_acc,
        'class_balance': correct_count/total
    }

# ====================
# 3. Dataset Preprocessing
# ====================
class TokenLogProbDataset(Dataset):
    """PyTorch Dataset for token log-probability sequences."""
    
    def __init__(self, df, max_seq_len=192, pad_value=-100.0, use_top20=False):
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value
        self.use_top20 = use_top20
        
        self.features = []
        self.labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            # Get the logprobs sequence (numpy array)
            logprobs_seq = row['sampled_token_logprobs']
            
            # Handle potential None or NaN
            if logprobs_seq is None:
                continue
            
            # Ensure it's a numpy array (it should be from parquet)
            if not isinstance(logprobs_seq, np.ndarray):
                # If it's a list, convert to numpy
                if isinstance(logprobs_seq, list):
                    logprobs_seq = np.array(logprobs_seq)
                else:
                    continue
            
            # Check if it's empty
            if len(logprobs_seq) == 0:
                continue
            
            # Convert to float32 for efficiency
            logprobs_seq = logprobs_seq.astype(np.float32)
            
            # Replace NaN/inf with pad_value
            logprobs_seq = np.nan_to_num(logprobs_seq, nan=pad_value, posinf=pad_value, neginf=pad_value)
            
            # Truncate if needed
            if len(logprobs_seq) > max_seq_len:
                logprobs_seq = logprobs_seq[:max_seq_len]
            
            if use_top20 and 'top20_token_logprobs' in row and row['top20_token_logprobs'] is not None:
                # Get top20 data
                top20_seq = row['top20_token_logprobs']
                
                if isinstance(top20_seq, np.ndarray):
                    # Top20 should be shape [seq_len, 20] or [seq_len] with each element being array of 20
                    if len(top20_seq.shape) == 1:
                        # Each element is an array of 20 values
                        top20_seq = np.stack([x if isinstance(x, np.ndarray) else np.array(x) for x in top20_seq])
                    
                    if len(top20_seq.shape) == 2 and top20_seq.shape[1] >= 20:
                        # Take first 20 if more than 20
                        top20_seq = top20_seq[:, :20]
                        top20_seq = top20_seq.astype(np.float32)
                        top20_seq = np.nan_to_num(top20_seq, nan=pad_value)
                        
                        if len(top20_seq) > max_seq_len:
                            top20_seq = top20_seq[:max_seq_len]
                        
                        # Create feature: [seq_len, 20]
                        feature = torch.tensor(top20_seq, dtype=torch.float32)
                    else:
                        # Fallback to single channel
                        feature = torch.tensor(logprobs_seq, dtype=torch.float32)
                        self.use_top20 = False
                else:
                    # Fallback to single channel
                    feature = torch.tensor(logprobs_seq, dtype=torch.float32)
                    self.use_top20 = False
            else:
                # Single channel: just the sampled logprob
                feature = torch.tensor(logprobs_seq, dtype=torch.float32)
            
            self.features.append(feature)
            self.labels.append(int(row['is_correct']))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        seq = self.features[idx]
        label = self.labels[idx]
        
        if len(seq.shape) == 2:
            # seq is [seq_len, channels]
            seq_len = seq.shape[0]
            channels = seq.shape[1]
            # Pad: [max_seq_len, channels]
            padded = torch.full((self.max_seq_len, channels), self.pad_value, dtype=torch.float32)
            padded[:seq_len] = seq
        else:
            # seq is [seq_len]
            seq_len = len(seq)
            # Pad: [max_seq_len]
            padded = torch.full((self.max_seq_len,), self.pad_value, dtype=torch.float32)
            padded[:seq_len] = seq
        
        # Mask: True for real tokens
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = True
        
        return padded, mask, label, seq_len

# ====================
# 4. 1D CNN Model
# ====================
class LogProbCNN1D(nn.Module):
    """
    1D CNN that processes token log-probability sequences.
    """
    
    def __init__(self, 
                 input_channels=1,
                 conv_channels=[64, 128, 128],
                 kernel_sizes=[3, 5, 7],
                 dropout=0.3):
        super().__init__()
        
        self.input_channels = input_channels
        
        # Input projection
        self.input_conv = nn.Conv1d(input_channels, conv_channels[0], kernel_size=1)
        
        # Multi-scale convolutional layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = conv_channels[0]
        for out_ch, k in zip(conv_channels[1:], kernel_sizes):
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = out_ch
        
        # Adaptive pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len] for single-channel, or [batch, seq_len, channels] for multi-channel
        """
        if len(x.shape) == 3:
            # Multi-channel: [batch, seq_len, channels] -> [batch, channels, seq_len]
            x = x.permute(0, 2, 1)
        else:
            # Single-channel: [batch, seq_len] -> [batch, 1, seq_len]
            x = x.unsqueeze(1)
        
        # Mask out padding
        if mask is not None:
            mask_float = mask.float().unsqueeze(1)
            x = x * mask_float + (-1e9) * (1 - mask_float)
        
        # Input convolution
        x = F.relu(self.input_conv(x))
        
        # Multi-scale convs with pooling
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x))
            x = pool(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classifier
        x = self.dropout(x)
        logit = self.fc(x).squeeze(-1)
        
        return logit

# ====================
# 5. Training Loop
# ====================
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
    """Train the CNN model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_brier = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_brier': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_mask, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_brier = brier_score_loss(val_labels, val_preds)
        
        history['val_loss'].append(val_loss)
        history['val_brier'].append(val_brier)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Brier: {val_brier:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_brier < best_val_brier:
            best_val_brier = val_brier
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model.pt')
            print(f"  -> New best model! Brier: {val_brier:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if Path('best_cnn_model.pt').exists():
        model.load_state_dict(torch.load('best_cnn_model.pt'))
    
    return model, history

def evaluate_model(model, loader, criterion, device='cuda'):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_mask, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.float().to(device)
            
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# ====================
# 6. Baseline Comparison
# ====================
def baseline_avg_logprob(df):
    """Simple baseline: use average logprob of the sequence as prediction."""
    avg_logprobs = []
    labels = []
    
    for idx, row in df.iterrows():
        logprobs = row['sampled_token_logprobs']
        
        if logprobs is None:
            continue
        
        if isinstance(logprobs, np.ndarray):
            logprobs = logprobs.astype(np.float32)
            logprobs = np.nan_to_num(logprobs, nan=-100.0, posinf=-100.0, neginf=-100.0)
            
            avg = np.mean(logprobs)
            avg_logprobs.append(avg)
            labels.append(int(row['is_correct']))
    
    if len(avg_logprobs) == 0:
        return 1.0, 0.5, np.array([])
    
    avg_logprobs = np.array(avg_logprobs)
    labels = np.array(labels)
    
    # Normalize to [0, 1] using min-max scaling
    min_val, max_val = avg_logprobs.min(), avg_logprobs.max()
    if max_val > min_val:
        pred_probs = (avg_logprobs - min_val) / (max_val - min_val)
    else:
        pred_probs = np.ones_like(avg_logprobs) * 0.5
    
    brier = brier_score_loss(labels, pred_probs)
    acc = accuracy_score(labels, pred_probs > 0.5)
    
    return brier, acc, pred_probs

# ====================
# 7. Main Execution
# ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_channels', type=str, default='64,128,128',
                        help='Comma-separated list of conv channels (e.g., 32,64,64)')
    parser.add_argument('--kernel_sizes', type=str, default='3,5,7',
                        help='Comma-separated list of kernel sizes (e.g., 3,5,7)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--use_top20', action='store_true',
                        help='Use top-20 logprobs instead of single channel')
    args = parser.parse_args()

    # Convert comma-separated strings to lists
    conv_channels = [int(x) for x in args.conv_channels.split(',')]
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    
    print("=" * 60)
    print("Loading MMLU Trace Dataset")
    print("=" * 60)
    
    # Load data
    df = load_mmlu_trace_dataset()
    
    print(f"\nTotal examples: {len(df)}")
    
    # ========== FILTER OUT PARSE FAILURES ==========
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    if 'parse_success' in df.columns:
        parse_success_count = df['parse_success'].sum()
        parse_failure_count = len(df) - parse_success_count
        print(f"Parse successes: {parse_success_count} ({parse_success_count/len(df)*100:.1f}%)")
        print(f"Parse failures: {parse_failure_count} ({parse_failure_count/len(df)*100:.1f}%)")
        
        # Filter to only successfully parsed examples
        df = df[df['parse_success'] == True].reset_index(drop=True)
        print(f"\nFiltered to parsed examples only: {len(df)} examples")
        print(f"Correct answers in parsed examples: {df['is_correct'].mean()*100:.1f}%")
    else:
        print("Warning: 'parse_success' column not found - using all examples")
    
    if len(df) == 0:
        print("ERROR: No successfully parsed examples found!")
        return None, None, None
    
    # ========== DIAGNOSTIC CHECKS ==========
    diagnostic_checks(df)
    
    # ========== PREPARE DATA FOR TRAINING ==========
    print("\n" + "="*60)
    print("PREPARING DATA FOR TRAINING")
    print("="*60)
    
    # Check data
    sample_logprobs = df['sampled_token_logprobs'].iloc[0]
    print(f"\nSample logprobs shape: {sample_logprobs.shape if hasattr(sample_logprobs, 'shape') else 'N/A'}")
    
    # Filter out rows with invalid logprobs
    valid_mask = df['sampled_token_logprobs'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)
    df = df[valid_mask].reset_index(drop=True)
    
    print(f"Valid examples after filtering: {len(df)}")
    
    if len(df) == 0:
        print("ERROR: No valid examples after filtering")
        return None, None, None
    
    # Get sequence lengths
    seq_lengths = df['sampled_token_logprobs'].apply(len)
    print(f"Sequence lengths - min: {seq_lengths.min()}, max: {seq_lengths.max()}, "
          f"mean: {seq_lengths.mean():.1f}, std: {seq_lengths.std():.1f}")
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_correct'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_correct'])
    
    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Class distribution - train: {train_df['is_correct'].mean():.2f}, "
          f"val: {val_df['is_correct'].mean():.2f}, test: {test_df['is_correct'].mean():.2f}")
    
    # Fixed sequence length (from inspection, all are 192)
    max_seq_len = 192
    print(f"\nUsing max sequence length: {max_seq_len}")
    
    use_top20 = 'top20_token_logprobs' in df.columns
    print(f"Using top-20 logprobs: {use_top20}")
    
    # Create datasets
    train_dataset = TokenLogProbDataset(train_df, max_seq_len=max_seq_len, use_top20=use_top20)
    val_dataset = TokenLogProbDataset(val_df, max_seq_len=max_seq_len, use_top20=use_top20)
    test_dataset = TokenLogProbDataset(test_df, max_seq_len=max_seq_len, use_top20=use_top20)
    
    print(f"\nDataset sizes after preprocessing: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training examples after preprocessing")
        return None, None, None
    
    # Create dataloaders
    batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    input_channels = 20 if use_top20 else 1
    model = LogProbCNN1D(
        input_channels=input_channels,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training 1D CNN on Token Log-Probabilities")
    print("=" * 60)
    model, history = train_model(model, train_loader, val_loader,
                             epochs=args.epochs, lr=args.lr, device=device)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_loss, test_preds, test_labels = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device)
    test_brier = brier_score_loss(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds > 0.5)
    
    print(f"\n1D CNN Results (on correctly parsed examples only):")
    print(f"  Brier Score: {test_brier:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Baseline comparison
    print("\n" + "=" * 60)
    print("Baseline: Average Token Log-Probability")
    print("=" * 60)
    baseline_brier, baseline_acc, _ = baseline_avg_logprob(test_df)
    print(f"Baseline Brier Score: {baseline_brier:.4f}")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # Improvement summary
    if baseline_brier > 0:
        improvement = baseline_brier - test_brier
        improvement_pct = (improvement / baseline_brier) * 100
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Brier Score Improvement: {improvement:.4f} ({improvement_pct:.1f}% lower)")
    
    return model, test_brier, test_acc

if __name__ == "__main__":
    model, brier, acc = main()
    if brier is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Test Brier Score: {brier:.4f}")
        print(f"Test Accuracy: {acc:.4f}")