import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from modeling.halt.models.halt import HALTModel
from modeling.halt.preprocessing.preprocess_halt import HF_DATASET, preprocess
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT training data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # Compute actual sequence lengths based on non-zero timesteps
        # (preprocess pads with zeros, so this reliably finds the true length)
        self.lengths = torch.tensor(
            [(x != 0).any(dim=-1).sum().item() for x in self.X],
            dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]

def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    # All sequences are already padded to MAX_LEN by preprocess(), so stack works directly
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)
    return inputs, targets, lengths

def train_epoch(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    sse = 0.0
    n_brier = 0

    for batch_idx, (inputs, targets, lengths) in enumerate(dataloader):
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        sse += ((probs - targets) ** 2).sum().item()
        n_brier += targets.numel()

        # Collect predictions for F1 calculation
        preds = probs > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)

    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    train_brier = sse / n_brier
    writer.add_scalar('train/macro_f1', macro_f1, epoch)
    writer.add_scalar('train/brier', train_brier, epoch)
    return total_loss / len(dataloader), macro_f1, train_brier

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    sse = 0.0
    n_brier = 0

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            sse += ((probs - targets) ** 2).sum().item()
            n_brier += targets.numel()

            # Collect predictions for F1 calculation
            preds = probs > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    val_brier = sse / n_brier
    return total_loss / len(dataloader), macro_f1, val_brier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HALT model (binary UQ classifier).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", "--learning-rate", type=float, default=4.41e-4, dest="lr")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val Brier improvement).")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to save best model (.pth). Default: modeling/halt/artifacts/checkpoints/best_halt_model.pth",
    )
    p.add_argument("--hf-dataset", type=str, default=HF_DATASET, help="Hugging Face dataset id for preprocess().")
    p.add_argument(
        "--tensorboard-comment",
        type=str,
        default="_halt_training",
        help="Suffix for TensorBoard run name.",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard event directory. Default: modeling/halt/artifacts/runs/<timestamp>_<comment>.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Compute device.",
    )
    # Model architecture (defaults match HALT paper setup in code)
    p.add_argument("--input-dim", type=int, default=25)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--top-q", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"

    best_model_path = args.checkpoint
    if best_model_path is None:
        best_model_path = artifacts_root / "checkpoints" / "best_halt_model.pth"
    else:
        best_model_path = Path(best_model_path).resolve()
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data (called once)
    X, y = preprocess(args.hf_dataset)

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Train correct: {y_train.sum()}, Train incorrect: {len(y_train) - y_train.sum()}")
    print(f"Val correct: {y_val.sum()}, Val incorrect: {len(y_val) - y_val.sum()}")

    # Create datasets and dataloaders
    train_dataset = HaltDataset(X_train, y_train)
    val_dataset = HaltDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model with paper-matching architecture
    model = HALTModel(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_q=args.top_q,
    ).to(device)

    # Loss and optimizer (BCEWithLogitsLoss is correct for binary classification with raw logits)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (minimize validation Brier score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    if args.tensorboard_dir is not None:
        tb_log_dir = Path(args.tensorboard_dir).resolve()
    else:
        stamp = datetime.now().strftime("%b%d_%H-%M-%S")
        tb_log_dir = artifacts_root / "runs" / f"{stamp}_{args.tensorboard_comment}"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log directory: {tb_log_dir}")

    # Early stopping variables (best = lowest validation Brier)
    best_val_brier = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss, train_f1, train_brier = train_epoch(
            model, train_loader, optimizer, criterion, device, writer, epoch
        )

        # Validate
        val_loss, val_f1, val_brier = eval_epoch(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        writer.add_scalar('val/brier', val_brier, epoch)
        print(
            f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
            f'Train Brier: {train_brier:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Brier: {val_brier:.4f}'
        )

        # Update learning rate scheduler
        scheduler.step(val_brier)

        # Early stopping check based on validation Brier (lower is better)
        if val_brier < best_val_brier:
            best_val_brier = val_brier
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path} with val Brier: {val_brier:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training complete. Best validation Brier: {best_val_brier:.4f}')
    writer.close()

if __name__ == '__main__':
    main()
