import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def _find_repo_root() -> Path:
    root = Path(__file__).resolve()
    for _ in range(12):
        if (root / "pyproject.toml").is_file() and (root / "UQ").is_dir():
            return root
        parent = root.parent
        if parent == root:
            break
        root = parent
    return Path(__file__).resolve().parents[3]


def _ensure_repo_on_path() -> None:
    r = _find_repo_root()
    s = str(r)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_repo_on_path()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from UQ.halt.models.halt import HALTModel
from UQ.halt.preprocessing.preprocess_halt import HF_DATASET, preprocess
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute ECE/MCE for binary probabilities."""
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:], right=False), n_bins - 1)
    total = y_true.size
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (count / total) * gap
        mce = max(mce, gap)

    return {"ece": float(ece), "mce": float(mce)}


def sharpness_stats(y_prob: np.ndarray) -> dict:
    """
    Compute sharpness proxies for binary probabilities.
    - mean_confidence_distance: mean(|p - 0.5|), higher means sharper.
    - mean_predictive_variance: mean(p * (1 - p)), lower means sharper.
    """
    p = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    return {
        "mean_confidence_distance": float(np.mean(np.abs(p - 0.5))),
        "mean_predictive_variance": float(np.mean(p * (1.0 - p))),
    }


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
    all_probs = []
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
        all_probs.extend(probs.detach().cpu().numpy())

        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)

    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    train_brier = sse / n_brier
    train_probs = np.array(all_probs, dtype=np.float32)
    train_labels = np.array(all_labels, dtype=np.float32)
    train_calib = calibration_stats(train_labels, train_probs, n_bins=10)
    train_sharp = sharpness_stats(train_probs)
    writer.add_scalar('train/macro_f1', macro_f1, epoch)
    writer.add_scalar('train/brier', train_brier, epoch)
    writer.add_scalar('train/ece', train_calib["ece"], epoch)
    writer.add_scalar('train/mce', train_calib["mce"], epoch)
    writer.add_scalar('train/sharpness_conf_distance', train_sharp["mean_confidence_distance"], epoch)
    writer.add_scalar('train/sharpness_pred_variance', train_sharp["mean_predictive_variance"], epoch)
    return total_loss / len(dataloader), macro_f1, train_brier, train_calib, train_sharp

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
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
            all_probs.extend(probs.detach().cpu().numpy())

    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    val_brier = sse / n_brier
    val_probs = np.array(all_probs, dtype=np.float32)
    val_labels = np.array(all_labels, dtype=np.float32)
    val_calib = calibration_stats(val_labels, val_probs, n_bins=10)
    val_sharp = sharpness_stats(val_probs)
    return total_loss / len(dataloader), macro_f1, val_brier, val_calib, val_sharp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HALT model (binary UQ classifier).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", "--learning-rate", type=float, default=4.41e-4, dest="lr")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val Brier improvement).")
    p.add_argument(
        "--ece-tolerance",
        type=float,
        default=0.01,
        help="Checkpoint guard: allow val ECE to worsen by at most this amount when val Brier improves.",
    )
    p.add_argument(
        "--brier-tolerance",
        type=float,
        default=0.002,
        help="Checkpoint tie window for Brier; within this, prefer lower val ECE.",
    )
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to save best model (.pth). Default: UQ/halt/artifacts/checkpoints/best_halt_model.pth",
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
        help="TensorBoard event directory. Default: UQ/halt/artifacts/runs/<timestamp>_<comment>.",
    )
    p.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path to write final/best training metrics for external tuners.",
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


def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _find_repo_root())
    except ValueError:
        return str(path)


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
    print(f"TensorBoard log directory: {_repo_rel(tb_log_dir)}")

    # Early stopping variables (balanced checkpointing on validation Brier + ECE)
    best_val_brier = float("inf")
    best_val_ece = float("inf")
    best_val_f1 = float("-inf")
    best_val_sharp = float("nan")
    best_epoch = -1
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        # Train
        train_loss, train_f1, train_brier, train_calib, train_sharp = train_epoch(
            model, train_loader, optimizer, criterion, device, writer, epoch
        )

        # Validate
        val_loss, val_f1, val_brier, val_calib, val_sharp = eval_epoch(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        writer.add_scalar('val/brier', val_brier, epoch)
        writer.add_scalar('val/ece', val_calib["ece"], epoch)
        writer.add_scalar('val/mce', val_calib["mce"], epoch)
        writer.add_scalar('val/sharpness_conf_distance', val_sharp["mean_confidence_distance"], epoch)
        writer.add_scalar('val/sharpness_pred_variance', val_sharp["mean_predictive_variance"], epoch)
        print(
            f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
            f'Train Brier: {train_brier:.4f}, Train ECE: {train_calib["ece"]:.4f}, '
            f'Train Sharp(|p-0.5|): {train_sharp["mean_confidence_distance"]:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Brier: {val_brier:.4f}, '
            f'Val ECE: {val_calib["ece"]:.4f}, Val Sharp(|p-0.5|): {val_sharp["mean_confidence_distance"]:.4f}'
        )

        # Update learning rate scheduler
        scheduler.step(val_brier)

        # Balanced checkpointing:
        # 1) Brier must improve and ECE must not degrade too much, OR
        # 2) Brier is near-tied and ECE improves.
        brier_improved = val_brier < (best_val_brier - 1e-12)
        ece_guard_ok = val_calib["ece"] <= (best_val_ece + args.ece_tolerance)
        brier_near_tie = val_brier <= (best_val_brier + args.brier_tolerance)
        ece_improved = val_calib["ece"] < (best_val_ece - 1e-12)
        should_save = (brier_improved and ece_guard_ok) or (brier_near_tie and ece_improved)

        if should_save:
            best_val_brier = val_brier
            best_val_ece = val_calib["ece"]
            best_val_f1 = val_f1
            best_val_sharp = val_sharp["mean_confidence_distance"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f'New best model saved to {_repo_rel(best_model_path)} '
                f'with val Brier: {val_brier:.4f}, val ECE: {val_calib["ece"]:.4f}'
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(
        f'Training complete. Best validation Brier: {best_val_brier:.4f}, '
        f'Best validation ECE: {best_val_ece:.4f}'
    )
    if args.metrics_output is not None:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        metrics = {
            "best_epoch": best_epoch,
            "best_val_brier": best_val_brier,
            "best_val_ece": best_val_ece,
            "best_val_f1": best_val_f1,
            "best_val_sharpness_conf_distance": best_val_sharp,
            "checkpoint": str(best_model_path.resolve()),
            "tensorboard_dir": str(tb_log_dir.resolve()),
            "hf_dataset": args.hf_dataset,
            "args": serializable_args,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics summary to {_repo_rel(metrics_path)}")
    writer.close()

if __name__ == '__main__':
    main()
