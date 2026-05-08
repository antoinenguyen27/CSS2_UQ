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
    s = str(_find_repo_root())
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_repo_on_path()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from UQ.halt_pro.models.halt import HALTModel
from UQ.halt_pro.preprocessing.preprocess_halt_pro import HF_DATASET, HF_DATA_FILE, preprocess


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:], right=False), n_bins - 1)
    total = y_true.size
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
        ece += (count / total) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def sharpness_stats(y_prob: np.ndarray) -> dict:
    p = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    return {
        "mean_confidence_distance": float(np.mean(np.abs(p - 0.5))),
        "mean_predictive_variance": float(np.mean(p * (1.0 - p))),
    }


class HaltDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lengths = torch.tensor([(a != 0).any(axis=-1).sum() for a in x], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.lengths[idx]


def collate_fn(batch):
    x, y, lengths = zip(*batch)
    return torch.stack(x), torch.stack(y), torch.stack(lengths)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    probs_all, labels_all = [], []
    sse, n = 0.0, 0

    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        if train_mode:
            optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.item())
        p = torch.sigmoid(logits)
        sse += float(((p - y) ** 2).sum().item())
        n += int(y.numel())
        probs_all.extend(p.detach().cpu().numpy())
        labels_all.extend(y.detach().cpu().numpy())

    probs = np.array(probs_all, dtype=np.float32)
    labels = np.array(labels_all, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int8)
    return {
        "loss": total_loss / max(1, len(loader)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "brier": sse / max(1, n),
        "calib": calibration_stats(labels, probs, n_bins=10),
        "sharp": sharpness_stats(probs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HALT-Pro on MMLU rerun traces (default: auhsoJ69/mmlu_rerun).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", "--learning-rate", type=float, default=4.41e-4, dest="lr")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ece-tolerance", type=float, default=0.01)
    p.add_argument("--brier-tolerance", type=float, default=0.002)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-dataset", type=str, default=HF_DATASET)
    p.add_argument("--data-file", type=str, default=HF_DATA_FILE)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--tensorboard-dir", type=str, default=None)
    p.add_argument("--tensorboard-comment", type=str, default="_halt_pro_training")
    p.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path to write final/best training metrics for external tuners.",
    )
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--input-dim", type=int, default=25)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--top-q", type=float, default=0.15)
    p.add_argument("--weight-decay", type=float, default=0.0)
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
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"
    ckpt = args.checkpoint or (artifacts_root / "checkpoints" / "best_halt_pro_model.pth")
    ckpt = Path(ckpt).resolve()
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    x, y = preprocess(args.hf_dataset, data_file=args.data_file)
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=args.val_split, stratify=y, random_state=args.seed)
    train_loader = DataLoader(HaltDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(HaltDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HALTModel(
        input_dim=args.input_dim, proj_dim=args.proj_dim, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, top_q=args.top_q
    ).to(device)
    n_neg = max(1, int((y_tr == 0).sum()))
    n_pos = max(1, int((y_tr == 1).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    if args.tensorboard_dir:
        tb_dir = Path(args.tensorboard_dir).resolve()
    else:
        stamp = datetime.now().strftime("%b%d_%H-%M-%S")
        tb_dir = artifacts_root / "runs" / f"{stamp}_{args.tensorboard_comment}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard log directory: {_repo_rel(tb_dir)}")

    best_brier, best_ece, best_epoch, patience = float("inf"), float("inf"), 0, 0
    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(va["brier"])
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for split, m in (("train", tr), ("val", va)):
            writer.add_scalar(f"{split}/loss", m["loss"], epoch)
            writer.add_scalar(f"{split}/macro_f1", m["f1"], epoch)
            writer.add_scalar(f"{split}/brier", m["brier"], epoch)
            writer.add_scalar(f"{split}/ece", m["calib"]["ece"], epoch)
            writer.add_scalar(f"{split}/mce", m["calib"]["mce"], epoch)
            writer.add_scalar(f"{split}/sharpness_conf_distance", m["sharp"]["mean_confidence_distance"], epoch)
            writer.add_scalar(f"{split}/sharpness_pred_variance", m["sharp"]["mean_predictive_variance"], epoch)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train brier={tr['brier']:.4f} ece={tr['calib']['ece']:.4f} sharp={tr['sharp']['mean_confidence_distance']:.4f} | "
            f"val brier={va['brier']:.4f} ece={va['calib']['ece']:.4f} sharp={va['sharp']['mean_confidence_distance']:.4f}"
        )

        brier_improved = va["brier"] < (best_brier - 1e-12)
        ece_guard_ok = va["calib"]["ece"] <= (best_ece + args.ece_tolerance)
        brier_near_tie = va["brier"] <= (best_brier + args.brier_tolerance)
        ece_improved = va["calib"]["ece"] < (best_ece - 1e-12)
        should_save = (brier_improved and ece_guard_ok) or (brier_near_tie and ece_improved)

        if should_save:
            best_brier = va["brier"]
            best_ece = va["calib"]["ece"]
            best_val_f1 = va["f1"]
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint: {_repo_rel(ckpt)} (val brier={best_brier:.4f}, val ece={best_ece:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()

    if args.metrics_output is not None:
        metrics_path = Path(args.metrics_output).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        metrics = {
            "best_epoch": best_epoch,
            "best_val_brier": best_brier,
            "best_val_ece": best_ece,
            "best_val_f1": best_val_f1,
            "checkpoint": str(ckpt.resolve()),
            "tensorboard_dir": str(tb_dir.resolve()),
            "hf_dataset": args.hf_dataset,
            "args": serializable_args,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics summary to {_repo_rel(metrics_path)}")

    print(f"Training complete. Best val brier={best_brier:.4f}, best val ece={best_ece:.4f}")


if __name__ == "__main__":
    main()

