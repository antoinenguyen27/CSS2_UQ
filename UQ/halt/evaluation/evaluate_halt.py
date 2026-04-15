import argparse
from datetime import datetime
import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Repo root contains pyproject.toml and UQ/. Fallback: parents[3] from this file."""
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
<<<<<<< HEAD
=======
from pathlib import Path
>>>>>>> 3aa4ece0bb9852336fb7a2871ddd5201d03e1040
from UQ.halt.models.halt import HALTModel
from UQ.halt.preprocessing.preprocess_halt import HF_DATASET, preprocess


<<<<<<< HEAD
def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _find_repo_root())
    except ValueError:
        return str(path)
=======
def _repo_root() -> Path:
    # UQ/halt/evaluation/evaluate_halt.py -> repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _repo_root())
    except ValueError:
        return str(path)

>>>>>>> 3aa4ece0bb9852336fb7a2871ddd5201d03e1040

class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT evaluation data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets)
    return inputs, targets, lengths

def brier_score(y_true, y_pred):
    """Compute Brier score: mean((y_true - y_pred)^2)"""
    return torch.mean((y_true - y_pred) ** 2)

def evaluate_model(model, dataloader, device):
    model.eval()
    sse = 0.0
    n_total = 0
    n_correct = 0
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            probs = model.predict_proba(inputs, lengths)
            sse += ((targets - probs) ** 2).sum().item()
            n_total += targets.numel()
            preds = (probs >= 0.5).float()
            n_correct += (preds == targets).sum().item()
    brier = sse / n_total
    accuracy = n_correct / n_total
    return brier, accuracy


def write_markdown_report(
    path: Path,
    *,
    brier: float,
    accuracy: float,
    n_samples: int,
    model_path: Path,
    hf_dataset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Write evaluation summary to a markdown file."""
<<<<<<< HEAD
    repo_root = _find_repo_root()
=======
    repo_root = _repo_root()
>>>>>>> 3aa4ece0bb9852336fb7a2871ddd5201d03e1040
    try:
        model_path_display = os.path.relpath(model_path, repo_root)
    except ValueError:
        model_path_display = str(model_path)

    lines = [
        "# HALT evaluation",
        "",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Checkpoint:** `{model_path_display}`",
        f"- **Hugging Face dataset:** `{hf_dataset}`",
        f"- **Device:** `{device}`",
        f"- **Examples evaluated:** {n_samples}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Brier score | {brier:.6f} |",
        f"| Accuracy | {accuracy:.6f} |",
        "",
        "## Run configuration",
        "",
        "| Argument | Value |",
        "| --- | --- |",
        f"| batch_size | {args.batch_size} |",
        f"| input_dim | {args.input_dim} |",
        f"| proj_dim | {args.proj_dim} |",
        f"| hidden_size | {args.hidden_size} |",
        f"| num_layers | {args.num_layers} |",
        f"| dropout | {args.dropout} |",
        f"| top_q | {args.top_q} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HALT model (Brier score and accuracy on preprocessed data).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained weights (.pth). Default: UQ/halt/artifacts/checkpoints/best_halt_model.pth",
    )
    p.add_argument("--hf-dataset", type=str, default=HF_DATASET, help="Hugging Face dataset id for preprocess().")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Compute device.",
    )
    p.add_argument("--input-dim", type=int, default=25)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--top-q", type=float, default=0.15)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for markdown report. Default: UQ/halt/evaluation/halt_eval_<timestamp>.md",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = args.checkpoint
    if model_path is None:
        model_path = (
            Path(__file__).resolve().parent.parent / "artifacts" / "checkpoints" / "best_halt_model.pth"
        )
    else:
        model_path = Path(model_path).resolve()

    eval_dir = Path(__file__).resolve().parent
    if args.output is None:
        report_path = eval_dir / f"halt_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    else:
        report_path = Path(args.output).resolve()

    # Load and preprocess test data
    X_test, y_test = preprocess(args.hf_dataset)

    # Create dataset and dataloader
    test_dataset = HaltDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = HALTModel(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_q=args.top_q,
    ).to(device)

    # Load trained model checkpoint
    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Loaded model checkpoint from {_repo_rel(model_path)}")
    else:
        raise FileNotFoundError(f"Model checkpoint {model_path} not found")

    # Evaluate model
    brier, accuracy = evaluate_model(model, test_loader, device)
    print(f"Evaluation complete. Brier score: {brier:.4f}, Accuracy: {accuracy:.4f}")

    write_markdown_report(
        report_path,
        brier=brier,
        accuracy=accuracy,
        n_samples=len(X_test),
        model_path=model_path,
        hf_dataset=args.hf_dataset,
        device=device,
        args=args,
    )
    print(f"Wrote markdown report to {_repo_rel(report_path)}")

if __name__ == '__main__':
    main()
