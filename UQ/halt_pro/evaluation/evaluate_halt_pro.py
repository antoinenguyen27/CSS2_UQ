import argparse
from datetime import datetime
import os
import sys
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
from torch.utils.data import DataLoader, Dataset

from UQ.halt_pro.models.halt import HALTModel
from UQ.halt_pro.preprocessing.preprocess_halt_pro import HF_DATASET, HF_DATA_FILE, preprocess


class HaltDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def collate_fn(batch):
    x, y = zip(*batch)
    lengths = torch.tensor([len(a) for a in x], dtype=torch.long)
    return nn.utils.rnn.pad_sequence(x, batch_first=True), torch.stack(y), lengths


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:], right=False), n_bins - 1)
    total = y_true.size
    ece, mce, bins = 0.0, 0.0, []
    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        lo, hi = float(edges[i]), float(edges[i + 1])
        br = f"[{lo:.1f}, {hi:.1f})" if i < n_bins - 1 else f"[{lo:.1f}, {hi:.1f}]"
        if count == 0:
            bins.append({"range": br, "count": 0, "avg_confidence": float("nan"), "accuracy": float("nan"), "gap": float("nan")})
            continue
        avg_conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append({"range": br, "count": count, "avg_confidence": avg_conf, "accuracy": acc, "gap": gap})
    return {"ece": float(ece), "mce": float(mce), "bins": bins}


def sharpness_stats(y_prob: np.ndarray) -> dict:
    p = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    return {
        "mean_confidence_distance": float(np.mean(np.abs(p - 0.5))),
        "mean_predictive_variance": float(np.mean(p * (1.0 - p))),
    }


def evaluate_model(model, loader, device):
    model.eval()
    sse = 0.0
    n_total = 0
    n_correct = 0
    probs_all, targets_all = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            p = model.predict_proba(x, lengths)
            sse += ((y - p) ** 2).sum().item()
            n_total += y.numel()
            n_correct += ((p >= 0.5).float() == y).sum().item()
            probs_all.append(p.detach().cpu().numpy())
            targets_all.append(y.detach().cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    return {
        "brier": sse / n_total,
        "accuracy": n_correct / n_total,
        "calib": calibration_stats(targets, probs, n_bins=10),
        "sharp": sharpness_stats(probs),
    }


def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _find_repo_root())
    except ValueError:
        return str(path)


def write_markdown_report(path: Path, metrics: dict, n_samples: int, model_path: Path, args: argparse.Namespace, device: torch.device):
    lines = [
        "# HALT-Pro evaluation",
        "",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Checkpoint:** `{_repo_rel(model_path)}`",
        f"- **Hugging Face dataset:** `{args.hf_dataset}`",
        f"- **Data file:** `{args.data_file}`",
        f"- **Device:** `{device}`",
        f"- **Examples evaluated:** {n_samples}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Brier score | {metrics['brier']:.6f} |",
        f"| Accuracy | {metrics['accuracy']:.6f} |",
        f"| ECE (10 bins) | {metrics['calib']['ece']:.6f} |",
        f"| MCE (10 bins) | {metrics['calib']['mce']:.6f} |",
        f"| Sharpness: mean |p-0.5| (higher=sharper) | {metrics['sharp']['mean_confidence_distance']:.6f} |",
        f"| Sharpness: mean p(1-p) (lower=sharper) | {metrics['sharp']['mean_predictive_variance']:.6f} |",
        "",
        "## Reliability bins",
        "",
        "| Bin range | Count | Avg confidence | Accuracy | Abs gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for b in metrics["calib"]["bins"]:
        if b["count"] == 0:
            lines.append(f"| {b['range']} | 0 | - | - | - |")
        else:
            lines.append(f"| {b['range']} | {b['count']} | {b['avg_confidence']:.4f} | {b['accuracy']:.4f} | {b['gap']:.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HALT-Pro on MMLU rerun traces (default: auhsoJ69/mmlu_rerun).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--hf-dataset", type=str, default=HF_DATASET)
    p.add_argument("--data-file", type=str, default=HF_DATA_FILE)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--input-dim", type=int, default=25)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--top-q", type=float, default=0.15)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    model_path = args.checkpoint or (Path(__file__).resolve().parent.parent / "artifacts" / "checkpoints" / "best_halt_pro_model.pth")
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found")

    x, y = preprocess(args.hf_dataset, data_file=args.data_file)
    loader = DataLoader(HaltDataset(x, y), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = HALTModel(
        input_dim=args.input_dim, proj_dim=args.proj_dim, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, top_q=args.top_q
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    metrics = evaluate_model(model, loader, device)

    print(
        f"Evaluation complete. Brier: {metrics['brier']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"ECE: {metrics['calib']['ece']:.4f}, Sharpness |p-0.5|: {metrics['sharp']['mean_confidence_distance']:.4f}"
    )

    out = args.output or (Path(__file__).resolve().parent / f"halt_pro_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    out = Path(out).resolve()
    write_markdown_report(out, metrics, n_samples=len(x), model_path=model_path, args=args, device=device)
    print(f"Wrote markdown report to {_repo_rel(out)}")


if __name__ == "__main__":
    main()

