import argparse
import os
from pathlib import Path

import numpy as np
import torch

from UQ.halt.models.halt import HALTModel
from UQ.halt.preprocessing.preprocess_halt import HF_DATASET, preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small HALT UQ demo on a few examples.")
    p.add_argument(
        "--num-examples",
        type=int,
        default=20,
        help="Target number of examples. For 'balanced', uses min(n//2, n0, n1) per class (total 2× that, even).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")
    p.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=("balanced", "random", "head"),
        help="'balanced': equal correct/incorrect when possible. 'random' / 'head': unchanged.",
    )
    p.add_argument("--hf-dataset", type=str, default=HF_DATASET)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to HALT checkpoint (.pth). Default: UQ/halt/artifacts/checkpoints/best_halt_model.pth",
    )
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--input-dim", type=int, default=25)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--top-q", type=float, default=0.15)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for pred=1 (model predicts is_correct).",
    )
    return p.parse_args()


def _repo_root() -> Path:
    # UQ/halt/demo_halt.py -> repo root
    return Path(__file__).resolve().parent.parent.parent


def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _repo_root())
    except ValueError:
        return str(path)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _compute_lengths(x: np.ndarray) -> np.ndarray:
    return (x != 0).any(axis=-1).sum(axis=1).astype(np.int64)


def _pick_indices(y: np.ndarray, n_target: int, sample_mode: str, seed: int) -> tuple[np.ndarray, int]:
    """
    Returns (indices, n_show) where n_show == len(indices).

    balanced: floor(n_target/2) incorrect + same number correct (strictly equal),
              capped by class counts. Total is always even.
    """
    y = np.asarray(y)
    n_total = len(y)
    rng = np.random.default_rng(seed)

    if sample_mode == "head":
        n_show = min(n_target, n_total)
        return np.arange(n_show, dtype=np.int64), n_show

    if sample_mode == "random":
        n_show = min(n_target, n_total)
        return np.sort(rng.choice(n_total, size=n_show, replace=False)), n_show

    # balanced
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n0, n1 = len(idx0), len(idx1)
    half = n_target // 2
    n_per = int(min(half, n0, n1))
    if n_per == 0:
        raise RuntimeError(
            "Balanced sampling needs at least one correct and one incorrect example in the dataset."
        )
    take0 = rng.choice(idx0, size=n_per, replace=False)
    take1 = rng.choice(idx1, size=n_per, replace=False)
    indices = np.sort(np.concatenate([take0, take1]))
    n_show = len(indices)
    if n_per < half:
        print(
            f"Note: balanced demo uses {n_show} examples "
            f"({n_per} incorrect + {n_per} correct); "
            f"requested up to {half} per class but only {n_per} per class available "
            f"(n_incorrect={n0}, n_correct={n1})."
        )
    elif n_target % 2 == 1:
        print(
            f"Note: balanced demo uses {n_show} examples ({n_per}+{n_per}); "
            f"odd --num-examples ({n_target}) uses floor(n/2) per class."
        )
    return indices, n_show


def main():
    args = parse_args()
    device = _resolve_device(args.device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / "artifacts" / "checkpoints" / "best_halt_model.pth"
    else:
        checkpoint = checkpoint.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {_repo_rel(checkpoint)}")

    print(f"Device: {device}")
    print(f"Checkpoint: {_repo_rel(checkpoint)}")
    print(f"Dataset: {args.hf_dataset}")

    x_all, y_all = preprocess(args.hf_dataset)
    y_all = np.asarray(y_all)
    indices, n_show = _pick_indices(y_all, args.num_examples, args.sample_mode, args.seed)

    x = x_all[indices]
    y = y_all[indices].astype(np.float32)
    lengths = _compute_lengths(x)
    n_in_batch = int((y == 0).sum())
    n_cor_batch = int((y == 1).sum())
    print(f"Batch labels: incorrect={n_in_batch}, correct={n_cor_batch} (total={n_show})")

    model = HALTModel(
        input_dim=args.input_dim,
        proj_dim=args.proj_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_q=args.top_q,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        l_t = torch.tensor(lengths, dtype=torch.long, device=device)
        probs = model.predict_proba(x_t, l_t).cpu().numpy()

    thr = float(args.threshold)
    preds = (probs >= thr).astype(np.float32)
    acc = float(np.mean(preds == y))
    brier = float(np.mean((y - probs) ** 2))

    mask0 = y < 0.5
    mask1 = y >= 0.5
    n0, n1 = int(mask0.sum()), int(mask1.sum())
    if n0:
        print(
            f"UQ conf for label=0 (incorrect): min={probs[mask0].min():.4f} "
            f"mean={probs[mask0].mean():.4f} max={probs[mask0].max():.4f}"
        )
    if n1:
        print(
            f"UQ conf for label=1 (correct):   min={probs[mask1].min():.4f} "
            f"mean={probs[mask1].mean():.4f} max={probs[mask1].max():.4f}"
        )

    pos = probs >= thr
    tn = int((~pos & (y < 0.5)).sum())
    fp = int((pos & (y < 0.5)).sum())
    fn = int((~pos & (y >= 0.5)).sum())
    tp = int((pos & (y >= 0.5)).sum())
    print(
        f"Confusion (rows; threshold={thr} on UQ conf): TN={tn} FP={fp} FN={fn} TP={tp} "
        f"(positive = UQ conf ≥ threshold → predict is_correct)"
    )

    print("")
    print(f"Demo on {n_show} samples | Accuracy: {acc:.4f} | Brier: {brier:.4f}")
    print("idx\tlabel\tuq_conf\tlength")
    for i, idx in enumerate(indices):
        print(f"{int(idx)}\t{int(y[i])}\t{probs[i]:.4f}\t{int(lengths[i])}")


if __name__ == "__main__":
    main()
