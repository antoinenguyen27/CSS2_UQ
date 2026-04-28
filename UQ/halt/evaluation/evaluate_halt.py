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
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from UQ.halt.models.halt import HALTModel
from UQ.halt.preprocessing.preprocess_halt import (
    HF_DATASET,
    preprocess,
    validate_row,
    build_feature_sequence,
)


def _repo_rel(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), _find_repo_root())
    except ValueError:
        return str(path)


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


def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute ECE/MCE and per-bin reliability statistics."""
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:], right=False), n_bins - 1)
    total = y_true.size
    ece = 0.0
    mce = 0.0
    bins = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        lo = float(edges[i])
        hi = float(edges[i + 1])
        bin_range = f"[{lo:.1f}, {hi:.1f})" if i < n_bins - 1 else f"[{lo:.1f}, {hi:.1f}]"
        if count == 0:
            bins.append(
                {
                    "range": bin_range,
                    "count": 0,
                    "avg_confidence": float("nan"),
                    "accuracy": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue

        avg_conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "range": bin_range,
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )

    return {"ece": float(ece), "mce": float(mce), "bins": bins}


def sharpness_stats(y_prob: np.ndarray) -> dict:
    """
    Compute sharpness for probabilistic binary predictions.
    - mean_confidence_distance: mean(|p - 0.5|), higher means sharper.
    - mean_predictive_variance: mean(p * (1 - p)), lower means sharper.
    """
    p = np.clip(y_prob.astype(np.float32).reshape(-1), 0.0, 1.0)
    return {
        "mean_confidence_distance": float(np.mean(np.abs(p - 0.5))),
        "mean_predictive_variance": float(np.mean(p * (1.0 - p))),
    }


def _load_hf_dataframe(hf_dataset: str):
    """Load the same Parquet split as `preprocess_halt.preprocess` (train split)."""
    print(f"Loading dataset: {hf_dataset}")
    ds = load_dataset(hf_dataset, data_files="examples.parquet")
    df = ds["train"].to_pandas()
    print(f"Total rows: {len(df)}")
    return df


def _filter_dataframe_like_1dcnn_before_split(df):
    """
    Match `UQ/1DCNN/main.py` before train/val/test split:
    - keep parse_success rows only
    - require sampled_token_logprobs to be a non-empty ndarray
    """
    if "parse_success" in df.columns:
        df = df[df["parse_success"] == True].reset_index(drop=True)  # noqa: E712
        print(f"After parse_success filter: {len(df)} rows")

    if "sampled_token_logprobs" not in df.columns:
        raise KeyError("Column 'sampled_token_logprobs' missing — cannot mirror 1DCNN split.")

    valid_mask = df["sampled_token_logprobs"].apply(
        lambda x: isinstance(x, np.ndarray) and len(x) > 0
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"After sampled_token_logprobs filter: {len(df)} rows")
    return df


def _one_dcnn_train_val_test(df, random_state: int = 42):
    """
    Same split as `UQ/1DCNN/main.py`: 70% train, 15% val, 15% test;
    stratified by is_correct (random_state=42).
    """
    y_strat = df["is_correct"]
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=random_state, stratify=y_strat
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["is_correct"]
    )
    return train_df, val_df, test_df


def _featurize_halt_from_dataframe(df):
    """Build HALT (N, T, 25) features and labels from a dataframe subset."""
    features_list = []
    labels_list = []
    skipped: dict[str, int] = {}

    for _, row in df.iterrows():
        is_valid, reason = validate_row(row)
        if not is_valid:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        features = build_feature_sequence(row["top20_token_logprobs"])
        if features is None:
            skipped["bad feature sequence"] = skipped.get("bad feature sequence", 0) + 1
            continue
        features_list.append(features)
        labels_list.append(float(row["is_correct"]))

    n = len(features_list)
    if n == 0:
        raise RuntimeError("No valid rows after HALT featurization for this split.")

    X = np.stack(features_list, axis=0)
    y = np.array(labels_list, dtype=np.int8)
    return X, y, skipped


def evaluate_model(model, dataloader, device):
    model.eval()
    sse = 0.0
    n_total = 0
    n_correct = 0
    probs_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            probs = model.predict_proba(inputs, lengths)
            sse += ((targets - probs) ** 2).sum().item()
            n_total += targets.numel()
            preds = (probs >= 0.5).float()
            n_correct += (preds == targets).sum().item()
            probs_all.append(probs.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())
    brier = sse / n_total
    accuracy = n_correct / n_total
    y_prob = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(targets_all, axis=0)
    calib = calibration_stats(y_true, y_prob, n_bins=10)
    sharp = sharpness_stats(y_prob)
    return brier, accuracy, calib, sharp


def write_markdown_report(
    path: Path,
    *,
    brier: float,
    accuracy: float,
    ece: float,
    mce: float,
    sharpness_confidence_distance: float,
    sharpness_predictive_variance: float,
    calibration_bins: list[dict],
    n_samples: int,
    model_path: Path,
    hf_dataset: str,
    device: torch.device,
    args: argparse.Namespace,
    split_meta: dict | None = None,
) -> None:
    """Write evaluation summary to a markdown file."""
    repo_root = _find_repo_root()
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
    ]
    if split_meta is not None:
        lines += ["## Data split", ""]
        if split_meta.get("scheme") == "1dcnn":
            lines += [
                "Aligned with `UQ/1DCNN/main.py`: `parse_success` filter, non-empty "
                "`sampled_token_logprobs` as `ndarray`, then "
                "`train_test_split(test_size=0.3, stratify=is_correct, random_state=42)` and "
                "`train_test_split(temp, test_size=0.5, stratify=is_correct, random_state=42)` "
                "(70% / 15% / 15% train / val / test).",
                "",
                f"- **Eval subset:** `{split_meta['eval_split']}`",
                f"- **Train / val / test row counts (after 1DCNN filters):** "
                f"{split_meta['train_rows']} / {split_meta['val_rows']} / {split_meta['test_rows']}",
                f"- **Rows in eval subset before HALT featurization:** {split_meta['eval_rows_before_feat']}",
                f"- **Rows after HALT featurization (evaluated):** {split_meta['eval_rows_featurized']}",
            ]
            if split_meta.get("skipped_summary"):
                lines.append(
                    f"- **Skipped within eval subset (HALT pipeline):** {split_meta['skipped_summary']}"
                )
            lines.append("")
        elif split_meta.get("scheme") == "full_preprocessed":
            n = split_meta.get("n_rows", 0)
            lines += [
                f"Full dataset: all {n} rows kept by `preprocess_halt.preprocess()` "
                "(not the 1DCNN train/val/test split).",
                "",
            ]

    lines += [
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Brier score | {brier:.6f} |",
        f"| Accuracy | {accuracy:.6f} |",
        f"| ECE (10 bins) | {ece:.6f} |",
        f"| MCE (10 bins) | {mce:.6f} |",
        f"| Sharpness: mean |p-0.5| (higher=sharper) | {sharpness_confidence_distance:.6f} |",
        f"| Sharpness: mean p(1-p) (lower=sharper) | {sharpness_predictive_variance:.6f} |",
        "",
        "## Reliability bins",
        "",
        "| Bin range | Count | Avg confidence | Accuracy | Abs gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for b in calibration_bins:
        if b["count"] == 0:
            lines.append(f"| {b['range']} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {b['range']} | {b['count']} | {b['avg_confidence']:.4f} | {b['accuracy']:.4f} | {b['gap']:.4f} |"
            )
    lines += [
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
        f"| eval_split | {args.eval_split} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate HALT (Brier, calibration, sharpness). "
            "By default uses the 1DCNN test split (70/15/15, stratified, random_state=42); "
            "use --eval-split full for all preprocessed rows."
        )
    )
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
        "--eval-split",
        type=str,
        default="test",
        choices=("test", "val", "train", "full"),
        help=(
            "Which examples to evaluate on. "
            "`test`/`val`/`train` use the same 70/15/15 stratified split as `UQ/1DCNN/main.py` "
            "(random_state=42). `full` evaluates on all rows kept by `preprocess_halt.preprocess()`."
        ),
    )
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

    split_meta: dict | None = None

    if args.eval_split == "full":
        X_test, y_test = preprocess(args.hf_dataset)
        split_meta = {"scheme": "full_preprocessed", "n_rows": len(X_test)}
    else:
        raw_df = _load_hf_dataframe(args.hf_dataset)
        filt_df = _filter_dataframe_like_1dcnn_before_split(raw_df)
        train_df, val_df, test_df = _one_dcnn_train_val_test(filt_df, random_state=42)
        if args.eval_split == "train":
            eval_df = train_df
        elif args.eval_split == "val":
            eval_df = val_df
        else:
            eval_df = test_df

        n_before = len(eval_df)
        X_test, y_test, skipped = _featurize_halt_from_dataframe(eval_df)
        skipped_summary = (
            ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items()) if v > 0) or "none"
        )
        print(
            f"1DCNN-aligned split — train/val/test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}. "
            f"Eval `{args.eval_split}`: {n_before} rows before HALT featurization, {len(X_test)} kept."
        )
        if skipped:
            print(f"  Skipped in eval subset: {skipped_summary}")

        split_meta = {
            "scheme": "1dcnn",
            "eval_split": args.eval_split,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "eval_rows_before_feat": n_before,
            "eval_rows_featurized": len(X_test),
            "skipped_summary": skipped_summary,
        }

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
    brier, accuracy, calib, sharp = evaluate_model(model, test_loader, device)
    print(
        "Evaluation complete. "
        f"Brier score: {brier:.4f}, Accuracy: {accuracy:.4f}, ECE: {calib['ece']:.4f}, "
        f"Sharpness |p-0.5|: {sharp['mean_confidence_distance']:.4f}"
    )

    write_markdown_report(
        report_path,
        brier=brier,
        accuracy=accuracy,
        ece=calib["ece"],
        mce=calib["mce"],
        sharpness_confidence_distance=sharp["mean_confidence_distance"],
        sharpness_predictive_variance=sharp["mean_predictive_variance"],
        calibration_bins=calib["bins"],
        n_samples=len(X_test),
        model_path=model_path,
        hf_dataset=args.hf_dataset,
        device=device,
        args=args,
        split_meta=split_meta,
    )
    print(f"Wrote markdown report to {_repo_rel(report_path)}")

if __name__ == '__main__':
    main()
