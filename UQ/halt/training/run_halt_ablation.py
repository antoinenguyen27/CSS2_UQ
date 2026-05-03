import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AblationConfig:
    name: str
    lr: float
    dropout: float
    hidden_size: int
    num_layers: int
    top_q: float
    epochs: int
    patience: int


GRID = [
    AblationConfig(
        name="baseline",
        lr=4.41e-4,
        dropout=0.4,
        hidden_size=256,
        num_layers=5,
        top_q=0.15,
        epochs=30,
        patience=6,
    ),
    AblationConfig(
        name="lower_lr",
        lr=2.0e-4,
        dropout=0.4,
        hidden_size=256,
        num_layers=5,
        top_q=0.15,
        epochs=30,
        patience=6,
    ),
    AblationConfig(
        name="smaller_model",
        lr=2.0e-4,
        dropout=0.35,
        hidden_size=192,
        num_layers=4,
        top_q=0.15,
        epochs=30,
        patience=6,
    ),
    AblationConfig(
        name="more_pooling",
        lr=2.0e-4,
        dropout=0.35,
        hidden_size=192,
        num_layers=4,
        top_q=0.25,
        epochs=30,
        patience=6,
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a small HALT ablation grid. "
            "Each run writes a separate checkpoint and TensorBoard log dir."
        )
    )
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    p.add_argument("--hf-dataset", type=str, default="auhsoJ69/mmlu_rerun")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("UQ/halt/artifacts/ablations"),
        help="Output root for checkpoints and logs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.root.resolve() / stamp
    print(f"Ablation output root: {root}")

    for i, cfg in enumerate(GRID, start=1):
        run_name = f"{i:02d}_{cfg.name}"
        ckpt = root / "checkpoints" / f"{run_name}.pth"
        tb = root / "runs" / run_name
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tb.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            "-m",
            "UQ.halt.training.train_halt",
            "--hf-dataset",
            args.hf_dataset,
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--val-split",
            str(args.val_split),
            "--lr",
            str(cfg.lr),
            "--dropout",
            str(cfg.dropout),
            "--hidden-size",
            str(cfg.hidden_size),
            "--num-layers",
            str(cfg.num_layers),
            "--top-q",
            str(cfg.top_q),
            "--epochs",
            str(cfg.epochs),
            "--patience",
            str(cfg.patience),
            "--checkpoint",
            str(ckpt),
            "--tensorboard-dir",
            str(tb),
        ]

        print(f"\n[{i}/{len(GRID)}] {run_name}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print("\nDone. Compare runs in TensorBoard and choose best by lowest val/brier with val/ece as guard.")


if __name__ == "__main__":
    main()

