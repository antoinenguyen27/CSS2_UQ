import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tune HALT-Pro hyperparameters with Optuna (TPE sampler), optimizing a weighted "
            "combination of val_brier and val_ece from train_halt_pro."
        )
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--hf-dataset", type=str, default="auhsoJ69/mmlu-pro-traces")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--brier-weight", type=float, default=0.7)
    p.add_argument("--ece-weight", type=float, default=0.3)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("UQ/halt_pro/artifacts/optuna"),
        help="Output root for checkpoints, summaries, and study JSON.",
    )
    p.add_argument("--sampler-seed", type=int, default=42)
    return p.parse_args()


def _trial_name(number: int) -> str:
    return f"trial_{number:03d}"


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.root.resolve() / stamp
    ckpt_root = root / "checkpoints"
    tb_root = root / "runs"
    metrics_root = root / "metrics"
    for p in (ckpt_root, tb_root, metrics_root):
        p.mkdir(parents=True, exist_ok=True)

    print(f"Optuna output root: {root}")

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # Enqueue best known params from first 8-trial search (Brier 0.2209)
    study.enqueue_trial({
        "lr": 0.00023951255499127385,
        "dropout": 0.27818484499495255,
        "hidden_size": 256,
        "num_layers": 3,
        "top_q": 0.1,
        "weight_decay": 1e-6,
        "batch_size": 32
    })

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-5, 8e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.4)
        hidden_size = trial.suggest_categorical("hidden_size", [128, 192, 256, 384])
        num_layers = trial.suggest_int("num_layers", 2, 5)
        top_q = trial.suggest_float("top_q", 0.05, 0.25)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        batch_size = 16  # Reduced for 4GB GPU

        run_name = _trial_name(trial.number)
        ckpt = ckpt_root / f"{run_name}.pth"
        tb = tb_root / run_name
        metrics = metrics_root / f"{run_name}.json"
        tb.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            "-m",
            "UQ.halt_pro.training.train_halt_pro",
            "--hf-dataset",
            args.hf_dataset,
            "--seed",
            str(args.seed),
            "--batch-size",
            str(batch_size),
            "--val-split",
            str(args.val_split),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--lr",
            str(lr),
            "--dropout",
            str(dropout),
            "--hidden-size",
            str(hidden_size),
            "--num-layers",
            str(num_layers),
            "--top-q",
            str(top_q),
            "--weight-decay",
            str(weight_decay),
            "--checkpoint",
            str(ckpt),
            "--tensorboard-dir",
            str(tb),
            "--metrics-output",
            str(metrics),
        ]

        print(f"\n[trial {trial.number + 1}/{args.trials}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        payload = json.loads(metrics.read_text(encoding="utf-8"))
        val_brier = float(payload["best_val_brier"])
        val_ece = float(payload["best_val_ece"])
        val_f1 = float(payload["best_val_f1"])
        score = args.brier_weight * val_brier + args.ece_weight * val_ece

        trial.set_user_attr("best_epoch", int(payload["best_epoch"]))
        trial.set_user_attr("best_val_brier", val_brier)
        trial.set_user_attr("best_val_ece", val_ece)
        trial.set_user_attr("best_val_f1", val_f1)
        trial.set_user_attr("checkpoint", payload["checkpoint"])
        trial.set_user_attr("tensorboard_dir", payload["tensorboard_dir"])
        trial.set_user_attr("metrics_json", str(metrics))

        print(
            f"trial={trial.number} score={score:.6f} "
            f"(brier={val_brier:.6f}, ece={val_ece:.6f}, f1={val_f1:.6f})"
        )
        return score

    study.optimize(objective, n_trials=args.trials)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hf_dataset": args.hf_dataset,
        "objective": {
            "type": "weighted_sum",
            "brier_weight": args.brier_weight,
            "ece_weight": args.ece_weight,
        },
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in sorted(study.trials, key=lambda x: (float("inf") if x.value is None else x.value))
        ],
    }
    out = root / "study_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest trial:")
    print(json.dumps(summary["best_params"], indent=2))
    print(f"Best score: {study.best_value:.6f}")
    print(f"Wrote study summary to {out}")


if __name__ == "__main__":
    main()

