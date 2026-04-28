# HALT-Pro (MMLU rerun)

This folder contains a standalone HALT pipeline for the MMLU rerun trace dataset:

- Preprocessing: `UQ/halt_pro/preprocessing/preprocess_halt_pro.py`
- Training: `UQ/halt_pro/training/train_halt_pro.py`
- Evaluation: `UQ/halt_pro/evaluation/evaluate_halt_pro.py`

Defaults are set to:

- `--hf-dataset auhsoJ69/mmlu_rerun` ([dataset card](https://huggingface.co/datasets/auhsoJ69/mmlu_rerun))
- `--data-file examples.parquet`

## Install

From repo root:

```bash
python -m pip install -r UQ/halt_pro/requirements.txt
python -m pip install -e .
```

## Train

```bash
python -m UQ.halt_pro.training.train_halt_pro
```

Useful flags:

- `--data-file examples.parquet` (default)
- `--checkpoint UQ/halt_pro/artifacts/checkpoints/best_halt_pro_model.pth`
- `--ece-tolerance 0.01 --brier-tolerance 0.002`

## Evaluate

```bash
python -m UQ.halt_pro.evaluation.evaluate_halt_pro
```

Useful flags:

- `--checkpoint UQ/halt_pro/artifacts/checkpoints/best_halt_pro_model.pth`
- `--output UQ/halt_pro/evaluation/halt_pro_eval_custom.md`

The evaluation report includes Brier, accuracy, ECE/MCE, sharpness, and reliability bins.

