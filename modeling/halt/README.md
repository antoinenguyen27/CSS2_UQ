# Modeling (HALT)

This folder contains the **HALT** sequence model for binary correctness / uncertainty-style prediction from token-level features. Training and evaluation scripts live under `training/` and `evaluation/`.

## Setup

From the **repository root** (`CSS2_UQ`), install dependencies from `modeling/halt/requirements.txt`:

```bash
# Linux / macOS
python -m pip install -r modeling/halt/requirements.txt

# Windows PowerShell
python -m pip install -r modeling/halt/requirements.txt
```

Install the project in editable mode so `modeling` imports work cleanly everywhere:

```bash
python -m pip install -e .
```

If you are using a fresh environment, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

All commands below assume your **current working directory is the repo root** so imports resolve (`python -m modeling...`).

## Train HALT

Run training as a module (recommended):

```bash
python -m modeling.halt.training.train_halt
```

Show all options:

```bash
python -m modeling.halt.training.train_halt --help
```

### Common arguments

| Argument | Role |
| --- | --- |
| `--batch-size` | Minibatch size (default: 32). |
| `--lr` / `--learning-rate` | Adam learning rate (default: `4.41e-4`). |
| `--epochs` | Maximum epochs (default: 100). |
| `--patience` | Early stopping: epochs without improvement on **validation Brier** (default: 10). |
| `--val-split` | Fraction of data for validation (default: 0.2). |
| `--seed` | RNG seed for splits and training (default: 42). |
| `--checkpoint` | Where to save the best weights (default: see Artifacts). |
| `--hf-dataset` | Hugging Face dataset id passed to preprocessing (default: `antoine3/CSS2_UQ`). |
| `--device` | `auto`, `cpu`, or `cuda`. |
| `--tensorboard-dir` | Override TensorBoard log directory (default: timestamped folder under Artifacts). |
| `--tensorboard-comment` | Suffix used in the default run folder name. |
| Architecture | `--input-dim`, `--proj-dim`, `--hidden-size`, `--num-layers`, `--dropout`, `--top-q` (defaults match the current HALT setup in code). |

### Training objective and selection

- **Loss optimized:** weighted `BCEWithLogitsLoss` (binary cross-entropy on logits).
- **Checkpoint and early stopping:** driven by **validation Brier score** (lower is better), along with learning-rate scheduling on that metric.

### TensorBoard

Training prints the event directory on startup. By default, runs are under `modeling/halt/artifacts/runs/`. From the repo root:

```bash
tensorboard --logdir=modeling/halt/artifacts/runs
```

## Evaluate HALT

Evaluation loads a checkpoint, runs the full preprocessed dataset through the model, reports **Brier score**, and writes a **Markdown report**.

```bash
python -m modeling.halt.evaluation.evaluate_halt
```

```bash
python -m modeling.halt.evaluation.evaluate_halt --help
```

### Useful arguments

| Argument | Role |
| --- | --- |
| `--checkpoint` | Weights file (default: best checkpoint under Artifacts). |
| `--hf-dataset` | Same preprocessing dataset id as training (must match how you trained). |
| `--output` | Markdown report path (default: timestamped file under `modeling/halt/artifacts/evaluation/`). |
| `--device` | `auto`, `cpu`, or `cuda`. |
| Architecture | Same flags as training; they must match the trained model. |

The evaluation script prints the Brier score and the path to the generated `.md` report.

## Data and splits

- **`preprocessing/preprocess_halt.py`** loads and featurizes the Hugging Face dataset and returns **all** valid rows (no train/val/test split inside preprocessing).
- **Training** applies a stratified **train/validation** split (`--val-split`).
- **Evaluation** currently preprocesses the **full** dataset again and scores every example; it does **not** automatically use a held-out test split. For a true test set, you would extend preprocessing or evaluation to filter splits (for example by id or by a separate file).

## Artifacts (default paths)

Generated files are kept under **`modeling/halt/artifacts/`** (gitignored):

| Path | Contents |
| --- | --- |
| `artifacts/checkpoints/best_halt_model.pth` | Best checkpoint (by validation Brier) unless `--checkpoint` overrides. |
| `artifacts/runs/<timestamp>_<comment>/` | TensorBoard events (default run layout). |
| `artifacts/evaluation/halt_eval_<timestamp>.md` | Evaluation Markdown report (unless `--output` is set). |

## Running scripts by file path

If you prefer `python modeling/halt/training/train_halt.py`, set `PYTHONPATH` to the repo root so `import modeling` works, for example:

```bash
# Linux / macOS
PYTHONPATH=. python modeling/halt/training/train_halt.py

# Windows PowerShell
$env:PYTHONPATH = "."
python modeling/halt/training/train_halt.py
```

Using `python -m modeling.halt.training.train_halt` from the repo root avoids this.
