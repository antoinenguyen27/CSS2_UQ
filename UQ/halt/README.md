# HALT (UQ)

This folder contains the **HALT** sequence model for binary correctness / uncertainty-style prediction from token-level features. Training and evaluation scripts live under `training/` and `evaluation/`.

## Setup

From the **repository root** (`CSS2_UQ`), install dependencies:

```bash
python -m pip install -r UQ/halt/requirements.txt
```

Install the project in editable mode so `UQ` package imports resolve:

```bash
python -m pip install -e .
```

If you are using a fresh environment, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

All commands below assume your **current working directory is the repo root** so imports resolve (`python -m UQ.halt...`).

## Train HALT

Run training as a module (recommended):

```bash
python -m UQ.halt.training.train_halt
```

Show all options:

```bash
python -m UQ.halt.training.train_halt --help
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

Training prints the TensorBoard log directory as a **path relative to the repo root**. By default, runs are under `UQ/halt/artifacts/runs/`. From the repo root:

```bash
tensorboard --logdir=UQ/halt/artifacts/runs
```

## Evaluate HALT

Evaluation loads a checkpoint, runs the full preprocessed dataset through the model, reports **Brier score** and **accuracy**, and writes a **Markdown report** (checkpoint path in the report is repo-relative).

```bash
python -m UQ.halt.evaluation.evaluate_halt
```

```bash
python -m UQ.halt.evaluation.evaluate_halt --help
```

### Useful arguments

| Argument | Role |
| --- | --- |
| `--checkpoint` | Weights file (default: best checkpoint under Artifacts). |
| `--hf-dataset` | Same preprocessing dataset id as training (must match how you trained). |
| `--output` | Markdown report path (default: timestamped file under `UQ/halt/artifacts/evaluation/`). |
| `--device` | `auto`, `cpu`, or `cuda`. |
| Architecture | Same flags as training; they must match the trained model. |

The evaluation script prints the Brier score, accuracy, and the path to the generated `.md` report (repo-relative when possible).

## Demo HALT

Run a quick demo on a small subset of examples (prints per-sample **UQ confidence**, gold `is_correct` label, and sequence length). By default **`--sample-mode balanced`** picks the same number of **incorrect** (`is_correct=0`) and **correct** (`is_correct=1`) rows (up to `--num-examples // 2` per class, capped by how many exist). Odd `--num-examples` uses `n//2` per class (total even). Checkpoint and dataset lines are printed **relative to the repo root** when possible.

In the table, **`label`** is `is_correct` (0 = answer wrong, 1 = answer right). **`uq_conf`** is the model’s **estimated probability the answer is correct** (same as `predict_proba` elsewhere). Summary lines and the optional confusion block use **`--threshold`** on `uq_conf` only for accuracy-style diagnostics (default 0.5).

```bash
python -m UQ.halt.demo_halt
```

Examples:

```bash
python -m UQ.halt.demo_halt --num-examples 20
python -m UQ.halt.demo_halt --num-examples 20 --sample-mode random --seed 7
```

## Data and splits

- **`preprocessing/preprocess_halt.py`** loads and featurizes the Hugging Face dataset and returns **all** valid rows (no train/val/test split inside preprocessing).
- **Training** applies a stratified **train/validation** split (`--val-split`).
- **Evaluation** currently preprocesses the **full** dataset again and scores every example; it does **not** automatically use a held-out test split. For a true test set, you would extend preprocessing or evaluation to filter splits (for example by id or by a separate file).

## Artifacts (default paths)

Generated files live under **`UQ/halt/artifacts/`**. Checkpoints and TensorBoard runs are gitignored; you can commit evaluation markdown under `UQ/halt/artifacts/evaluation/` if you choose.

| Path | Contents |
| --- | --- |
| `UQ/halt/artifacts/checkpoints/best_halt_model.pth` | Best checkpoint (by validation Brier) unless `--checkpoint` overrides. |
| `UQ/halt/artifacts/runs/<timestamp>_<comment>/` | TensorBoard events (default run layout). |
| `UQ/halt/artifacts/evaluation/halt_eval_<timestamp>.md` | Evaluation Markdown report (unless `--output` is set). |

## Running scripts by file path

If you prefer running the `.py` file directly, set `PYTHONPATH` to the repo root so `import UQ` works:

```bash
PYTHONPATH=. python UQ/halt/training/train_halt.py
```

Using `python -m UQ.halt.training.train_halt` from the repo root avoids this.
