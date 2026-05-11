# MMLU-Pro Semantic Entropy Runbook

This runbook covers the `UQ/SE_PRO` semantic-entropy baseline for MMLU-Pro on Modal. The command semantics intentionally match `UQ/SE` for MMLU; only the module path changes.

## Dataset And Volume

- Hugging Face dataset: `auhsoJ69/mmlu-pro-traces`
- Data file: `examples.parquet`
- Modal app: `uq-semantic-entropy-mmlu-pro`
- Modal volume: `mmlu-pro-trace-volume`

The loader always passes `data_files="examples.parquet"` so the token-step parquet is not loaded as a 7M-row training table.

## Prerequisites

Install the local Modal CLI dependency:

```bash
python -m pip install -r UQ/SE_PRO/requirements.txt
```

Ensure Modal is authenticated and the shared Hugging Face secret exists:

```bash
modal setup
modal secret create hf-auth HF_TOKEN=your_hf_token_here
```

Ensure the MMLU-Pro asset-preparation pipeline has populated `mmlu-pro-trace-volume` with `/vol/asset_metadata.json` and the cached Gemma snapshot. If those assets are missing, bootstrap them from `data_work/`:

```bash
cd data_work
python -m pip install -e .
modal run -m mmlu_pro_trace_eval.modal_app --split validation --limit 1
```

If `/vol/asset_metadata.json` is missing, `UQ/SE_PRO` fails fast instead of silently downloading a separate model snapshot.

## Run Commands

Smoke test:

```bash
modal run UQ/SE_PRO/modal_se_pro.py --eval-mode split --num-eval-rows 32 --n-samples 10
```

Budget pilot:

```bash
modal run UQ/SE_PRO/modal_se_pro.py --eval-mode split --num-eval-rows 154 --n-samples 10
```

Full held-out split:

```bash
modal run UQ/SE_PRO/modal_se_pro.py --eval-mode split
```

Full valid source table:

```bash
modal run UQ/SE_PRO/modal_se_pro.py --eval-mode full
```

## Semantics

- `--eval-mode split` builds the same 70/15/15 split as the MMLU SE path and evaluates only the held-out `test` partition.
- `--num-eval-rows N` caps the actual selected eval partition to `N` rows. `0` means uncapped.
- `--n-samples 10` means ten stochastic generations per selected question before retries.

MMLU-Pro rows can have 3 to 10 answer choices. Entropy is normalized by the per-question choice count, not by a global fixed K.

## Outputs

Every run writes to:

```text
/vol/runs/<run_id>/
```

Artifacts:

- `semantic_entropy.parquet`: per-question counts, probabilities, entropy, certainty, and drop metadata
- `semantic_entropy.csv`: CSV copy of the per-question results
- `summary.json`: aggregate metrics and baseline comparisons
- `manifest.json`: run configuration, package versions, and artifact paths

Download results with:

```bash
modal volume get mmlu-pro-trace-volume runs/<run_id>/semantic_entropy.parquet ./mmlu_pro_semantic_entropy.parquet
modal volume get mmlu-pro-trace-volume runs/<run_id>/summary.json ./mmlu_pro_semantic_entropy_summary.json
modal volume get mmlu-pro-trace-volume runs/<run_id>/manifest.json ./mmlu_pro_semantic_entropy_manifest.json
```
