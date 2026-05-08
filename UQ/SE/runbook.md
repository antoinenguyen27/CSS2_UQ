# Semantic Entropy Runbook

This runbook covers the `UQ/SE` semantic-entropy baseline for **MMLU** on Modal. It uses the same cached Gemma 3 12B snapshot and shared `mmlu-trace-volume` volume layout as `data_work/mmlu_trace_eval`, but runs new stochastic sampling for uncertainty estimation.

## Prerequisites

1. Install the local Modal CLI dependency:

```bash
python -m pip install -r UQ/SE/requirements.txt
```

2. Authenticate the Modal CLI:

```bash
modal setup
```

3. Ensure the shared Hugging Face secret exists and has Gemma access:

```bash
modal secret create hf-auth HF_TOKEN=your_hf_token_here
```

4. Ensure the MMLU asset-preparation pipeline has already populated the shared Modal volume with `asset_metadata.json` and the cached model snapshot. If those assets are missing, bootstrap them from `data_work/`:

```bash
cd data_work
python -m pip install -e .
modal run -m mmlu_trace_eval.modal_app --split validation --limit 1
```

If `/vol/asset_metadata.json` is missing, `UQ/SE` will fail fast with an explicit error instead of re-downloading the model.

## Smoke Test

Run a small end-to-end Modal smoke test on 32 valid questions:

```bash
modal run UQ/SE/modal_se.py --eval-mode full --limit 32
```

This uses the reduced `max_num_seqs=64` startup profile so cold-start validation is safer before larger runs.

## Common Runs

Evaluate on all valid rows:

```bash
modal run UQ/SE/modal_se.py --eval-mode full
```

Evaluate with the 70/15/15 split for 1DCNN comparison. This executes only the held-out `test` partition:

```bash
modal run UQ/SE/modal_se.py --eval-mode split
```

Debug a larger but still bounded run:

```bash
modal run UQ/SE/modal_se.py --eval-mode split --limit 256 --n-samples 10
```

Override sampling settings explicitly:

```bash
modal run UQ/SE/modal_se.py -- \
  --eval-mode full \
  --n-samples 10 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-tokens 512 \
  --seed 42
```

To pin an explicit run id instead of using the auto-generated one:

```bash
modal run UQ/SE/modal_se.py -- --eval-mode full --run-name my-se-run
```

## Outputs

Every run writes to the shared Modal volume under:

```text
/vol/runs/<run_id>/
```

Artifacts:

- `semantic_entropy.parquet`: per-question results with counts, probabilities, entropy, certainty, and drop metadata
- `semantic_entropy.csv`: CSV copy of the same per-question results
- `summary.json`: aggregate metrics and baseline comparisons
- `manifest.json`: run configuration, package versions, and artifact paths

## Downloading Results

Replace `<run_id>` with the ID returned by the remote function:

```bash
modal volume get mmlu-trace-volume runs/<run_id>/semantic_entropy.parquet ./semantic_entropy.parquet
modal volume get mmlu-trace-volume runs/<run_id>/summary.json ./semantic_entropy_summary.json
modal volume get mmlu-trace-volume runs/<run_id>/manifest.json ./semantic_entropy_manifest.json
```

## Notes

- This pipeline is **MMLU-only**. It does not touch `mmlu_pro_trace_eval` or the MMLU-Pro dataset.
- Rows are filtered up front using the same validity logic as `UQ/halt/preprocessing/preprocess_halt.py`, with additional checks that `choices` has exactly four entries and `gold_answer` is one of `A-D`.
- The sampler uses one prompt per sample with `n=1` and deterministic distinct seeds. That preserves independent draws while still using vLLM batching in each Modal sub-batch.
- Questions with fewer than 5 valid parsed samples after retries are kept in the output with `drop_reason="fewer_than_min_valid_samples"` and excluded from the aggregate metrics.
