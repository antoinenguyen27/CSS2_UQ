# Dataset Creation Runbook

This runbook covers both `mmlu_trace_eval` (cais/mmlu, 4 choices) and `mmlu_pro_trace_eval` (TIGER-Lab/MMLU-Pro, 10 choices). Both pipelines produce identically-shaped output Parquet files and run on Modal with a single H200 GPU.

---

## Prerequisites

These steps are one-time setup. Skip anything already done.

**1. Install Modal**

```bash
python -m pip install -U modal
```

**2. Authenticate Modal CLI**

```bash
modal setup
```

**3. Create the HuggingFace secret**

Your HF token must have Gemma 3 access approved at huggingface.co/google/gemma-3-12b-it.

```bash
modal secret create hf-auth HF_TOKEN=your_hf_token_here
```

This secret is shared by both pipelines (`SECRET_NAME = "hf-auth"` in both configs).

**4. Install the local package (for local test runs)**

```bash
cd data_work
pip install -e .
```

---

## MMLU (`cais/mmlu`, 4 choices)

**Modal app:** `mmlu-trace-eval`  
**Volume:** `mmlu-trace-volume`  
**Splits:** `test`, `validation`, `dev`

### Smoke test — validation split, 64 examples

Runs on the `validation` split. vLLM initialises with `max_num_seqs=64` (reduced profile). Use this to confirm the pipeline works end-to-end before committing to a full run.

```bash
modal run -m mmlu_trace_eval.modal_app --split validation --limit 64
```

Expected duration: ~5–10 minutes including cold start.

### Smoke test — dev split

```bash
modal run -m mmlu_trace_eval.modal_app --split dev --limit 64
```

### Full test run (detached)

Runs the entire `test` split (~14,000 examples). Always use `--detach` so a local terminal disconnect does not abort the run.

```bash
modal run --detach -m mmlu_trace_eval.modal_app --split test
```

The run ID is printed at launch. Keep it — you need it to download artifacts.

### Resume an interrupted run

Pass the same `--run-name` used in the original invocation. The pipeline reads completed shard metadata and skips already-processed examples.

```bash
modal run --detach -m mmlu_trace_eval.modal_app \
  --split test \
  --run-name gemma3-12b-it__mmlu-test__reasoning_v1__<original_timestamp>
```

### Run a single subject (debugging)

```bash
modal run -m mmlu_trace_eval.modal_app --split test --subject "high_school_mathematics" --limit 50
```

### Download artifacts

Replace `<run_id>` with the ID printed at launch (also visible in `manifest.json`).

```bash
modal volume get mmlu-trace-volume runs/<run_id>/examples.parquet     ./mmlu_examples.parquet
modal volume get mmlu-trace-volume runs/<run_id>/token_steps.parquet  ./mmlu_token_steps.parquet
modal volume get mmlu-trace-volume runs/<run_id>/manifest.json        ./mmlu_manifest.json
modal volume get mmlu-trace-volume runs/<run_id>/metrics.json         ./mmlu_metrics.json
```

### List runs on the volume

```bash
modal volume ls mmlu-trace-volume runs/
```

---

## MMLU Pro (`TIGER-Lab/MMLU-Pro`, 10 choices)

**Modal app:** `mmlu-pro-trace-eval`  
**Volume:** `mmlu-pro-trace-volume`  
**Splits:** `test`, `validation` (no `dev` split)

The commands are identical in structure to MMLU — swap the module name and volume name.

### Smoke test — validation split, 32 examples

```bash
modal run -m mmlu_pro_trace_eval.modal_app --split validation --limit 32
```

vLLM initialises with `max_num_seqs=64`. Expected duration: ~5–10 minutes.

### Full test run (detached)

```bash
modal run --detach -m mmlu_pro_trace_eval.modal_app --split test
```

### Resume an interrupted run

```bash
modal run --detach -m mmlu_pro_trace_eval.modal_app \
  --split test \
  --run-name gemma3-12b-it__mmlu-pro-test__reasoning_v1__<original_timestamp>
```

### Run a single category (debugging)

MMLU Pro uses `category` internally (mapped to `subject` in our schema). Pass it the same way:

```bash
modal run -m mmlu_pro_trace_eval.modal_app --split test --subject "math" --limit 50
```

### Download artifacts

```bash
modal volume get mmlu-pro-trace-volume runs/<run_id>/examples.parquet     ./pro_examples.parquet
modal volume get mmlu-pro-trace-volume runs/<run_id>/token_steps.parquet  ./pro_token_steps.parquet
modal volume get mmlu-pro-trace-volume runs/<run_id>/manifest.json        ./pro_manifest.json
modal volume get mmlu-pro-trace-volume runs/<run_id>/metrics.json         ./pro_metrics.json
```

### List runs on the volume

```bash
modal volume ls mmlu-pro-trace-volume runs/
```

---

## CLI flag reference

Both `modal_app.py` entrypoints accept the same flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--split` | str | `"test"` | Dataset split to evaluate |
| `--limit` | int | `0` (no limit) | Cap on number of examples; `0` means all |
| `--subject` | str | `""` (all) | Filter to a single subject/category |
| `--run-name` | str | `""` (auto) | Explicit run ID for resumption; auto-generated if omitted |
| `--resume` | bool | `true` | Skip already-completed examples when resuming |

---

## Config reference

Key inference constants. Edit in `config.py` to change behaviour — each change produces a new dataset version (the `prompt_hash` and `decode_config_json` in the manifest will differ).

| Constant | MMLU | MMLU Pro | Description |
|----------|------|----------|-------------|
| `MAX_OUTPUT_TOKENS` | 192 | 512 | Generation cap per example |
| `MAX_MODEL_LEN` | 2048 | 3072 | vLLM context window |
| `MAX_NUM_SEQS` | 224 | 224 | Max concurrent sequences (full run) |
| `SMOKE_MAX_NUM_SEQS` | 64 | 64 | Max concurrent sequences (validation/dev) |
| `MAX_NUM_BATCHED_TOKENS` | 98,304 | 98,304 | Token budget per vLLM batch |
| `SHARD_SIZE` | 256 | 256 | Examples per shard file |
| `MAX_TOP_LOGPROBS` | 20 | 20 | Top-k logprob candidates stored per token |

Decoding is greedy (`temperature=0.0`, `seed=0`) and is not configurable from the CLI — change `DecodeConfig` in `config.py` if needed.

---

## Volume layout

Both pipelines write to their own Modal volumes but **share the same model snapshot** path convention (same `MODEL_ID`, same content-addressed revision directory). The first pipeline to run will download the ~25 GB model; the second reuses it if run on the same volume — but since the volumes are separate (`mmlu-trace-volume` vs `mmlu-pro-trace-volume`), each volume holds its own model copy.

```
/vol/
  models/
    google__gemma-3-12b-it/
      <model_revision_sha>/        # model weights
  datasets/
    <DATASET_ID__CONFIG>/
      <dataset_revision_sha>/
        test.parquet
        validation.parquet
        .ready
  cache/
    huggingface/
    datasets/
    vllm/
  runs/
    <run_id>/
      manifest.json
      metrics.json
      examples.parquet
      token_steps.parquet
      shards/
        examples-00000.parquet
        token_steps-00000.parquet
        meta-00000.json
        ...
  asset_metadata.json        # MMLU only
  asset_metadata_pro.json    # MMLU Pro only
```

---

## Checking run status

Print the manifest of a live or completed run:

```bash
# MMLU
modal volume get mmlu-trace-volume runs/<run_id>/manifest.json - | python -m json.tool

# MMLU Pro
modal volume get mmlu-pro-trace-volume runs/<run_id>/manifest.json - | python -m json.tool
```

Print metrics:

```bash
modal volume get mmlu-trace-volume runs/<run_id>/metrics.json - | python -m json.tool
```

The `finished_at_utc` field in `manifest.json` is `null` while a run is in progress and set to a timestamp when complete.
