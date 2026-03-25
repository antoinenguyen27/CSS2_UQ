# Reasoning-Prompted MMLU Trace Dataset on Modal

This repository builds a fixed, reproducible evaluation dataset for uncertainty-quantification research. It runs `google/gemma-3-12b-it` on `cais/mmlu` inside Modal, forces a structured visible reasoning trace, captures token-level top-20 logprobs for every generated step, and writes the resulting data into Parquet artifacts that are ready for downstream sequence models.

This repo is deliberately not a canonical MMLU benchmark harness.

## Purpose

The goal is to collect a static dataset for calibration and uncertainty-prediction work, so downstream models do not need to rerun the underlying LLM every time they train or recalibrate.

Each dataset row represents one realized generation:

- one prompt template
- one reasoning trace inside `<thinking>...</thinking>`
- one parsed final answer inside `<answer>X</answer>`
- one correctness label against the MMLU gold answer

The dataset therefore calibrates this exact prompted behavior, not the base model in the abstract.

## Why MMLU

MMLU is still useful here even though it is multiple-choice:

- it provides exact answer checking without a judge model
- it covers many subjects
- it allows long visible reasoning traces while still ending in a strict answer target

That makes it a practical choice for a small research team that wants rich token trajectories with unambiguous correctness labels.

## Why This Is Not Standard MMLU

Standard MMLU typically uses short answer-only prompting. This repo intentionally departs from that setup.

We force the model to generate structured visible reasoning because the downstream UQ task depends on longer token trajectories. The resulting score is therefore a property of:

- Gemma 3 12B
- the tokenizer revision
- the reasoning prompt template
- greedy decoding
- the MMLU split

If any of those change, you should treat the output as a different dataset version.

## Why Greedy Decoding Is Still Correct

Greedy decoding does not make logprobs useless. It gives you:

- a deterministic token trajectory
- the model's local uncertainty at every generated step
- stable calibration targets across reruns

That is the right tradeoff for an offline calibration dataset. You are measuring uncertainty along one deterministic realized trace, which is exactly what downstream models consume.

## Dataset Overview

The run produces two main Parquet artifacts under the Modal Volume:

- `examples.parquet`: one row per MMLU item, with the full token trajectory attached
- `token_steps.parquet`: one row per generated token step, for inspection and analytics

Key fields include:

- full prompt text
- completion text
- parsed answer and correctness
- per-token sampled ids, token pieces, logprobs, ranks
- per-token top-20 candidate ids, token pieces, and logprobs
- segment labels distinguishing `thinking`, `answer`, and `structure`

See [docs/dataset.md](/Users/an/Documents/CSS2_UQ/docs/dataset.md) for the exact schema and invariants.

## Compute and Runtime

The default Modal setup is:

- `1x H200`
- `6 CPU` cores
- `64 GiB` requested host memory for the evaluator container
- `32 GiB` requested host memory for the asset-preparation container
- `vLLM 0.18.0`
- `Modal 1.3.5`
- Python `3.11`

Reasoning traces are longer than standard MMLU completions, so the evaluator uses adaptive microbatching:

- hard ceiling: `224` sequences
- token budget ceiling: `98,304` prompt-plus-generation tokens
- generation cap: `192` tokens

This keeps throughput high without assuming that every prompt fits safely into a fixed `224`-way batch.

For startup safety, the smoke path uses a reduced initialization profile:

- `validation` and `dev` runs initialize vLLM with `max_num_seqs=64`
- the main `test` run keeps `max_num_seqs=224`
- vLLM is started with `enforce_eager=True` to avoid CUDA-graph cold-start issues

## Quickstart

# NOTICE: ANTOINE HAS ALREADY RUN THE EVAL AND THE DATASET
Access the dataset at: [huggingface](https://huggingface.co/datasets/antoine3/CSS2_UQ/tree/main).
Follow the instructions below at "Using this Dataset" to learn how it can be used. 

1. Install the Modal client locally:

```bash
python -m pip install -U modal
```

2. Authenticate your Modal CLI:

```bash
modal setup
```

3. Create a Modal secret with a Hugging Face token that already has Gemma 3 access:

```bash
modal secret create hf-auth HF_TOKEN=your_hf_token_here
```

4. Run a small smoke test:

```bash
modal run -m mmlu_trace_eval.modal_app --split validation --limit 64
```

5. Launch the full test split:

```bash
modal run --detach -m mmlu_trace_eval.modal_app --split test
```

6. Download artifacts from the Volume:

```bash
modal volume get mmlu-trace-volume runs/<run_id>/examples.parquet ./examples.parquet
modal volume get mmlu-trace-volume runs/<run_id>/token_steps.parquet ./token_steps.parquet
modal volume get mmlu-trace-volume runs/<run_id>/manifest.json ./manifest.json
```

## Repo Layout

- [`mmlu_trace_eval/modal_app.py`](/Users/an/Documents/CSS2_UQ/mmlu_trace_eval/modal_app.py): Modal app, asset preparation, evaluator class, and local entrypoint
- [`docs/dataset.md`](/Users/an/Documents/CSS2_UQ/docs/dataset.md): dataset schema and invariants
- [`docs/modeling_lstm.md`](/Users/an/Documents/CSS2_UQ/docs/modeling_lstm.md): how to consume the output for sequence models
- [`examples/load_lstm_example.py`](/Users/an/Documents/CSS2_UQ/examples/load_lstm_example.py): minimal example for loading one training row

## Using the Dataset

The canonical training artifact is `examples.parquet`. Each row is one full sequence example with one binary target `is_correct`.

That means a downstream sequence model can load one row and immediately obtain:

- `T`: trajectory length
- per-step features such as `top20_token_logprobs[T, 20]`
- optional token identity features such as `sampled_token_ids[T]`
- one scalar target `is_correct`

This is the correct shape for:

- LSTMs
- GRUs
- temporal CNNs
- transformers over token-time features
- hand-engineered trajectory-level calibrators

For a concrete LSTM-oriented walkthrough, see [docs/modeling_lstm.md](/Users/an/Documents/CSS2_UQ/docs/modeling_lstm.md).

## Notes for Teammates

- The first run is slower because the evaluator downloads both the model snapshot and the normalized MMLU source parquet files into the Modal Volume.
- Later runs reuse the cached assets.
- The evaluator now emits explicit `tokenizer_load_*` and `vllm_init_*` log events so you can distinguish “still cold-starting” from “actually hung in startup”.
- The evaluator writes immutable shard files and can resume an interrupted run when you pass the same `--run-name` with `--resume true`.
- This repo stores debug-rich artifacts on purpose, so output size is materially larger than a numeric-only dataset.

## Primary References

- [Modal user account setup](https://frontend.modal.com/docs/guide/modal-user-account-setup)
- [Modal run CLI](https://modal.com/docs/reference/cli/run)
- [Modal GPU guide](https://modal.com/docs/guide/gpu)
- [Modal Volumes guide](https://modal.com/docs/guide/volumes)
- [Modal model weights guide](https://modal.com/docs/guide/model-weights)
- [Modal Secrets guide](https://modal.com/docs/guide/secrets)
- [Modal changelog](https://modal.com/docs/reference/changelog)
- [vLLM sampling params](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html)
- [vLLM supported models](https://docs.vllm.ai/en/stable/models/supported_models.html)
- [MMLU dataset card](https://huggingface.co/datasets/cais/mmlu)
- [Gemma 3 12B model card](https://huggingface.co/google/gemma-3-12b-it)
