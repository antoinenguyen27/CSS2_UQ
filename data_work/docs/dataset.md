# Dataset Specification

This document defines the canonical output schema for the reasoning-prompted MMLU trace dataset.

## Dataset Identity

Each dataset version is defined by the full generation regime:

- model id and revision
- tokenizer revision
- dataset id, config, and revision
- prompt variant and prompt text hash
- decode configuration
- split and any subject filter

Changing any of the above means creating a new dataset version.

## Run Layout

Each run is written under:

```text
/vol/runs/<run_id>/
```

Required artifacts:

- `manifest.json`
- `metrics.json`
- `examples.parquet`
- `token_steps.parquet`
- immutable shard files under `shards/`

## `examples.parquet`

One row per evaluated MMLU example.

### Required scalar columns

- `run_id`
- `example_id`
- `split`
- `subject`
- `question_idx`
- `question`
- `choices`
- `gold_answer`
- `predicted_answer`
- `parse_success`
- `parse_error`
- `is_correct`
- `model_name`
- `model_revision`
- `tokenizer_revision`
- `dataset_name`
- `dataset_revision`
- `prompt_variant`
- `prompt_text`
- `decode_config_json`
- `completion_text`
- `thinking_text`
- `answer_text`
- `trajectory_length`
- `thinking_token_count`
- `answer_token_count`
- `thinking_start_idx`
- `thinking_end_idx`
- `answer_start_idx`
- `answer_end_idx`

### Required sequence columns

All of these have logical length `T = trajectory_length`.

- `sampled_token_ids[T]`
- `sampled_token_texts[T]`
- `sampled_token_logprobs[T]`
- `sampled_token_ranks[T]`
- `segment_ids[T]`
- `segment_names[T]`
- `cumulative_logprobs[T]`

### Required top-k columns

All of these have logical shape `[T, 20]`.

- `top20_token_ids[T,20]`
- `top20_token_texts[T,20]`
- `top20_token_logprobs[T,20]`

## `token_steps.parquet`

One row per generated token step.

Required columns:

- `run_id`
- `example_id`
- `step_idx`
- `subject`
- `gold_answer`
- `predicted_answer`
- `is_correct`
- `segment_id`
- `segment_name`
- `sampled_token_id`
- `sampled_token_text`
- `sampled_token_logprob`
- `sampled_token_rank`
- `cumulative_logprob`
- `top20_token_ids`
- `top20_token_texts`
- `top20_token_logprobs`

## Invariants

- Only generated tokens are stored. Prompt tokens are excluded.
- Segment ids are fixed:
  - `0 = thinking`
  - `1 = answer`
  - `2 = structure`
- `structure` covers the literal tags and any stray text outside the semantic inner regions.
- The top-k width is always exactly 20.
- If fewer than 20 candidates are returned at a step, the row is padded with:
  - token id `-1`
  - token text `""`
  - logprob `-inf`
- Logprobs are raw model logprobs from vLLM and are not renormalized over the top-20 slice.
- Token id columns use `int32`, logprobs use `float32`, segment ids use `uint8`, and boolean targets use Arrow booleans.
- `sampled_token_texts` and `top20_token_texts` are tokenizer pieces from `convert_ids_to_tokens`, not cleaned detokenized spans.
- Under greedy decoding, `sampled_token_ids[t] == top20_token_ids[t][0]` must hold for every step.

## Parsing and Scoring Rules

- The scorer parses the first valid `<answer>[ABCD]</answer>` block.
- Whitespace and line breaks inside the answer tag are allowed.
- If no valid answer tag exists:
  - `predicted_answer = null`
  - `parse_success = false`
  - `parse_error = "missing_answer_tag"`
  - `is_correct = false`
- If multiple valid answer tags exist:
  - the first valid one is used
  - `parse_success = true`
  - `parse_error = "multiple_answer_tags"`
- No retries are performed for malformed outputs.

## Segment Boundaries

Segment boundaries are reported as token index ranges over the generated token sequence.

- `*_start_idx` is inclusive
- `*_end_idx` is exclusive
- absent regions use `-1` for both fields

That means the Python slice:

```python
tokens[start_idx:end_idx]
```

extracts the relevant segment when the indices are present.

## Manifest

`manifest.json` records:

- run id
- split and subject filter
- requested limit
- total example count
- asset metadata with model and dataset revisions
- prompt variant and prompt hash
- decode config JSON
- package versions
- shard inventory
- start and finish timestamps

## Metrics

`metrics.json` records:

- overall accuracy
- raw correct and total counts
- parse-failure count and rate
- mean and median trajectory length
- mean and median answer-region length
- per-subject accuracy summary
