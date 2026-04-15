# Using the Dataset for an LSTM UQ Model

This dataset is designed so that one row in `examples.parquet` is one supervised sequence example.

That is the key ergonomic property for sequence models:

- one MMLU item
- one realized token trajectory
- one binary target `is_correct`

## Why `examples.parquet` Is the Canonical Artifact

For training, you want to load a row and immediately get:

- `T`: the sequence length
- per-step features
- one scalar label

That maps directly onto recurrent and temporal models. The exploded `token_steps.parquet` view is still useful for debugging and plotting, but it should not be the primary training source.

## Minimal Feature Set

The smallest sensible LSTM baseline can use:

- `top20_token_logprobs[T, 20]`

At each time step, the model sees the logprob shape of the top-20 next-token distribution.

## Recommended Richer Feature Set

A stronger baseline should concatenate:

- `top20_token_logprobs[T, 20]`
- `sampled_token_logprobs[T, 1]`
- `segment_ids[T, 1]`
- optional normalized position feature `t / T`
- optional learned embedding for `sampled_token_ids[T]`

This produces a per-step feature matrix `X` of shape `[T, F]`.

## Supervision Target

Use:

- `y = is_correct`

This is a binary classification target: predict the probability that the final answer extracted from this exact trajectory is correct.

## Padding and Masks

Sequence lengths vary, so batches need padding.

Recommended batch outputs:

- `features`: `[B, T_max, F]`
- `lengths`: `[B]`
- `mask`: `[B, T_max]`
- `labels`: `[B]`

Use either:

- `pack_padded_sequence` / `pad_packed_sequence` for recurrent models, or
- explicit masking if you build your own recurrent loop or use attention later

## Example Tensor Construction

The script [`examples/load_lstm_example.py`](../examples/load_lstm_example.py) shows a minimal loading flow. The core logic is:

```python
import numpy as np
import pyarrow.parquet as pq

table = pq.read_table("examples.parquet")
row = table.slice(0, 1).to_pylist()[0]

top20 = np.asarray(row["top20_token_logprobs"], dtype=np.float32)
sampled = np.asarray(row["sampled_token_logprobs"], dtype=np.float32)[:, None]
segments = np.asarray(row["segment_ids"], dtype=np.float32)[:, None]

positions = np.arange(top20.shape[0], dtype=np.float32)[:, None]
positions /= max(top20.shape[0], 1)

features = np.concatenate([top20, sampled, segments, positions], axis=1)
label = np.asarray(row["is_correct"], dtype=np.float32)
```

This yields:

- `features.shape == [T, 23]`
- `label.shape == []`

## Suggested Baselines

Start with:

1. LSTM over `[top20_token_logprobs]`
2. LSTM over `[top20_token_logprobs, sampled_logprob, segment_id]`
3. LSTM over the richer feature set plus sampled-token embeddings

This lets you quantify how much value comes from:

- distribution shape alone
- the sampled token's own confidence
- segment structure
- token identity

## Evaluation Suggestions

At minimum, report:

- AUROC
- AUPRC
- Brier score
- expected calibration error

Use held-out rows from the same dataset version. If you want stronger generalization tests, also try subject-held-out validation splits.

## Practical Advice

- Ignore string columns during training. They are for debugging and analysis.
- Keep `manifest.json` with every experiment so you know which prompt/config/model produced the data.
- Do not mix multiple prompt variants into one training set unless that is the experiment.
