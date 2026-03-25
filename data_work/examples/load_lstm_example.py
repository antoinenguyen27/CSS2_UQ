from __future__ import annotations

import sys

import numpy as np
import pyarrow.parquet as pq


def main(path: str) -> None:
    table = pq.read_table(path)
    row = table.slice(0, 1).to_pylist()[0]

    top20 = np.asarray(row["top20_token_logprobs"], dtype=np.float32)
    sampled = np.asarray(row["sampled_token_logprobs"], dtype=np.float32)[:, None]
    segments = np.asarray(row["segment_ids"], dtype=np.float32)[:, None]
    positions = np.arange(top20.shape[0], dtype=np.float32)[:, None]
    positions /= max(top20.shape[0], 1)

    features = np.concatenate([top20, sampled, segments, positions], axis=1)
    label = np.asarray(row["is_correct"], dtype=np.float32)

    print(f"example_id={row['example_id']}")
    print(f"trajectory_length={row['trajectory_length']}")
    print(f"features_shape={features.shape}")
    print(f"label={label.item()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/load_lstm_example.py <examples.parquet>")
    main(sys.argv[1])
