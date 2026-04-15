from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import DATASET_CONFIG, DATASET_ID, SUPPORTED_SPLITS, slugify


def normalized_dataset_dir(dataset_root: str, dataset_revision: str) -> Path:
    return Path(dataset_root) / DATASET_ID.replace("/", "__") / DATASET_CONFIG / dataset_revision


def split_parquet_path(dataset_root: str, dataset_revision: str, split: str) -> Path:
    return normalized_dataset_dir(dataset_root, dataset_revision) / f"{split}.parquet"


def load_materialized_split(dataset_root: str, dataset_revision: str, split: str) -> list[dict[str, Any]]:
    path = split_parquet_path(dataset_root, dataset_revision, split)
    table = pq.read_table(path)
    return table.to_pylist()


def filter_examples(
    examples: list[dict[str, Any]],
    *,
    subject: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = examples
    if subject:
        filtered = [row for row in filtered if row["subject"] == subject]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def build_example_id(split: str, subject: str, question_idx: int) -> str:
    return f"mmlu_pro__{split}__{question_idx:05d}__{slugify(subject)}"
