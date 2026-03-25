from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import SHARD_SIZE
from .schema import pa_table_from_records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def shard_path(run_dir: Path, prefix: str, shard_index: int) -> Path:
    return run_dir / "shards" / f"{prefix}-{shard_index:05d}.parquet"


def shard_metadata_path(run_dir: Path, shard_index: int) -> Path:
    return run_dir / "shards" / f"meta-{shard_index:05d}.json"


def list_completed_example_ids(run_dir: Path) -> set[str]:
    completed: set[str] = set()
    shards_dir = run_dir / "shards"
    if not shards_dir.exists():
        return completed
    for metadata_path in sorted(shards_dir.glob("meta-*.json")):
        payload = read_json(metadata_path)
        completed.update(payload.get("example_ids", []))
    return completed


def write_shard(
    run_dir: Path,
    shard_index: int,
    example_records: list[dict[str, Any]],
    token_step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ensure_dir(run_dir / "shards")
    example_path = shard_path(run_dir, "examples", shard_index)
    token_path = shard_path(run_dir, "token_steps", shard_index)
    metadata_path = shard_metadata_path(run_dir, shard_index)

    pq.write_table(pa_table_from_records(example_records), example_path)
    pq.write_table(pa_table_from_records(token_step_records), token_path)

    metadata = {
        "shard_index": shard_index,
        "example_count": len(example_records),
        "token_step_count": len(token_step_records),
        "example_ids": [record["example_id"] for record in example_records],
        "examples_path": example_path.name,
        "token_steps_path": token_path.name,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metadata_path, metadata)
    return metadata


def compact_shards(run_dir: Path, prefix: str, output_name: str) -> Path:
    shard_paths = sorted((run_dir / "shards").glob(f"{prefix}-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for prefix {prefix}")
    dataset = ds.dataset([str(path) for path in shard_paths], format="parquet")
    table = dataset.to_table()
    output_path = run_dir / output_name
    pq.write_table(table, output_path)
    return output_path


def iter_example_records_from_shards(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard_path in sorted((run_dir / "shards").glob("examples-*.parquet")):
        records.extend(pq.read_table(shard_path).to_pylist())
    return records


@dataclass
class MetricsAccumulator:
    correct: int = 0
    total: int = 0
    parse_failures: int = 0
    trajectory_lengths: list[int] = field(default_factory=list)
    answer_lengths: list[int] = field(default_factory=list)
    subject_correct: dict[str, int] = field(default_factory=dict)
    subject_total: dict[str, int] = field(default_factory=dict)

    def update(self, example_record: dict[str, Any]) -> None:
        subject = example_record["subject"]
        self.total += 1
        self.correct += int(example_record["is_correct"])
        self.parse_failures += int(not example_record["parse_success"])
        self.trajectory_lengths.append(int(example_record["trajectory_length"]))
        self.answer_lengths.append(int(example_record["answer_token_count"]))
        self.subject_total[subject] = self.subject_total.get(subject, 0) + 1
        self.subject_correct[subject] = self.subject_correct.get(subject, 0) + int(example_record["is_correct"])

    def to_dict(self) -> dict[str, Any]:
        per_subject = {
            subject: {
                "correct": self.subject_correct.get(subject, 0),
                "total": total,
                "accuracy": (self.subject_correct.get(subject, 0) / total) if total else 0.0,
            }
            for subject, total in sorted(self.subject_total.items())
        }
        return {
            "overall_accuracy": (self.correct / self.total) if self.total else 0.0,
            "correct": self.correct,
            "total": self.total,
            "parse_failure_count": self.parse_failures,
            "parse_failure_rate": (self.parse_failures / self.total) if self.total else 0.0,
            "trajectory_length_mean": mean(self.trajectory_lengths) if self.trajectory_lengths else 0.0,
            "trajectory_length_median": median(self.trajectory_lengths) if self.trajectory_lengths else 0.0,
            "answer_length_mean": mean(self.answer_lengths) if self.answer_lengths else 0.0,
            "answer_length_median": median(self.answer_lengths) if self.answer_lengths else 0.0,
            "per_subject_accuracy": per_subject,
        }


def build_manifest(
    *,
    run_id: str,
    split: str,
    subject: str | None,
    requested_limit: int | None,
    total_examples: int,
    asset_metadata: dict[str, Any],
    prompt_variant: str,
    prompt_hash: str,
    decode_config_json: str,
    package_versions: dict[str, str],
    shard_inventory: list[dict[str, Any]],
    started_at_utc: str,
    finished_at_utc: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "split": split,
        "subject": subject,
        "requested_limit": requested_limit,
        "total_examples": total_examples,
        "asset_metadata": asset_metadata,
        "prompt_variant": prompt_variant,
        "prompt_hash": prompt_hash,
        "decode_config_json": decode_config_json,
        "package_versions": package_versions,
        "shard_inventory": shard_inventory,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }
