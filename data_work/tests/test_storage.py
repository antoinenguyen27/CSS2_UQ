from pathlib import Path

import pyarrow.parquet as pq

from mmlu_trace_eval.storage import (
    MetricsAccumulator,
    build_manifest,
    compact_shards,
    iter_example_records_from_shards,
    list_completed_example_ids,
    write_shard,
)


def _example_record(example_id: str, subject: str, correct: bool) -> dict:
    return {
        "run_id": "run-1",
        "example_id": example_id,
        "split": "test",
        "subject": subject,
        "question_idx": 0,
        "question": "q",
        "choices": ["a", "b", "c", "d"],
        "gold_answer": "A",
        "predicted_answer": "A" if correct else "B",
        "parse_success": True,
        "parse_error": "",
        "is_correct": correct,
        "model_name": "m",
        "model_revision": "mr",
        "tokenizer_revision": "tr",
        "dataset_name": "d",
        "dataset_revision": "dr",
        "prompt_variant": "reasoning_v1",
        "prompt_text": "prompt",
        "decode_config_json": "{}",
        "completion_text": "<thinking>x</thinking><answer>A</answer>",
        "thinking_text": "x",
        "answer_text": "A",
        "trajectory_length": 1,
        "thinking_token_count": 1,
        "answer_token_count": 0,
        "thinking_start_idx": 0,
        "thinking_end_idx": 1,
        "answer_start_idx": -1,
        "answer_end_idx": -1,
        "sampled_token_ids": [1],
        "sampled_token_texts": ["tok"],
        "sampled_token_logprobs": [-0.1],
        "sampled_token_ranks": [1],
        "segment_ids": [0],
        "segment_names": ["thinking"],
        "cumulative_logprobs": [-0.1],
        "top20_token_ids": [[1] + [-1] * 19],
        "top20_token_texts": [["tok"] + [""] * 19],
        "top20_token_logprobs": [[-0.1] + [float("-inf")] * 19],
    }


def _token_rows(example_id: str) -> list[dict]:
    return [
        {
            "run_id": "run-1",
            "example_id": example_id,
            "step_idx": 0,
            "subject": "math",
            "gold_answer": "A",
            "predicted_answer": "A",
            "is_correct": True,
            "segment_id": 0,
            "segment_name": "thinking",
            "sampled_token_id": 1,
            "sampled_token_text": "tok",
            "sampled_token_logprob": -0.1,
            "sampled_token_rank": 1,
            "cumulative_logprob": -0.1,
            "top20_token_ids": [1] + [-1] * 19,
            "top20_token_texts": ["tok"] + [""] * 19,
            "top20_token_logprobs": [-0.1] + [float("-inf")] * 19,
        }
    ]


def test_write_shard_and_resume_metadata(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    example_records = [_example_record("e1", "math", True)]
    token_rows = _token_rows("e1")
    metadata = write_shard(run_dir, 0, example_records, token_rows)
    assert metadata["example_count"] == 1
    assert list_completed_example_ids(run_dir) == {"e1"}


def test_compact_shards(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    write_shard(run_dir, 0, [_example_record("e1", "math", True)], _token_rows("e1"))
    write_shard(run_dir, 1, [_example_record("e2", "history", False)], _token_rows("e2"))
    records = iter_example_records_from_shards(run_dir)
    assert {record["example_id"] for record in records} == {"e1", "e2"}
    output = compact_shards(run_dir, "examples", "examples.parquet")
    table = pq.read_table(output)
    assert table.num_rows == 2


def test_metrics_accumulator():
    metrics = MetricsAccumulator()
    metrics.update(_example_record("e1", "math", True))
    metrics.update(_example_record("e2", "math", False))
    payload = metrics.to_dict()
    assert payload["correct"] == 1
    assert payload["total"] == 2
    assert payload["per_subject_accuracy"]["math"]["accuracy"] == 0.5


def test_build_manifest():
    manifest = build_manifest(
        run_id="run-1",
        split="test",
        subject=None,
        requested_limit=None,
        total_examples=2,
        asset_metadata={"model_revision": "abc"},
        prompt_variant="reasoning_v1",
        prompt_hash="hash",
        decode_config_json="{}",
        package_versions={"modal": "1.3.5"},
        shard_inventory=[],
        started_at_utc="2026-03-25T00:00:00+00:00",
        finished_at_utc=None,
    )
    assert manifest["run_id"] == "run-1"
    assert manifest["prompt_hash"] == "hash"
