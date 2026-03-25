from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pyarrow as pa

from .config import (
    MAX_TOP_LOGPROBS,
    SEGMENT_ID_TO_NAME,
    SEGMENT_NAME_TO_ID,
)
from .prompting import parse_answer


@dataclass(frozen=True)
class NormalizedTopK:
    token_ids: list[int]
    token_texts: list[str]
    logprobs: list[float]
    ranks: list[int]
    sampled_rank: int
    sampled_logprob: float


def _get_attr_or_item(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _normalize_logprob_entries(step_logprobs: Any) -> list[tuple[int, float, int | None]]:
    items: list[tuple[int, float, int | None]] = []

    if step_logprobs is None:
        return items

    if isinstance(step_logprobs, dict):
        iterable = step_logprobs.items()
    else:
        iterable = step_logprobs

    for entry in iterable:
        if isinstance(step_logprobs, dict):
            maybe_key, maybe_value = entry
            token_id = int(maybe_key)
            value = maybe_value
        elif isinstance(entry, tuple) and len(entry) == 2:
            maybe_key, maybe_value = entry
            token_id = _get_attr_or_item(maybe_key, "token_id")
            value = maybe_value
            if token_id is None and isinstance(maybe_key, int):
                token_id = maybe_key
            if token_id is None:
                token_id = _get_attr_or_item(maybe_value, "token_id")
        else:
            token_id = _get_attr_or_item(entry, "token_id")
            value = entry
        logprob = _get_attr_or_item(value, "logprob", value)
        rank = _get_attr_or_item(value, "rank")
        if token_id is None or logprob is None:
            continue
        items.append((int(token_id), float(logprob), int(rank) if rank is not None else None))
    return items


def normalize_topk(
    tokenizer: Any,
    step_logprobs: Any,
    sampled_token_id: int,
    top_k: int = MAX_TOP_LOGPROBS,
) -> NormalizedTopK:
    entries = _normalize_logprob_entries(step_logprobs)
    by_token_id = {token_id: (logprob, rank) for token_id, logprob, rank in entries}

    if sampled_token_id not in by_token_id:
        raise ValueError(f"Sampled token {sampled_token_id} missing from logprob candidates")

    entries_sorted = sorted(
        ((token_id, logprob, rank) for token_id, (logprob, rank) in by_token_id.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    token_ids = [token_id for token_id, _, _ in entries_sorted[:top_k]]
    logprobs = [float(logprob) for _, logprob, _ in entries_sorted[:top_k]]
    ranks: list[int] = []

    sampled_rank = -1
    sampled_logprob = by_token_id[sampled_token_id][0]
    for index, (token_id, _, declared_rank) in enumerate(entries_sorted[:top_k]):
        rank = declared_rank if declared_rank is not None else index + 1
        ranks.append(int(rank))
        if token_id == sampled_token_id:
            sampled_rank = int(rank)

    while len(token_ids) < top_k:
        token_ids.append(-1)
        logprobs.append(float("-inf"))
        ranks.append(-1)

    token_texts = [
        "" if token_id == -1 else tokenizer.convert_ids_to_tokens(int(token_id))
        for token_id in token_ids
    ]

    if sampled_rank == -1:
        sampled_rank = next(
            (index + 1 for index, token_id in enumerate(token_ids) if token_id == sampled_token_id),
            -1,
        )

    return NormalizedTopK(
        token_ids=[int(token_id) for token_id in token_ids],
        token_texts=token_texts,
        logprobs=[float(value) for value in logprobs],
        ranks=[int(rank) for rank in ranks],
        sampled_rank=int(sampled_rank),
        sampled_logprob=float(sampled_logprob),
    )


def compute_token_surfaces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    surfaces: list[str] = []
    previous = ""
    for end in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(
            token_ids[:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        surfaces.append(decoded[len(previous) :])
        previous = decoded
    return surfaces


def compute_char_spans(token_surfaces: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for surface in token_surfaces:
        start = cursor
        cursor += len(surface)
        spans.append((start, cursor))
    return spans


def segment_token_surfaces(completion_text: str, token_surfaces: list[str]) -> tuple[list[int], list[str]]:
    spans = compute_char_spans(token_surfaces)
    segment_ids = [SEGMENT_NAME_TO_ID["structure"]] * len(token_surfaces)

    regions: list[tuple[int, int, str]] = []
    for match in re.finditer(r"<thinking>(.*?)</thinking>", completion_text, re.DOTALL):
        regions.append((match.start(1), match.end(1), "thinking"))
    for match in re.finditer(r"<answer>(.*?)</answer>", completion_text, re.DOTALL):
        regions.append((match.start(1), match.end(1), "answer"))

    for index, (start, end) in enumerate(spans):
        if start == end:
            continue
        best = "structure"
        for region_start, region_end, region_name in regions:
            overlap = min(end, region_end) - max(start, region_start)
            if overlap > 0:
                best = region_name
                break
        segment_ids[index] = SEGMENT_NAME_TO_ID[best]

    return segment_ids, [SEGMENT_ID_TO_NAME[segment_id] for segment_id in segment_ids]


def find_segment_bounds(segment_ids: list[int], target: int) -> tuple[int, int]:
    indices = [index for index, segment_id in enumerate(segment_ids) if segment_id == target]
    if not indices:
        return -1, -1
    return indices[0], indices[-1] + 1


def build_example_record(
    *,
    run_id: str,
    split: str,
    prompt_variant: str,
    decode_config_json: str,
    model_name: str,
    model_revision: str,
    tokenizer_revision: str,
    dataset_name: str,
    dataset_revision: str,
    prompt_text: str,
    example: dict[str, Any],
    completion_text: str,
    sampled_token_ids: list[int],
    sampled_token_texts: list[str],
    sampled_token_logprobs: list[float],
    sampled_token_ranks: list[int],
    segment_ids: list[int],
    segment_names: list[str],
    cumulative_logprobs: list[float],
    top20_token_ids: list[list[int]],
    top20_token_texts: list[list[str]],
    top20_token_logprobs: list[list[float]],
) -> dict[str, Any]:
    parsed = parse_answer(completion_text)
    answer_start_idx, answer_end_idx = find_segment_bounds(segment_ids, SEGMENT_NAME_TO_ID["answer"])
    thinking_start_idx, thinking_end_idx = find_segment_bounds(segment_ids, SEGMENT_NAME_TO_ID["thinking"])
    is_correct = parsed.predicted_answer == example["gold_answer"] if parsed.parse_success else False

    return {
        "run_id": run_id,
        "example_id": example["example_id"],
        "split": split,
        "subject": example["subject"],
        "question_idx": int(example["question_idx"]),
        "question": example["question"],
        "choices": list(example["choices"]),
        "gold_answer": example["gold_answer"],
        "predicted_answer": parsed.predicted_answer,
        "parse_success": bool(parsed.parse_success),
        "parse_error": parsed.parse_error,
        "is_correct": bool(is_correct),
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "prompt_variant": prompt_variant,
        "prompt_text": prompt_text,
        "decode_config_json": decode_config_json,
        "completion_text": completion_text,
        "thinking_text": parsed.thinking_text,
        "answer_text": parsed.answer_text,
        "trajectory_length": len(sampled_token_ids),
        "thinking_token_count": sum(1 for value in segment_ids if value == SEGMENT_NAME_TO_ID["thinking"]),
        "answer_token_count": sum(1 for value in segment_ids if value == SEGMENT_NAME_TO_ID["answer"]),
        "thinking_start_idx": thinking_start_idx,
        "thinking_end_idx": thinking_end_idx,
        "answer_start_idx": answer_start_idx,
        "answer_end_idx": answer_end_idx,
        "sampled_token_ids": sampled_token_ids,
        "sampled_token_texts": sampled_token_texts,
        "sampled_token_logprobs": sampled_token_logprobs,
        "sampled_token_ranks": sampled_token_ranks,
        "segment_ids": segment_ids,
        "segment_names": segment_names,
        "cumulative_logprobs": cumulative_logprobs,
        "top20_token_ids": top20_token_ids,
        "top20_token_texts": top20_token_texts,
        "top20_token_logprobs": top20_token_logprobs,
    }


def build_token_step_records(example_record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = example_record["trajectory_length"]
    for step_idx in range(length):
        rows.append(
            {
                "run_id": example_record["run_id"],
                "example_id": example_record["example_id"],
                "step_idx": step_idx,
                "subject": example_record["subject"],
                "gold_answer": example_record["gold_answer"],
                "predicted_answer": example_record["predicted_answer"],
                "is_correct": example_record["is_correct"],
                "segment_id": example_record["segment_ids"][step_idx],
                "segment_name": example_record["segment_names"][step_idx],
                "sampled_token_id": example_record["sampled_token_ids"][step_idx],
                "sampled_token_text": example_record["sampled_token_texts"][step_idx],
                "sampled_token_logprob": example_record["sampled_token_logprobs"][step_idx],
                "sampled_token_rank": example_record["sampled_token_ranks"][step_idx],
                "cumulative_logprob": example_record["cumulative_logprobs"][step_idx],
                "top20_token_ids": example_record["top20_token_ids"][step_idx],
                "top20_token_texts": example_record["top20_token_texts"][step_idx],
                "top20_token_logprobs": example_record["top20_token_logprobs"][step_idx],
            }
        )
    return rows


def validate_example_record(example_record: dict[str, Any]) -> None:
    length = example_record["trajectory_length"]
    vector_columns = [
        "sampled_token_ids",
        "sampled_token_texts",
        "sampled_token_logprobs",
        "sampled_token_ranks",
        "segment_ids",
        "segment_names",
        "cumulative_logprobs",
        "top20_token_ids",
        "top20_token_texts",
        "top20_token_logprobs",
    ]
    for column in vector_columns:
        if len(example_record[column]) != length:
            raise ValueError(f"Column {column} has length {len(example_record[column])}, expected {length}")

    for step_ids, step_texts, step_logprobs in zip(
        example_record["top20_token_ids"],
        example_record["top20_token_texts"],
        example_record["top20_token_logprobs"],
        strict=True,
    ):
        if len(step_ids) != MAX_TOP_LOGPROBS:
            raise ValueError("top20_token_ids width mismatch")
        if len(step_texts) != MAX_TOP_LOGPROBS:
            raise ValueError("top20_token_texts width mismatch")
        if len(step_logprobs) != MAX_TOP_LOGPROBS:
            raise ValueError("top20_token_logprobs width mismatch")

    for sampled, top1 in zip(
        example_record["sampled_token_ids"],
        (row[0] for row in example_record["top20_token_ids"]),
        strict=True,
    ):
        if sampled != top1:
            raise ValueError("Greedy decoded sampled token does not match top-1 logprob candidate")


def pa_table_from_records(records: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(records)
