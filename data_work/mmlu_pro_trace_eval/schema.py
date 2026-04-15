"""
Override of mmlu_trace_eval.schema.build_example_record using the pro parse_answer (A-J).
All other schema functions are re-exported from mmlu_trace_eval.schema unchanged.
"""
from __future__ import annotations

from typing import Any

from mmlu_trace_eval.schema import (
    build_token_step_records,
    compute_token_surfaces,
    find_segment_bounds,
    normalize_topk,
    pa_table_from_records,
    segment_token_surfaces,
    validate_example_record,
)
from mmlu_trace_eval.config import SEGMENT_NAME_TO_ID

from .prompting import parse_answer


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
    parsed = parse_answer(completion_text)  # uses A-J pattern from pro config
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
        "thinking_token_count": sum(1 for v in segment_ids if v == SEGMENT_NAME_TO_ID["thinking"]),
        "answer_token_count": sum(1 for v in segment_ids if v == SEGMENT_NAME_TO_ID["answer"]),
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
