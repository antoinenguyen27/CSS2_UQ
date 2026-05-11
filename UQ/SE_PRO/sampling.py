from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from UQ.SE_PRO.config import (
    ANSWER_LETTERS,
    DEFAULT_N_SAMPLES,
    HF_DATA_FILE,
    HF_DATASET,
    MAX_CHOICES,
    MAX_NUM_BATCHED_TOKENS,
    MAX_RETRIES_PER_INVALID_SAMPLE,
    MIN_CHOICES,
    MIN_VALID_SAMPLES,
    answer_letters_for_choices,
    build_messages,
    count_prompt_tokens,
    parse_answer,
    render_prompt,
)


@dataclass(frozen=True)
class PreparedQuestion:
    example_id: str
    subject: str
    question: str
    choices: list[str]
    answer_letters: tuple[str, ...]
    gold_answer: str
    is_correct: int
    source_split: str | None
    sampled_token_logprobs: Any
    prompt_text: str
    prompt_tokens: int


@dataclass(frozen=True)
class SampleRequest:
    question_index: int
    sample_index: int
    attempt: int
    prompt_text: str
    prompt_tokens: int
    seed: int


def load_mmlu_pro_trace_dataframe(
    hf_dataset: str = HF_DATASET,
    data_file: str = HF_DATA_FILE,
) -> pd.DataFrame:
    dataset = load_dataset(hf_dataset, data_files=data_file)
    return dataset["train"].to_pandas()


def validate_row(row: dict[str, Any]) -> tuple[bool, str]:
    if not row.get("parse_success", False):
        return False, "parse_success=False"
    if row.get("top20_token_logprobs") is None or len(row["top20_token_logprobs"]) == 0:
        return False, "empty top20_token_logprobs"
    if row.get("is_correct") is None:
        return False, "missing is_correct"
    if not row.get("example_id"):
        return False, "missing example_id"
    if not row.get("subject"):
        return False, "missing subject"
    if not row.get("question"):
        return False, "missing question"

    choices = row.get("choices")
    if choices is None or isinstance(choices, (str, bytes)) or not hasattr(choices, "__len__"):
        return False, "invalid choices"
    if not MIN_CHOICES <= len(choices) <= MAX_CHOICES:
        return False, "invalid choices"
    if any(choice is None for choice in choices):
        return False, "invalid choices"

    if row.get("gold_answer") not in answer_letters_for_choices(choices):
        return False, "invalid gold_answer"
    return True, ""


def filter_valid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    keep_indices: list[int] = []
    dropped: dict[str, int] = {}
    for idx, row in enumerate(df.to_dict(orient="records")):
        is_valid, reason = validate_row(row)
        if is_valid:
            keep_indices.append(idx)
            continue
        dropped[reason] = dropped.get(reason, 0) + 1
    filtered_df = df.iloc[keep_indices].reset_index(drop=True)
    return filtered_df, dropped


def build_eval_partitions(df: pd.DataFrame, eval_mode: str, seed: int) -> dict[str, pd.DataFrame]:
    if eval_mode == "full":
        return {"full": df.reset_index(drop=True)}
    if eval_mode != "split":
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df["is_correct"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["is_correct"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def select_eval_partition(
    partitions: dict[str, pd.DataFrame],
    eval_mode: str,
    num_eval_rows: int,
) -> tuple[str, dict[str, pd.DataFrame]]:
    if num_eval_rows < 0:
        raise ValueError(f"num_eval_rows must be non-negative, got {num_eval_rows}")

    if eval_mode == "full":
        primary_split = "full"
    elif eval_mode == "split":
        primary_split = "test"
    else:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    if primary_split not in partitions:
        available = ", ".join(sorted(partitions))
        raise ValueError(f"Missing expected eval partition {primary_split!r}; available partitions: {available}")

    selected_df = partitions[primary_split].reset_index(drop=True)
    if num_eval_rows > 0:
        if num_eval_rows > len(selected_df):
            raise ValueError(
                f"Requested num_eval_rows={num_eval_rows}, but only {len(selected_df)} rows "
                f"are available in the {primary_split!r} eval partition."
            )
        selected_df = selected_df.head(num_eval_rows).reset_index(drop=True)

    return primary_split, {primary_split: selected_df}


def prepare_questions(df: pd.DataFrame, tokenizer: Any) -> list[PreparedQuestion]:
    prepared: list[PreparedQuestion] = []
    for row in df.to_dict(orient="records"):
        choices = list(row["choices"])
        example = {
            "subject": row["subject"],
            "question": row["question"],
            "choices": choices,
        }
        messages = build_messages(example)
        prompt_text = render_prompt(tokenizer, messages)
        prompt_tokens = count_prompt_tokens(tokenizer, messages)
        prepared.append(
            PreparedQuestion(
                example_id=row["example_id"],
                subject=row["subject"],
                question=row["question"],
                choices=choices,
                answer_letters=answer_letters_for_choices(choices),
                gold_answer=row["gold_answer"],
                is_correct=int(row["is_correct"]),
                source_split=row.get("split"),
                sampled_token_logprobs=row.get("sampled_token_logprobs"),
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
            )
        )
    return prepared


def sample_seed(master_seed: int, question_index: int, sample_index: int, attempt: int) -> int:
    return int(master_seed + question_index * 10_000 + sample_index * 100 + attempt)


def batch_sample_requests(
    requests: list[SampleRequest],
    max_num_seqs: int,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
    max_output_tokens: int = 2048,
) -> list[list[SampleRequest]]:
    batches: list[list[SampleRequest]] = []
    current: list[SampleRequest] = []
    current_budget = 0

    for request in requests:
        request_budget = request.prompt_tokens + max_output_tokens
        if request_budget > max_num_batched_tokens:
            if current:
                batches.append(current)
                current = []
                current_budget = 0
            batches.append([request])
            continue

        would_exceed_count = len(current) >= max_num_seqs
        would_exceed_budget = current_budget + request_budget > max_num_batched_tokens
        if current and (would_exceed_count or would_exceed_budget):
            batches.append(current)
            current = []
            current_budget = 0

        current.append(request)
        current_budget += request_budget

    if current:
        batches.append(current)
    return batches


def sample_questions(
    *,
    llm: Any,
    questions: list[PreparedQuestion],
    n_samples: int = DEFAULT_N_SAMPLES,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    max_num_seqs: int,
    max_retries: int = MAX_RETRIES_PER_INVALID_SAMPLE,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    slot_outputs: dict[tuple[int, int], str | None] = {}
    slot_answers: dict[tuple[int, int], str | None] = {}
    pending_slots = {
        (question_index, sample_index)
        for question_index in range(len(questions))
        for sample_index in range(n_samples)
    }

    for attempt in range(max_retries + 1):
        if not pending_slots:
            break

        requests = [
            SampleRequest(
                question_index=question_index,
                sample_index=sample_index,
                attempt=attempt,
                prompt_text=questions[question_index].prompt_text,
                prompt_tokens=questions[question_index].prompt_tokens,
                seed=sample_seed(seed, question_index, sample_index, attempt),
            )
            for question_index, sample_index in sorted(pending_slots)
        ]
        batches = batch_sample_requests(
            requests,
            max_num_seqs=max_num_seqs,
            max_output_tokens=max_tokens,
        )

        resolved_slots: set[tuple[int, int]] = set()
        for batch in batches:
            prompts = [request.prompt_text for request in batch]
            sampling_params = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                    seed=request.seed,
                    n=1,
                )
                for request in batch
            ]
            outputs = llm.generate(prompts, sampling_params)

            for request, output in zip(batch, outputs, strict=True):
                generated_text = output.outputs[0].text if output.outputs else ""
                parsed = parse_answer(generated_text)
                if not parsed.parse_success:
                    continue
                if parsed.predicted_answer not in questions[request.question_index].answer_letters:
                    continue
                key = (request.question_index, request.sample_index)
                slot_outputs[key] = generated_text
                slot_answers[key] = parsed.predicted_answer
                resolved_slots.add(key)

        pending_slots -= resolved_slots

    question_results: list[dict[str, Any]] = []
    dropped = 0
    for question_index, question in enumerate(questions):
        valid_answers = [
            slot_answers[(question_index, sample_index)]
            for sample_index in range(n_samples)
            if slot_answers.get((question_index, sample_index)) is not None
        ]
        valid_answers = [answer for answer in valid_answers if answer is not None]
        missing_sample_count = n_samples - len(valid_answers)
        drop_reason = "" if len(valid_answers) >= MIN_VALID_SAMPLES else "fewer_than_min_valid_samples"
        if drop_reason:
            dropped += 1

        if (question_index + 1) % 100 == 0 or question_index + 1 == len(questions):
            print(
                json.dumps(
                    {
                        "event": "se_pro_sampling_progress",
                        "processed_questions": question_index + 1,
                        "total_questions": len(questions),
                        "dropped_questions": dropped,
                    },
                    sort_keys=True,
                )
            )

        question_results.append(
            {
                "example_id": question.example_id,
                "subject": question.subject,
                "source_split": question.source_split,
                "gold_answer": question.gold_answer,
                "is_correct": question.is_correct,
                "choices": question.choices,
                "valid_answers": valid_answers,
                "missing_sample_count": missing_sample_count,
                "drop_reason": drop_reason,
                "sampled_token_logprobs": question.sampled_token_logprobs,
            }
        )

    return question_results
