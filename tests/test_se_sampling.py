from types import SimpleNamespace

import numpy as np
import pandas as pd

from UQ.SE.sampling import (
    build_eval_partitions,
    filter_valid_rows,
    sample_questions,
    sample_seed,
    select_eval_partition,
)


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors=None):
        rendered = "\n".join(message["content"] for message in messages)
        if tokenize:
            return rendered.split()
        return rendered


class FakeLLM:
    def generate(self, prompts, sampling_params):
        outputs = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            if params.seed % 2 == 0:
                text = "<thinking>ok</thinking><answer>A</answer>"
            else:
                text = "<thinking>ok</thinking><answer>B</answer>"
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
        return outputs


def _valid_row(example_id: str, is_correct: int) -> dict:
    return {
        "example_id": example_id,
        "split": "test",
        "subject": "math",
        "question": "2 + 2 = ?",
        "choices": ["1", "2", "3", "4"],
        "gold_answer": "D",
        "parse_success": True,
        "top20_token_logprobs": [[-1.0] * 20],
        "sampled_token_logprobs": [-1.0, -2.0],
        "is_correct": is_correct,
    }


def test_filter_valid_rows_matches_halt_style_checks_and_extra_shape_checks():
    df = pd.DataFrame(
        [
            {**_valid_row("e1", 1), "choices": np.array(["1", "2", "3", "4"], dtype=object)},
            {**_valid_row("e2", 0), "parse_success": False},
            {**_valid_row("e3", 1), "choices": ["1", "2"]},
        ]
    )
    filtered_df, drops = filter_valid_rows(df)
    assert list(filtered_df["example_id"]) == ["e1"]
    assert drops == {"parse_success=False": 1, "invalid choices": 1}


def test_build_eval_partitions_preserves_total_count():
    rows = [_valid_row(f"e{i}", i % 2) for i in range(20)]
    df = pd.DataFrame(rows)
    partitions = build_eval_partitions(df, eval_mode="split", seed=42)
    assert sum(len(partition) for partition in partitions.values()) == len(df)
    assert set(partitions) == {"train", "val", "test"}
    assert len(partitions["test"]) == 3


def test_select_eval_partition_caps_after_split():
    rows = [_valid_row(f"e{i}", i % 2) for i in range(40)]
    df = pd.DataFrame(rows)
    partitions = build_eval_partitions(df, eval_mode="split", seed=42)

    split_name, selected = select_eval_partition(
        partitions,
        eval_mode="split",
        num_eval_rows=4,
    )

    assert split_name == "test"
    assert set(selected) == {"test"}
    assert len(selected["test"]) == 4
    assert len(partitions["test"]) == 6


def test_select_eval_partition_raises_when_request_exceeds_partition():
    rows = [_valid_row(f"e{i}", i % 2) for i in range(20)]
    df = pd.DataFrame(rows)
    partitions = build_eval_partitions(df, eval_mode="split", seed=42)

    import pytest

    with pytest.raises(ValueError, match="num_eval_rows=4"):
        select_eval_partition(partitions, eval_mode="split", num_eval_rows=4)


def test_select_eval_partition_zero_keeps_entire_selected_partition():
    rows = [_valid_row(f"e{i}", i % 2) for i in range(20)]
    df = pd.DataFrame(rows)
    partitions = build_eval_partitions(df, eval_mode="split", seed=42)

    split_name, selected = select_eval_partition(
        partitions,
        eval_mode="split",
        num_eval_rows=0,
    )

    assert split_name == "test"
    assert len(selected["test"]) == len(partitions["test"])


def test_sample_questions_runs_end_to_end_with_fake_llm():
    tokenizer = FakeTokenizer()
    df = pd.DataFrame([_valid_row("e1", 1)])
    from UQ.SE.sampling import prepare_questions

    prepared = prepare_questions(df, tokenizer)
    results = sample_questions(
        llm=FakeLLM(),
        questions=prepared,
        n_samples=6,
        temperature=0.7,
        top_p=0.95,
        max_tokens=32,
        seed=7,
        max_num_seqs=8,
    )
    assert len(results) == 1
    assert len(results[0]["valid_answers"]) == 6
    assert results[0]["drop_reason"] == ""


def test_sample_seed_is_deterministic_and_distinct_per_slot():
    assert sample_seed(42, 0, 0, 0) == 42
    assert sample_seed(42, 1, 0, 0) != sample_seed(42, 0, 1, 0)
