from types import SimpleNamespace

import pandas as pd

from UQ.SE_PRO.sampling import (
    filter_valid_rows,
    prepare_questions,
    sample_questions,
)


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors=None):
        rendered = "\n".join(message["content"] for message in messages)
        if tokenize:
            return rendered.split()
        return rendered


class OutOfRangeThenValidLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompts, sampling_params):
        self.calls += 1
        if self.calls == 1:
            text = "<thinking>bad</thinking><answer>J</answer>"
        else:
            text = "<thinking>ok</thinking><answer>B</answer>"
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for _ in prompts]


def _valid_row(example_id: str, choices: list[str], gold_answer: str = "A") -> dict:
    return {
        "example_id": example_id,
        "split": "test",
        "subject": "business",
        "question": "Pick one.",
        "choices": choices,
        "gold_answer": gold_answer,
        "parse_success": True,
        "top20_token_logprobs": [[-1.0] * 20],
        "sampled_token_logprobs": [-1.0, -2.0],
        "is_correct": 1,
    }


def test_filter_valid_rows_accepts_three_to_ten_choices_and_rejects_outside_range():
    df = pd.DataFrame(
        [
            _valid_row("three", ["a", "b", "c"], "C"),
            _valid_row("ten", list("abcdefghij"), "J"),
            _valid_row("two", ["a", "b"], "B"),
            _valid_row("eleven", list("abcdefghijk"), "K"),
            _valid_row("bad_gold", ["a", "b", "c"], "D"),
        ]
    )

    filtered_df, drops = filter_valid_rows(df)

    assert list(filtered_df["example_id"]) == ["three", "ten"]
    assert drops == {"invalid choices": 2, "invalid gold_answer": 1}


def test_sample_questions_retries_answers_outside_question_choice_letters():
    tokenizer = FakeTokenizer()
    df = pd.DataFrame([_valid_row("three", ["a", "b", "c"], "B")])
    questions = prepare_questions(df, tokenizer)

    results = sample_questions(
        llm=OutOfRangeThenValidLLM(),
        questions=questions,
        n_samples=1,
        temperature=0.7,
        top_p=0.95,
        max_tokens=32,
        seed=7,
        max_num_seqs=8,
        max_retries=1,
    )

    assert results[0]["valid_answers"] == ["B"]
    assert results[0]["choices"] == ["a", "b", "c"]
    assert results[0]["drop_reason"] == "fewer_than_min_valid_samples"
