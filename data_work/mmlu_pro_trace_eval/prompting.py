from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    ANSWER_BLOCK_PATTERN,
    ANSWER_PATTERN,
    SYSTEM_PROMPT,
    THINKING_PATTERN,
    USER_PROMPT_TEMPLATE,
)


@dataclass(frozen=True)
class ParsedAnswer:
    predicted_answer: str | None
    parse_success: bool
    parse_error: str
    thinking_text: str
    answer_text: str


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    choices = example["choices"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                subject=example["subject"],
                question=example["question"],
                choice_a=choices[0],
                choice_b=choices[1],
                choice_c=choices[2],
                choice_d=choices[3],
                choice_e=choices[4],
                choice_f=choices[5],
                choice_g=choices[6],
                choice_h=choices[7],
                choice_i=choices[8],
                choice_j=choices[9],
            ),
        },
    ]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def count_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    return len(tokenized)


def parse_answer(completion_text: str) -> ParsedAnswer:
    answer_matches = list(ANSWER_PATTERN.finditer(completion_text))
    thinking_match = THINKING_PATTERN.search(completion_text)
    answer_block_match = ANSWER_BLOCK_PATTERN.search(completion_text)

    thinking_text = thinking_match.group(1).strip() if thinking_match else ""
    answer_text = answer_block_match.group(1).strip() if answer_block_match else ""

    if not answer_matches:
        return ParsedAnswer(
            predicted_answer=None,
            parse_success=False,
            parse_error="missing_answer_tag",
            thinking_text=thinking_text,
            answer_text=answer_text,
        )

    predicted_answer = answer_matches[0].group(1).upper()
    parse_error = ""
    if len(answer_matches) > 1:
        parse_error = "multiple_answer_tags"

    return ParsedAnswer(
        predicted_answer=predicted_answer,
        parse_success=True,
        parse_error=parse_error,
        thinking_text=thinking_text,
        answer_text=answer_text,
    )
