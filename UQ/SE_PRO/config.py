from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Sequence


APP_NAME = "uq-semantic-entropy-mmlu-pro"
VOLUME_NAME = "mmlu-pro-trace-volume"
SECRET_NAME = "hf-auth"

MODEL_ID = "google/gemma-3-12b-it"
HF_DATASET = "auhsoJ69/mmlu-pro-traces"
HF_DATA_FILE = "examples.parquet"

VOLUME_ROOT = "/vol"
MODEL_ROOT = f"{VOLUME_ROOT}/models"
CACHE_ROOT = f"{VOLUME_ROOT}/cache"
RUNS_ROOT = f"{VOLUME_ROOT}/runs"

MODEL_CACHE_DIR = f"{CACHE_ROOT}/huggingface"
HF_HOME = MODEL_CACHE_DIR
TRANSFORMERS_CACHE = MODEL_CACHE_DIR
HF_DATASETS_CACHE = f"{CACHE_ROOT}/datasets"
VLLM_CACHE_ROOT = f"{CACHE_ROOT}/vllm"

MAX_NUM_SEQS = 224
SMOKE_MAX_NUM_SEQS = 64
SMOKE_LIMIT_THRESHOLD = 100
MAX_NUM_BATCHED_TOKENS = 98_304
MAX_MODEL_LEN = 3072

DEFAULT_N_SAMPLES = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SEED = 42
DEFAULT_EVAL_MODE = "full"
DEFAULT_OUTPUT_DIR = RUNS_ROOT

MIN_CHOICES = 3
MAX_CHOICES = 10
MIN_VALID_SAMPLES = 5
MAX_RETRIES_PER_INVALID_SAMPLE = 3
ANSWER_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

ANSWER_PATTERN = re.compile(r"<answer>\s*([A-Ja-j])\s*</answer>", re.DOTALL)
THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SYSTEM_PROMPT = """You are answering multiple-choice university exam questions for uncertainty-calibration data collection.

For each question:
1. Write your reasoning only inside <thinking>...</thinking>.
2. Inside <thinking>, use this exact order:
   - Core concept
   - Evaluate each provided option in order (Option A, Option B, ...)
   - Final decision
3. After the thinking section, output exactly one answer tag:
   <answer>X</answer>
   where X is one of the letters corresponding to the provided options.
4. Do not output anything after </answer>.
5. Even if uncertain, you must choose exactly one answer.
6. Take your time to carefully evaluate all provided options before committing to your final decision.
"""

USER_PROMPT_TEMPLATE = """Subject: {subject}

Question:
{question}

Choices:
{choices_block}

Respond exactly in this format:

<thinking>
Core concept: ...
{thinking_scaffold}
Final decision: ...
</thinking>
<answer>X</answer>
"""


@dataclass(frozen=True)
class ParsedAnswer:
    predicted_answer: str | None
    parse_success: bool
    parse_error: str
    thinking_text: str
    answer_text: str


def answer_letters_for_choices(choices: Sequence[Any]) -> tuple[str, ...]:
    return ANSWER_LETTERS[: len(choices)]


def build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    choices = list(example["choices"])
    letters = answer_letters_for_choices(choices)
    choices_block = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=True))
    thinking_scaffold = "\n".join(f"Option {letter}: ..." for letter in letters)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                subject=example["subject"],
                question=example["question"],
                choices_block=choices_block,
                thinking_scaffold=thinking_scaffold,
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


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "unknown"


def default_run_id(eval_mode: str) -> str:
    return f"gemma3-12b-it__semantic-entropy-mmlu-pro__{slugify(eval_mode)}__{utc_timestamp()}"


def runtime_max_num_seqs(requested_eval_rows: int | None) -> int:
    if requested_eval_rows is not None and 0 < requested_eval_rows <= SMOKE_LIMIT_THRESHOLD:
        return SMOKE_MAX_NUM_SEQS
    return MAX_NUM_SEQS
