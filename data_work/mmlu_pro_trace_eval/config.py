from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re


APP_NAME = "mmlu-pro-trace-eval"
VOLUME_NAME = "mmlu-pro-trace-volume"
SECRET_NAME = "hf-auth"

MODEL_ID = "google/gemma-3-12b-it"
DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_CONFIG = "default"
PROMPT_VARIANT = "reasoning_v1"

VOLUME_ROOT = "/vol"
MODEL_ROOT = f"{VOLUME_ROOT}/models"
DATASET_ROOT = f"{VOLUME_ROOT}/datasets"
CACHE_ROOT = f"{VOLUME_ROOT}/cache"
RUNS_ROOT = f"{VOLUME_ROOT}/runs"

MODEL_CACHE_DIR = f"{CACHE_ROOT}/huggingface"
HF_HOME = MODEL_CACHE_DIR
TRANSFORMERS_CACHE = MODEL_CACHE_DIR
HF_DATASETS_CACHE = f"{CACHE_ROOT}/datasets"
VLLM_CACHE_ROOT = f"{CACHE_ROOT}/vllm"

MAX_TOP_LOGPROBS = 20
MAX_OUTPUT_TOKENS = 512
MAX_NUM_SEQS = 224
SMOKE_MAX_NUM_SEQS = 64
MAX_NUM_BATCHED_TOKENS = 98_304
MAX_MODEL_LEN = 3072
SHARD_SIZE = 256

DEFAULT_EVALUATOR_CPU = 6
DEFAULT_PREPARE_CPU = 6
DEFAULT_EVALUATOR_MEMORY_MB = 65_536
DEFAULT_PREPARE_MEMORY_MB = 32_768

SUPPORTED_SPLITS = ("test", "validation")
SEGMENT_ID_TO_NAME = {
    0: "thinking",
    1: "answer",
    2: "structure",
}
SEGMENT_NAME_TO_ID = {value: key for key, value in SEGMENT_ID_TO_NAME.items()}

ANSWER_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

ANSWER_PATTERN = re.compile(r"<answer>\s*([A-Ja-j])\s*</answer>", re.DOTALL)
THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SYSTEM_PROMPT = """You are answering multiple-choice university exam questions for uncertainty-calibration data collection.

For each question:
1. Write your reasoning only inside <thinking>...</thinking>.
2. Inside <thinking>, use this exact order:
   - Core concept
   - Option A
   - Option B
   - Option C
   - Option D
   - Option E
   - Option F
   - Option G
   - Option H
   - Option I
   - Option J
   - Final decision
3. After the thinking section, output exactly one answer tag:
   <answer>X</answer>
   where X is A, B, C, D, E, F, G, H, I, or J.
4. Do not output anything after </answer>.
5. Even if uncertain, you must choose exactly one answer.
6. Take your time to carefully evaluate all ten options before committing to your final decision.
"""

USER_PROMPT_TEMPLATE = """Subject: {subject}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
E. {choice_e}
F. {choice_f}
G. {choice_g}
H. {choice_h}
I. {choice_i}
J. {choice_j}

Respond exactly in this format:

<thinking>
Core concept: ...
Option A: ...
Option B: ...
Option C: ...
Option D: ...
Option E: ...
Option F: ...
Option G: ...
Option H: ...
Option I: ...
Option J: ...
Final decision: ...
</thinking>
<answer>X</answer>
"""


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = MAX_OUTPUT_TOKENS
    logprobs: int = MAX_TOP_LOGPROBS
    stop: tuple[str, ...] = ("</answer>",)
    include_stop_str_in_output: bool = True
    seed: int = 0
    n: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class AssetMetadata:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_config: str
    dataset_revision: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "unknown"


def default_run_id(split: str, subject: str | None = None) -> str:
    base = f"gemma3-12b-it__mmlu-pro-{split}__{PROMPT_VARIANT}__{utc_timestamp()}"
    if subject:
        return f"{base}__subject-{slugify(subject)}"
    return base


def runtime_max_num_seqs(split: str) -> int:
    if split in {"validation"}:
        return SMOKE_MAX_NUM_SEQS
    return MAX_NUM_SEQS
