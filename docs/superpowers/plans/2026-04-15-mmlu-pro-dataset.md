# MMLU Pro Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `mmlu_pro_trace_eval` package that runs the same token log-prob trajectory collection pipeline on `TIGER-Lab/MMLU-Pro` (10 choices) as `mmlu_trace_eval` does on `cais/mmlu` (4 choices), producing output Parquet files with the identical downstream schema.

**Architecture:** New Python package `mmlu_pro_trace_eval/` living alongside `mmlu_trace_eval/` inside `data_work/`. It duplicates only the four files that need dataset-specific changes (`config`, `dataset`, `prompting`, `modal_app`) and imports the three fully agnostic modules (`schema`, `storage`, `batching`) directly from `mmlu_trace_eval`. The output schema — both `examples.parquet` and `token_steps.parquet` — is byte-for-byte compatible with the existing MMLU schema; the only semantic differences are that `choices` holds 10 strings instead of 4 and `gold_answer` ranges over A–J instead of A–D.

**Tech Stack:** Python 3.11, Modal, vLLM 0.18.0, HuggingFace `datasets`, PyArrow — identical stack to the existing package.

---

## Context: What MMLU Pro Looks Like on HuggingFace

- **Dataset ID:** `TIGER-Lab/MMLU-Pro`
- **No sub-configs** (unlike `cais/mmlu` which uses `"all"`). Load with just the dataset ID; omit the `name` argument.
- **Splits:** `test` and `validation` only — there is no `dev` split.
- **Column mapping** (MMLU Pro → our internal names):
  - `options` (list[str], length 10) → `choices`
  - `category` (str) → `subject`
  - `answer` (str, already a letter like `"A"`) → `gold_answer` (used directly, no index conversion)
  - `question` → `question`
  - `question_id` → used to derive `question_idx` (enumerate position in split, same as MMLU)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `data_work/mmlu_pro_trace_eval/__init__.py` | Empty package marker |
| Create | `data_work/mmlu_pro_trace_eval/config.py` | All MMLU Pro constants: dataset ID, 10-choice prompts, increased token limits, A–J answer patterns |
| Create | `data_work/mmlu_pro_trace_eval/dataset.py` | `build_example_id` with `mmlu_pro__` prefix; `normalized_dataset_dir`; `load_materialized_split`; `filter_examples` — imports pro config |
| Create | `data_work/mmlu_pro_trace_eval/prompting.py` | `build_messages` for 10 choices; `parse_answer` using A–J pattern; `render_prompt`; `count_prompt_tokens` — imports pro config |
| Create | `data_work/mmlu_pro_trace_eval/modal_app.py` | Modal `prepare_assets` (maps `options`/`category`/letter `answer`), `Evaluator` class, `main` entrypoint — imports pro config + shared `schema`/`storage`/`batching` from `mmlu_trace_eval` |
| Modify | `data_work/pyproject.toml` | Add `"mmlu_pro_trace_eval*"` to `tool.setuptools.packages.find.include` |
| Create | `data_work/tests/test_pro_config.py` | Tests for `runtime_max_num_seqs`, `default_run_id`, `ANSWER_LETTERS` length |
| Create | `data_work/tests/test_pro_prompting.py` | Tests for `build_messages` (10-choice shape) and `parse_answer` (A–J acceptance) |

**Shared (imported, not copied):**
- `mmlu_trace_eval/schema.py` — format-agnostic, works unchanged
- `mmlu_trace_eval/storage.py` — format-agnostic, works unchanged
- `mmlu_trace_eval/batching.py` — format-agnostic, works unchanged

---

## Key Constant Differences vs MMLU

| Constant | MMLU value | MMLU Pro value | Reason |
|----------|-----------|----------------|--------|
| `APP_NAME` | `"mmlu-trace-eval"` | `"mmlu-pro-trace-eval"` | Separate Modal app |
| `VOLUME_NAME` | `"mmlu-trace-volume"` | `"mmlu-pro-trace-volume"` | Separate Modal volume |
| `DATASET_ID` | `"cais/mmlu"` | `"TIGER-Lab/MMLU-Pro"` | Different dataset |
| `DATASET_CONFIG` | `"all"` | `"default"` | Path slug only; not passed to `load_dataset` for MMLU Pro |
| `SUPPORTED_SPLITS` | `("test", "validation", "dev")` | `("test", "validation")` | MMLU Pro has no `dev` split |
| `ANSWER_LETTERS` | `("A","B","C","D")` | `("A","B","C","D","E","F","G","H","I","J")` | 10 choices |
| `ANSWER_PATTERN` | `r"<answer>\s*([ABCDabcd])\s*</answer>"` | `r"<answer>\s*([A-Ja-j])\s*</answer>"` | 10 choices |
| `MAX_OUTPUT_TOKENS` | `192` | `512` | More choices need more reasoning tokens |
| `MAX_MODEL_LEN` | `2048` | `3072` | Longer prompts + outputs; extra headroom |
| `PROMPT_VARIANT` | `"reasoning_v1"` | `"reasoning_v1"` | Same variant concept, different content — `prompt_hash` distinguishes runs |

---

## Task 1: Create the package skeleton and config

**Files:**
- Create: `data_work/mmlu_pro_trace_eval/__init__.py`
- Create: `data_work/mmlu_pro_trace_eval/config.py`
- Modify: `data_work/pyproject.toml`

- [ ] **Step 1: Write `__init__.py`**

```python
# data_work/mmlu_pro_trace_eval/__init__.py
```
(empty file — just the package marker)

- [ ] **Step 2: Write `config.py`**

```python
# data_work/mmlu_pro_trace_eval/config.py
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
DATASET_CONFIG = "default"   # path slug only — not passed to load_dataset
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
MAX_OUTPUT_TOKENS = 512          # increased from 192 (MMLU) to accommodate 10-choice reasoning
MAX_NUM_SEQS = 224
SMOKE_MAX_NUM_SEQS = 64
MAX_NUM_BATCHED_TOKENS = 98_304
MAX_MODEL_LEN = 3072             # increased from 2048 (MMLU) — longer prompts + outputs
SHARD_SIZE = 256

DEFAULT_EVALUATOR_CPU = 6
DEFAULT_PREPARE_CPU = 6
DEFAULT_EVALUATOR_MEMORY_MB = 65_536
DEFAULT_PREPARE_MEMORY_MB = 32_768

SUPPORTED_SPLITS = ("test", "validation")   # MMLU Pro has no dev split

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
    import re as _re
    cleaned = _re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
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
```

- [ ] **Step 3: Patch `pyproject.toml`**

In `data_work/pyproject.toml`, change the `packages.find` section from:

```toml
[tool.setuptools.packages.find]
include = ["mmlu_trace_eval*"]
```

to:

```toml
[tool.setuptools.packages.find]
include = ["mmlu_trace_eval*", "mmlu_pro_trace_eval*"]
```

- [ ] **Step 4: Write failing tests for config**

Create `data_work/tests/test_pro_config.py`:

```python
from mmlu_pro_trace_eval.config import (
    ANSWER_LETTERS,
    MAX_OUTPUT_TOKENS,
    MAX_MODEL_LEN,
    SMOKE_MAX_NUM_SEQS,
    SUPPORTED_SPLITS,
    runtime_max_num_seqs,
    default_run_id,
)


def test_answer_letters_has_ten_entries():
    assert len(ANSWER_LETTERS) == 10
    assert ANSWER_LETTERS[0] == "A"
    assert ANSWER_LETTERS[-1] == "J"


def test_max_output_tokens_increased():
    assert MAX_OUTPUT_TOKENS >= 512


def test_max_model_len_increased():
    assert MAX_MODEL_LEN >= 3072


def test_supported_splits_has_no_dev():
    assert "dev" not in SUPPORTED_SPLITS
    assert "test" in SUPPORTED_SPLITS
    assert "validation" in SUPPORTED_SPLITS


def test_runtime_max_num_seqs_smoke_on_validation():
    assert runtime_max_num_seqs("validation") == SMOKE_MAX_NUM_SEQS


def test_runtime_max_num_seqs_full_on_test():
    from mmlu_pro_trace_eval.config import MAX_NUM_SEQS
    assert runtime_max_num_seqs("test") == MAX_NUM_SEQS


def test_default_run_id_contains_mmlu_pro():
    run_id = default_run_id("test")
    assert "mmlu-pro" in run_id
```

- [ ] **Step 5: Run tests to verify they fail (package not importable yet)**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest tests/test_pro_config.py -v
```

Expected: `ModuleNotFoundError` for `mmlu_pro_trace_eval`.

- [ ] **Step 6: Reinstall the package so the new package is discoverable**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
pip install -e .
```

- [ ] **Step 7: Run tests again — they should now pass**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest tests/test_pro_config.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
git add mmlu_pro_trace_eval/__init__.py mmlu_pro_trace_eval/config.py pyproject.toml tests/test_pro_config.py
git commit -m "feat: add mmlu_pro_trace_eval config skeleton with 10-choice constants"
```

---

## Task 2: Create `dataset.py` and `prompting.py`

**Files:**
- Create: `data_work/mmlu_pro_trace_eval/dataset.py`
- Create: `data_work/mmlu_pro_trace_eval/prompting.py`
- Create: `data_work/tests/test_pro_prompting.py`

- [ ] **Step 1: Write failing tests for prompting**

Create `data_work/tests/test_pro_prompting.py`:

```python
from mmlu_pro_trace_eval.prompting import build_messages, parse_answer


def test_build_messages_has_ten_choices():
    example = {
        "subject": "math",
        "question": "What is 2+2?",
        "choices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    }
    messages = build_messages(example)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert "Subject: math" in content
    assert "A. 1" in content
    assert "J. 10" in content


def test_build_messages_system_mentions_ten_options():
    example = {
        "subject": "science",
        "question": "Q?",
        "choices": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    }
    messages = build_messages(example)
    system = messages[0]["content"]
    assert "J" in system


def test_parse_answer_accepts_a_through_j():
    for letter in "ABCDEFGHIJ":
        parsed = parse_answer(f"<thinking>ok</thinking><answer>{letter}</answer>")
        assert parsed.parse_success is True
        assert parsed.predicted_answer == letter


def test_parse_answer_rejects_k():
    parsed = parse_answer("<thinking>ok</thinking><answer>K</answer>")
    assert parsed.parse_success is False
    assert parsed.parse_error == "missing_answer_tag"


def test_parse_answer_lowercases_are_upcased():
    parsed = parse_answer("<thinking>ok</thinking><answer>e</answer>")
    assert parsed.parse_success is True
    assert parsed.predicted_answer == "E"


def test_parse_answer_missing_tag():
    parsed = parse_answer("<thinking>stuff</thinking>")
    assert parsed.parse_success is False
    assert parsed.predicted_answer is None
    assert parsed.parse_error == "missing_answer_tag"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest tests/test_pro_prompting.py -v
```

Expected: `ModuleNotFoundError` for `mmlu_pro_trace_eval.prompting`.

- [ ] **Step 3: Write `dataset.py`**

```python
# data_work/mmlu_pro_trace_eval/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import DATASET_CONFIG, DATASET_ID, SUPPORTED_SPLITS, slugify


def normalized_dataset_dir(dataset_root: str, dataset_revision: str) -> Path:
    return Path(dataset_root) / DATASET_ID.replace("/", "__") / DATASET_CONFIG / dataset_revision


def split_parquet_path(dataset_root: str, dataset_revision: str, split: str) -> Path:
    return normalized_dataset_dir(dataset_root, dataset_revision) / f"{split}.parquet"


def load_materialized_split(dataset_root: str, dataset_revision: str, split: str) -> list[dict[str, Any]]:
    path = split_parquet_path(dataset_root, dataset_revision, split)
    table = pq.read_table(path)
    return table.to_pylist()


def filter_examples(
    examples: list[dict[str, Any]],
    *,
    subject: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = examples
    if subject:
        filtered = [row for row in filtered if row["subject"] == subject]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def build_example_id(split: str, subject: str, question_idx: int) -> str:
    return f"mmlu_pro__{split}__{question_idx:05d}__{slugify(subject)}"
```

- [ ] **Step 4: Write `prompting.py`**

```python
# data_work/mmlu_pro_trace_eval/prompting.py
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
```

- [ ] **Step 5: Run tests — expect pass**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest tests/test_pro_prompting.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Run the full test suite to catch any regressions**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest -v
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
git add mmlu_pro_trace_eval/dataset.py mmlu_pro_trace_eval/prompting.py tests/test_pro_prompting.py
git commit -m "feat: add mmlu_pro_trace_eval dataset and prompting modules for 10-choice format"
```

---

## Task 3: Create `modal_app.py`

**Files:**
- Create: `data_work/mmlu_pro_trace_eval/modal_app.py`

This is the largest file. It is structurally identical to `mmlu_trace_eval/modal_app.py` with four targeted changes:

1. All config imports come from `.config` (pro config).
2. `dataset.*` imports come from `.dataset` (pro dataset with `mmlu_pro__` IDs).
3. `prompting.*` imports come from `.prompting` (pro prompting, 10 choices).
4. `schema`, `storage`, `batching` are **imported from `mmlu_trace_eval`** directly (shared, agnostic modules).
5. `prepare_assets` maps MMLU Pro columns differently: `options`→`choices`, `category`→`subject`, `answer` used directly as a letter (no `"ABCD"[int(...)]` conversion). The `load_dataset` call omits `DATASET_CONFIG` (MMLU Pro has no named sub-config).

- [ ] **Step 1: Write `modal_app.py`**

```python
# data_work/mmlu_pro_trace_eval/modal_app.py
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import modal

# Pro-specific modules
from .config import (
    APP_NAME,
    CACHE_ROOT,
    DATASET_ID,
    DATASET_ROOT,
    DEFAULT_EVALUATOR_CPU,
    DEFAULT_EVALUATOR_MEMORY_MB,
    DEFAULT_PREPARE_CPU,
    DEFAULT_PREPARE_MEMORY_MB,
    DecodeConfig,
    HF_DATASETS_CACHE,
    HF_HOME,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MODEL_CACHE_DIR,
    MODEL_ID,
    MODEL_ROOT,
    PROMPT_VARIANT,
    RUNS_ROOT,
    SECRET_NAME,
    SHARD_SIZE,
    SUPPORTED_SPLITS,
    SYSTEM_PROMPT,
    TRANSFORMERS_CACHE,
    USER_PROMPT_TEMPLATE,
    VLLM_CACHE_ROOT,
    VOLUME_NAME,
    VOLUME_ROOT,
    default_run_id,
    runtime_max_num_seqs,
)
from .dataset import build_example_id, filter_examples, load_materialized_split, normalized_dataset_dir
from .prompting import build_messages, count_prompt_tokens, render_prompt

# Shared agnostic modules imported directly from mmlu_trace_eval
from mmlu_trace_eval.batching import PreparedRequest
from mmlu_trace_eval.schema import (
    build_example_record,
    build_token_step_records,
    compute_token_surfaces,
    normalize_topk,
    segment_token_surfaces,
    validate_example_record,
)
from mmlu_trace_eval.storage import (
    MetricsAccumulator,
    build_manifest,
    compact_shards,
    ensure_dir,
    iter_example_records_from_shards,
    list_completed_example_ids,
    read_json,
    write_json,
    write_shard,
)


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name(SECRET_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .env(
        {
            "HF_HOME": HF_HOME,
            "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE,
            "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
            "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .uv_pip_install(
        "datasets==4.0.0",
        "huggingface_hub>=0.34.0,<1",
        "pyarrow>=18,<22",
        "transformers>=4.50.0,<5",
        "vllm==0.18.0",
    )
    .add_local_python_source("mmlu_trace_eval")
    .add_local_python_source("mmlu_pro_trace_eval")
)

app = modal.App(APP_NAME)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("modal", "vllm", "transformers", "datasets", "huggingface_hub", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _asset_metadata_path() -> Path:
    return Path(VOLUME_ROOT) / "asset_metadata_pro.json"


def _model_dir(model_revision: str) -> Path:
    # Reuse the same cached model directory as the MMLU run (same model).
    return Path(MODEL_ROOT) / MODEL_ID.replace("/", "__") / model_revision


def _dataset_ready_path(dataset_revision: str) -> Path:
    return normalized_dataset_dir(DATASET_ROOT, dataset_revision) / ".ready"


def _model_ready_path(model_revision: str) -> Path:
    return _model_dir(model_revision) / ".ready"


@app.function(
    image=image,
    cpu=DEFAULT_PREPARE_CPU,
    memory=DEFAULT_PREPARE_MEMORY_MB,
    timeout=60 * 60 * 2,
    startup_timeout=60 * 20,
    secrets=[secret],
    volumes={VOLUME_ROOT: volume},
)
def prepare_assets() -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    ensure_dir(Path(MODEL_ROOT))
    ensure_dir(Path(DATASET_ROOT))
    ensure_dir(Path(CACHE_ROOT))

    api = HfApi(token=os.environ["HF_TOKEN"])
    model_revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha

    model_dir = _model_dir(model_revision)
    model_ready = _model_ready_path(model_revision)
    if not model_ready.exists():
        ensure_dir(model_dir)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=model_revision,
            local_dir=str(model_dir),
            token=os.environ["HF_TOKEN"],
        )
        model_ready.write_text(model_revision + "\n")
        volume.commit()

    dataset_dir = normalized_dataset_dir(DATASET_ROOT, dataset_revision)
    dataset_ready = _dataset_ready_path(dataset_revision)
    if not dataset_ready.exists():
        ensure_dir(dataset_dir)
        for split in SUPPORTED_SPLITS:
            # MMLU Pro has no named sub-config: omit name argument to load_dataset.
            dataset = load_dataset(
                DATASET_ID,
                split=split,
                revision=dataset_revision,
                cache_dir=HF_DATASETS_CACHE,
            )
            records: list[dict[str, Any]] = []
            for question_idx, row in enumerate(dataset):
                records.append(
                    {
                        "example_id": build_example_id(split, row["category"], question_idx),
                        "question_idx": question_idx,
                        "subject": row["category"],          # MMLU Pro uses "category"
                        "question": row["question"],
                        "choices": list(row["options"]),     # MMLU Pro uses "options" (10 items)
                        "gold_answer": row["answer"],        # already a letter, e.g. "A"
                    }
                )
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(records), dataset_dir / f"{split}.parquet")
        dataset_ready.write_text(dataset_revision + "\n")
        volume.commit()

    asset_metadata = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "dataset_id": DATASET_ID,
        "dataset_config": "default",
        "dataset_revision": dataset_revision,
    }
    write_json(_asset_metadata_path(), asset_metadata)
    volume.commit()
    return asset_metadata


@app.cls(
    image=image,
    gpu="H200",
    cpu=DEFAULT_EVALUATOR_CPU,
    memory=DEFAULT_EVALUATOR_MEMORY_MB,
    timeout=60 * 60 * 24,
    startup_timeout=60 * 30,
    secrets=[secret],
    volumes={VOLUME_ROOT: volume},
)
class Evaluator:
    requested_split: str = modal.parameter(default="test")

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        volume.reload()
        asset_metadata = read_json(_asset_metadata_path())
        self.asset_metadata = asset_metadata
        self.model_dir = _model_dir(asset_metadata["model_revision"])
        self.max_num_seqs = runtime_max_num_seqs(self.requested_split)

        print(
            json.dumps(
                {
                    "event": "tokenizer_load_start",
                    "requested_split": self.requested_split,
                    "model_dir": str(self.model_dir),
                },
                sort_keys=True,
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            revision=asset_metadata["tokenizer_revision"],
        )
        print(
            json.dumps(
                {
                    "event": "tokenizer_load_complete",
                    "requested_split": self.requested_split,
                    "max_num_seqs": self.max_num_seqs,
                },
                sort_keys=True,
            )
        )
        print(
            json.dumps(
                {
                    "event": "vllm_init_start",
                    "requested_split": self.requested_split,
                    "max_model_len": MAX_MODEL_LEN,
                    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                    "max_num_seqs": self.max_num_seqs,
                    "enforce_eager": True,
                    "model_dir": str(self.model_dir),
                },
                sort_keys=True,
            )
        )
        self.llm = LLM(
            model=str(self.model_dir),
            tokenizer=str(self.model_dir),
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 0},
        )
        print(
            json.dumps(
                {
                    "event": "vllm_init_complete",
                    "requested_split": self.requested_split,
                    "max_num_seqs": self.max_num_seqs,
                    "enforce_eager": True,
                },
                sort_keys=True,
            )
        )

    @modal.method()
    def run(
        self,
        *,
        split: str,
        limit: int | None,
        subject: str | None,
        run_id: str,
        resume: bool = True,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        volume.reload()

        dataset_revision = self.asset_metadata["dataset_revision"]
        examples = load_materialized_split(DATASET_ROOT, dataset_revision, split)
        examples = filter_examples(examples, subject=subject, limit=limit)
        if not examples:
            raise ValueError("No examples remain after applying split, subject, and limit filters")

        run_dir = Path(RUNS_ROOT) / run_id
        shards_dir = run_dir / "shards"
        if run_dir.exists() and not resume and any(run_dir.iterdir()):
            raise ValueError(f"Run directory already exists and resume is false: {run_dir}")
        ensure_dir(shards_dir)

        completed_example_ids = list_completed_example_ids(run_dir) if resume else set()
        pending_examples = [row for row in examples if row["example_id"] not in completed_example_ids]

        existing_manifest_path = run_dir / "manifest.json"
        existing_manifest = read_json(existing_manifest_path) if resume and existing_manifest_path.exists() else None

        prompt_hash = hashlib.sha256(
            f"{SYSTEM_PROMPT}\n{USER_PROMPT_TEMPLATE}\n{self.tokenizer.chat_template or ''}".encode("utf-8")
        ).hexdigest()
        decode_config = DecodeConfig()
        decode_config_json = decode_config.to_json()
        started_at_utc = (
            existing_manifest["started_at_utc"]
            if existing_manifest and existing_manifest.get("started_at_utc")
            else datetime.now(timezone.utc).isoformat()
        )
        shard_inventory: list[dict[str, Any]] = []
        metrics = MetricsAccumulator()

        if resume:
            for metadata_path in sorted(shards_dir.glob("meta-*.json")):
                shard_inventory.append(read_json(metadata_path))
            for example_record in iter_example_records_from_shards(run_dir):
                metrics.update(example_record)

        manifest = build_manifest(
            run_id=run_id,
            split=split,
            subject=subject,
            requested_limit=limit,
            total_examples=len(examples),
            asset_metadata=self.asset_metadata,
            prompt_variant=PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            decode_config_json=decode_config_json,
            package_versions=_package_versions(),
            shard_inventory=shard_inventory,
            started_at_utc=started_at_utc,
            finished_at_utc=None,
        )
        write_json(run_dir / "manifest.json", manifest)

        prepared_requests = self._prepare_requests(pending_examples)

        sampling_params = SamplingParams(
            temperature=decode_config.temperature,
            top_p=decode_config.top_p,
            max_tokens=decode_config.max_tokens,
            logprobs=decode_config.logprobs,
            stop=list(decode_config.stop),
            include_stop_str_in_output=decode_config.include_stop_str_in_output,
            seed=decode_config.seed,
            n=decode_config.n,
        )

        all_prompts = [r.prompt_text for r in prepared_requests]
        all_outputs = self.llm.generate(all_prompts, sampling_params)

        next_shard_index = len(shard_inventory)
        shard_example_records: list[dict[str, Any]] = []
        shard_token_step_records: list[dict[str, Any]] = []

        for request, output in zip(prepared_requests, all_outputs, strict=True):
            example_record = self._convert_output_to_record(
                request=request,
                output=output,
                run_id=run_id,
                split=split,
                decode_config_json=decode_config_json,
            )
            validate_example_record(example_record)
            token_step_records = build_token_step_records(example_record)
            shard_example_records.append(example_record)
            shard_token_step_records.extend(token_step_records)
            metrics.update(example_record)

            if len(shard_example_records) >= SHARD_SIZE:
                metadata = write_shard(run_dir, next_shard_index, shard_example_records, shard_token_step_records)
                shard_inventory.append(metadata)
                write_json(
                    run_dir / "manifest.json",
                    build_manifest(
                        run_id=run_id,
                        split=split,
                        subject=subject,
                        requested_limit=limit,
                        total_examples=len(examples),
                        asset_metadata=self.asset_metadata,
                        prompt_variant=PROMPT_VARIANT,
                        prompt_hash=prompt_hash,
                        decode_config_json=decode_config_json,
                        package_versions=_package_versions(),
                        shard_inventory=shard_inventory,
                        started_at_utc=started_at_utc,
                        finished_at_utc=None,
                    ),
                )
                volume.commit()
                next_shard_index += 1
                shard_example_records = []
                shard_token_step_records = []

        if shard_example_records:
            metadata = write_shard(run_dir, next_shard_index, shard_example_records, shard_token_step_records)
            shard_inventory.append(metadata)
            volume.commit()

        compact_shards(run_dir, "examples", "examples.parquet")
        compact_shards(run_dir, "token_steps", "token_steps.parquet")
        metrics_payload = metrics.to_dict()
        write_json(run_dir / "metrics.json", metrics_payload)

        finished_at_utc = datetime.now(timezone.utc).isoformat()
        final_manifest = build_manifest(
            run_id=run_id,
            split=split,
            subject=subject,
            requested_limit=limit,
            total_examples=len(examples),
            asset_metadata=self.asset_metadata,
            prompt_variant=PROMPT_VARIANT,
            prompt_hash=prompt_hash,
            decode_config_json=decode_config_json,
            package_versions=_package_versions(),
            shard_inventory=shard_inventory,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
        )
        write_json(run_dir / "manifest.json", final_manifest)
        volume.commit()

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "examples_path": str(run_dir / "examples.parquet"),
            "token_steps_path": str(run_dir / "token_steps.parquet"),
            "manifest_path": str(run_dir / "manifest.json"),
            "metrics_path": str(run_dir / "metrics.json"),
            "metrics": metrics_payload,
        }

    def _prepare_requests(self, examples: list[dict[str, Any]]) -> list[PreparedRequest]:
        requests: list[PreparedRequest] = []
        for example in examples:
            messages = build_messages(example)
            prompt_text = render_prompt(self.tokenizer, messages)
            prompt_tokens = count_prompt_tokens(self.tokenizer, messages)
            requests.append(
                PreparedRequest(
                    example=example,
                    prompt_text=prompt_text,
                    prompt_tokens=prompt_tokens,
                )
            )
        return requests

    def _convert_output_to_record(
        self,
        *,
        request: PreparedRequest,
        output: Any,
        run_id: str,
        split: str,
        decode_config_json: str,
    ) -> dict[str, Any]:
        generated = output.outputs[0]
        sampled_token_ids = [int(token_id) for token_id in generated.token_ids]
        sampled_token_texts = [
            self.tokenizer.convert_ids_to_tokens(int(token_id))
            for token_id in sampled_token_ids
        ]

        top20_token_ids: list[list[int]] = []
        top20_token_texts: list[list[str]] = []
        top20_token_logprobs: list[list[float]] = []
        sampled_token_logprobs: list[float] = []
        sampled_token_ranks: list[int] = []
        cumulative_logprobs: list[float] = []
        running_logprob = 0.0

        step_logprob_iterable = generated.logprobs or [None] * len(sampled_token_ids)
        for sampled_token_id, step_logprobs in zip(sampled_token_ids, step_logprob_iterable, strict=True):
            normalized = normalize_topk(self.tokenizer, step_logprobs, sampled_token_id)
            sampled_token_logprobs.append(normalized.sampled_logprob)
            sampled_token_ranks.append(normalized.sampled_rank)
            running_logprob += normalized.sampled_logprob
            cumulative_logprobs.append(running_logprob)
            top20_token_ids.append(normalized.token_ids)
            top20_token_texts.append(normalized.token_texts)
            top20_token_logprobs.append(normalized.logprobs)

        completion_text = generated.text
        token_surfaces = compute_token_surfaces(self.tokenizer, sampled_token_ids)
        segment_ids, segment_names = segment_token_surfaces(completion_text, token_surfaces)

        return build_example_record(
            run_id=run_id,
            split=split,
            prompt_variant=PROMPT_VARIANT,
            decode_config_json=decode_config_json,
            model_name=self.asset_metadata["model_id"],
            model_revision=self.asset_metadata["model_revision"],
            tokenizer_revision=self.asset_metadata["tokenizer_revision"],
            dataset_name=self.asset_metadata["dataset_id"],
            dataset_revision=self.asset_metadata["dataset_revision"],
            prompt_text=request.prompt_text,
            example=request.example,
            completion_text=completion_text,
            sampled_token_ids=sampled_token_ids,
            sampled_token_texts=sampled_token_texts,
            sampled_token_logprobs=sampled_token_logprobs,
            sampled_token_ranks=sampled_token_ranks,
            segment_ids=segment_ids,
            segment_names=segment_names,
            cumulative_logprobs=cumulative_logprobs,
            top20_token_ids=top20_token_ids,
            top20_token_texts=top20_token_texts,
            top20_token_logprobs=top20_token_logprobs,
        )


@app.local_entrypoint()
def main(
    split: str = "test",
    limit: int = 0,
    subject: str = "",
    run_name: str = "",
    resume: bool = True,
) -> None:
    limit_value = limit if limit > 0 else None
    subject_value = subject or None

    asset_metadata = prepare_assets.remote()
    run_id = run_name or default_run_id(split=split, subject=subject_value)
    result = Evaluator(requested_split=split).run.remote(
        split=split,
        limit=limit_value,
        subject=subject_value,
        run_id=run_id,
        resume=resume,
    )
    print(json.dumps({"asset_metadata": asset_metadata, "result": result}, indent=2, sort_keys=True))
```

- [ ] **Step 2: Verify the full test suite still passes after adding the new file**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -m pytest -v
```

Expected: all tests pass (modal_app.py is not unit-tested; its correctness is verified at run time).

- [ ] **Step 3: Verify the module is importable locally (no Modal call)**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
python -c "from mmlu_pro_trace_eval import config, dataset, prompting; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/an/Documents/CSS2_UQ/data_work
git add mmlu_pro_trace_eval/modal_app.py
git commit -m "feat: add mmlu_pro_trace_eval modal app for MMLU Pro 10-choice evaluation"
```

---

## Running the MMLU Pro Eval

Once code is merged and the Modal secret `hf-auth` is present:

**Smoke test (validation split, 32 examples):**
```bash
modal run -m mmlu_pro_trace_eval.modal_app --split validation --limit 32
```

**Full test split:**
```bash
modal run --detach -m mmlu_pro_trace_eval.modal_app --split test
```

**Download artifacts:**
```bash
modal volume get mmlu-pro-trace-volume runs/<run_id>/examples.parquet ./pro_examples.parquet
modal volume get mmlu-pro-trace-volume runs/<run_id>/token_steps.parquet ./pro_token_steps.parquet
modal volume get mmlu-pro-trace-volume runs/<run_id>/manifest.json ./pro_manifest.json
```

---

## Self-Review

### Spec coverage

| Requirement | Covered by |
|-------------|-----------|
| Same output schema (examples + token_steps parquet) | `modal_app.py` imports `build_example_record`, `build_token_step_records` from shared `schema.py` — identical column set |
| 10-choice awareness in model | `config.py` `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` enumerate A–J |
| Verbose reasoning encouraged in prompt | System prompt line 6: "Take your time to carefully evaluate all ten options before committing to your final decision." |
| Increased `max_tokens` in inference config | `MAX_OUTPUT_TOKENS = 512`; `DecodeConfig.max_tokens` defaults to this |
| `MAX_MODEL_LEN` increased | `3072` (up from `2048`) |
| No changes to downstream schema | `choices` stays a list (just longer); `gold_answer` stays a string letter; all other columns identical |
| Separate Modal app / volume (no collision with MMLU run) | `APP_NAME = "mmlu-pro-trace-eval"`, `VOLUME_NAME = "mmlu-pro-trace-volume"` |
| MMLU Pro column mapping (`options`→`choices`, `category`→`subject`, letter `answer`) | `prepare_assets` in `modal_app.py` |
| No `dev` split for MMLU Pro | `SUPPORTED_SPLITS = ("test", "validation")` |

### Placeholder scan

No TBDs, TODOs, or "handle this later" items in the plan.

### Type consistency

- `PreparedRequest` imported from `mmlu_trace_eval.batching` — same dataclass, used identically.
- `build_example_record` signature is unchanged — all keyword arguments match.
- `parse_answer` returns `ParsedAnswer` dataclass — same field names as MMLU version.
- `build_example_id` returns `str` — downstream `example_id` column type unaffected.
