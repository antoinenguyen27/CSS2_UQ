# Spec for implementation of semantic entropy

---

## Implementation Requirements

### 1. Project Structure

Create a new directory `UQ/SE/` with the following layout:

```
UQ/SE/
├── se_spec.md          # This specification
├── config.py           # Shared constants and configuration
├── sampling.py         # Dataset loading and model inference (Modal side only)
├── evaluation.py       # SE computation and metrics
├── modal_se.py         # Modal app with remote @app.function() entrypoint
└── requirements.txt    # Local CLI dependency notes (primarily modal)
```

**Execution model:**
- `modal_se.py` — the Modal app module. All inference, sampling, parsing, evaluation, and artifact writing run **on Modal** (GPU) through a remote Modal function.
- `config.py`, `sampling.py`, `evaluation.py` — shared library code imported by the remote Modal function.

---

### 2. Model

**Model ID:** `google/gemma-3-12b-it`

**Model location:** Already cached in Modal volume by `prepare_assets()` in `data_work/mmlu_trace_eval/modal_app.py`. Do **not** re-download. Load directly from the volume path:

```
{VOLUME_ROOT}/models/google__gemma-3-12b-it/<revision>/
```

Where `revision` is the SHA stored in `{VOLUME_ROOT}/asset_metadata.json` (written by `prepare_assets()`). Read it at runtime rather than hardcoding.

**Tokenizer:** Load from the same volume path using `AutoTokenizer.from_pretrained()`.

**Inference backend:** `vllm.LLM` — same initialization pattern as `data_work/mmlu_trace_eval/modal_app.py`. No fallback, no local download.

---

### 3. Dataset

**Dataset ID:** `auhsoJ69/mmlu_rerun`

Load the dataset exactly as `UQ/halt/preprocessing/preprocess_halt.py` does:

```python
from datasets import load_dataset
ds = load_dataset("auhsoJ69/mmlu_rerun", data_files="examples.parquet")
df = ds["train"].to_pandas()
```

**Required columns:**
- `example_id`: unique identifier for each question
- `subject`: MMLU subject category
- `question`: the question text
- `choices`: list of 4 choices `[choice_a, choice_b, choice_c, choice_d]`
- `gold_answer`: the correct answer as a single letter `"A"`, `"B"`, `"C"`, or `"D"`
- `is_correct`: binary label (1 = model answered correctly, 0 = incorrect)

**Filter to valid rows only** — rows where:
1. `parse_success == True` (model produced a valid answer)
2. `top20_token_logprobs` is not empty
3. `is_correct` is not null

This mirrors the validation logic in `preprocess_halt.py:validate_row()`.

---

### 4. Prompting

Use the **identical prompting setup** as `data_work/mmlu_trace_eval/`:

**System prompt** (`SYSTEM_PROMPT` from `config.py`):
```
You are answering multiple-choice university exam questions for uncertainty-calibration data collection.

For each question:
1. Write your reasoning only inside <thinking>...</thinking>.
2. Inside <thinking>, use this exact order:
   - Core concept
   - Option A
   - Option B
   - Option C
   - Option D
   - Final decision
3. After the thinking section, output exactly one answer tag:
   <answer>X</answer>
   where X is A, B, C, or D.
4. Do not output anything after </answer>.
5. Even if uncertain, you must choose exactly one answer.
```

**User prompt template** (`USER_PROMPT_TEMPLATE` from `config.py`):
```
Subject: {subject}

Question:
{question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Respond exactly in this format:

<thinking>
Core concept: ...
Option A: ...
Option B: ...
Option C: ...
Option D: ...
Final decision: ...
</thinking>
<answer>X</answer>
```

Build messages and render prompts using the existing `build_messages()` and `render_prompt()` utilities from `data_work/mmlu_trace_eval/prompting.py`. Copy these functions directly into `UQ/SE/config.py` or import from `data_work` if added to `PYTHONPATH`.

---

### 5. Sampling

**Key difference from data collection:** The original MMLU trace data was collected with `temperature=0.0` (greedy). For semantic entropy, we need **non-zero temperature** to generate diverse samples.

**Generation config and sampling parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | `0.7` | Sufficient diversity for entropy estimation |
| `top_p` | `0.95` | Standard nucleus sampling |
| `max_tokens` | `512` | Allow longer reasoning chains; sufficient for full `<thinking>` + `<answer>` output |
| `n` | `10` | Number of samples per question (N=10 per the algorithm spec) |
| `stop` | `("</answer>",)` | Match the prompt format's end marker |
| `seed` | per-sample | Different seed per sample for independent draws (seed=0..9 or derived from master seed) |

**Per-sample seeds:** Use different seeds for each of the 10 samples to ensure independent draws. E.g., `seed=0` through `seed=9` for the 10 samples, or derive seeds deterministically from a master seed.

**Answer parsing:** Use the existing `parse_answer()` from `data_work/mmlu_trace_eval/prompting.py` to extract the answer letter from each completion:

```python
from prompting import parse_answer
parsed = parse_answer(completion_text)
# parsed.predicted_answer ∈ {"A", "B", "C", "D", None}
# parsed.parse_success is a bool
```

**Drop invalid samples:**
- If `parse_success == False`, discard that sample and re-sample (up to a max retries limit, e.g., 3).
- If after retries no valid answer is obtained, record that sample as `None` and handle gracefully in entropy computation (treat as missing data, not as a valid category).
- If fewer than 5 valid samples remain for a question after retries, **drop that question entirely** from the SE evaluation (insufficient samples for reliable entropy estimation).

---

### 6. Semantic Entropy Computation

For each question with at least 5 valid samples, compute:

**Step 1: Count answer frequencies**
```
n_k = number of valid samples where predicted_answer == k  (k ∈ {A, B, C, D})
N_valid = total number of valid samples for this question
```

**Step 2: Estimate category probabilities**
```
p_k = n_k / N_valid
```

**Step 3: Compute entropy**
```
H = -sum(p_k * log(p_k))  for all k where p_k > 0
```
Use `np.nan_to_num` to handle `0 * log(0) = 0`.

**Step 4: Normalize by log(K)** where K=4 (number of choices)
```
H_norm = H / log(4)
```

**Step 5: Convert to certainty**
```
certainty = 1 - H_norm
```

This yields `certainty ∈ [0, 1]`:
- `1.0` = all samples chose the same answer (maximum certainty)
- `0.0` = samples evenly distributed across all K options (maximum uncertainty)
- Values between reflect partial agreement

---

### 7. Train/Val/Test Split

To ensure **fair comparison** with 1DCNN (which uses 70/15/15) and halt (which uses 80/20 with no held-out test), SE should use the **same split as 1DCNN**:

```
train: 70%
val:   15%
test:  15%
```

Using the **same random seed (42)** and **stratified split on `is_correct`**:

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['is_correct']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_correct']
)
```

This ensures that when we compare SE (a sampling-based baseline) against 1DCNN and halt on the **same test set**, the evaluation is unbiased.

**Alternatively**, since SE is a per-question computation that does not require training, SE can be evaluated on the **full dataset** (after filtering for valid rows) as a zero-training baseline. In this case, report:
1. **Per-question certainty scores** for all questions
2. **Aggregate Brier score and accuracy** when thresholding certainty at 0.5 to produce binary predictions
3. Make clear that this is evaluated on all data without a held-out test set, for comparability with halt's evaluation protocol

**Recommendation:** Implement both modes:
- `--eval-mode full` (default): evaluate on all valid data, report aggregate metrics
- `--eval-mode split`: create the 70/15/15 split for comparison with 1DCNN, but run SE only on the held-out `test` partition

---

### 8. Evaluation Metrics

Compute and report:

**Primary metric: Brier Score**
```python
from sklearn.metrics import brier_score_loss
# Convert certainty (∈ [0,1]) to binary prediction: 1 if certainty >= 0.5 else 0
# OR use certainty directly as a probability-like score
brier = brier_score_loss(y_true_is_correct, certainty_scores)
```

Note: If using certainty directly as `predicted probability`, Brier score is `mean((y_true - certainty)²)`. If thresholding at 0.5 for binary predictions, Brier score is computed on binary labels.

**Decision:** Use **certainty as the predicted probability** (not thresholded), consistent with how halt and 1DCNN report Brier scores on continuous confidence outputs.

**Secondary metrics:**
- **Accuracy** (at 0.5 threshold): `accuracy = mean((certainty >= 0.5) == y_true_is_correct)`
- **ECE** (Expected Calibration Error): optional, if time permits
- **AUROC**: optional

**Baseline comparisons:** For each question, also compute:
1. **Majority class baseline**: always predict the most common class
2. **Token-level logprob baseline**: average logprob as a certainty score (from existing `UQ/1DCNN` code)

Print comparison table showing SE vs. these baselines.

---

### 9. Output

**Per-question output (CSV or parquet):**
```
example_id, subject, gold_answer, is_correct, n_valid_samples,
  n_A, n_B, n_C, n_D,
  p_A, p_B, p_C, p_D,
  entropy, normalized_entropy, certainty
```

**Aggregate summary (printed and written to JSON):**
```
{
  "method": "semantic_entropy",
  "n_questions_total": ...,
  "n_questions_evaluated": ...,
  "n_questions_dropped": ...,
  "drop_reason": "...",
  "brier_score": ...,
  "accuracy": ...,
  "certainty_mean": ...,
  "certainty_std": ...
}
```

Write outputs to `{VOLUME_ROOT}/runs/<run_id>/` on the Modal volume. Results are accessible locally after Modal execution completes via the volume mount or by downloading from Modal.

---

### 10. Dependencies

All SE compute runs **inside the Modal container**. Define the Modal image in `UQ/SE/modal_se.py` using the same dependency pattern as `data_work/mmlu_trace_eval/modal_app.py` so the remote function has vLLM, transformers, datasets, and sklearn available.

For the **local CLI environment** that runs `modal run`:

```
modal>=0.64
```

**Container-internal packages** (install in the Modal image):

```
torch
numpy
pandas
sklearn
datasets
huggingface_hub
transformers
vllm>=0.18.0
tqdm
```

Pin versions loosely to match `data_work/mmlu_trace_eval/modal_app.py`.

---

### 11. Execution Model

**All computation runs on Modal.** The local machine only invokes `modal run` against a remote Modal function and streams logs/results. There is no local entrypoint or local orchestration script.

**Direct remote invocation:**

```bash
modal run UQ/SE/modal_se.py --eval-mode split --num-eval-rows 154 --n-samples 10
```

Because `UQ/SE/modal_se.py` exposes a single decorated Modal function, `modal run UQ/SE/modal_se.py ...` can invoke that function directly. If more decorated functions are ever added later, invoke the specific target explicitly, e.g. `modal run UQ/SE/modal_se.py::app.run_semantic_entropy ...`.

**Modal function layout** (same image/volume/secret pattern as `data_work/mmlu_trace_eval/modal_app.py`, but using functions instead of `@app.cls`):

```python
@app.function(
    image=image,           # same image as modal_app.py (has vllm, torch, etc.)
    gpu="H200",
    secrets=[secret],      # same HF secret
    volumes={VOLUME_ROOT: volume},
)
def run_semantic_entropy(eval_mode="full", n_samples=10, temperature=0.7, top_p=0.95, max_tokens=512, seed=42, num_eval_rows=0, run_name=""):
    # All logic lives here: load cached model/tokenizer, load dataset,
    # sample, parse, evaluate, write results, return artifact paths.
    ...
```

Load the tokenizer and `vllm.LLM` inside the remote function body. Modal function startup replaces the previous class lifecycle hook.

In `split` mode, `train` and `val` are used only to define the held-out partitioning. Do not run inference on them.

**CLI interface (remote function arguments):**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | 10 | Number of samples N per question |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.95 | Nucleus sampling parameter |
| `--max-tokens` | 512 | Maximum output tokens per sample |
| `--eval-mode` | full | `full` or `split` (see Section 7) |
| `--seed` | 42 | Master random seed |
| `--output-dir` | /vol/runs | Modal volume path for results |
| `--num-eval-rows` | 0 | Cap the actual selected eval partition; `0` means no cap |
| `--run-name` | auto-generated | Optional explicit run id |

`--device` is removed — execution is always on Modal GPU. `--output-dir` refers to a path on the Modal volume (e.g. inside `{VOLUME_ROOT}/runs/`), not a local filesystem.

---

### 12. Implementation Notes

1. **Batching**: All questions are submitted to vLLM `generate()` in a **single call** per sub-batch. vLLM's internal continuous batching keeps the GPU fully utilized. For very large runs, cap with `max_num_seqs=224` and process in sub-batches of that size, consistent with `modal_app.py`.

   For SE specifically, use one prompt per sample slot with `n=1` and a deterministic per-slot seed. Build a `Sequence[SamplingParams]` matching the prompt list and submit the whole sub-batch through one `generate()` call. This preserves independent per-sample seeds while still using vLLM batching.

2. **Retry logic**: If parsing fails for a sample, re-sample with a new seed. Cap retries at 3 to avoid infinite loops on genuinely malformed outputs.

3. **Numerical stability**: When computing entropy, use `log` base `e` (natural log) consistently. The normalization by `log(K)` uses the same base.

4. **Handling K=4 always**: MMLU has exactly 4 choices per question, so `K=4` is hardcoded. Do not attempt to generalize to other K values.

5. **Reproducibility**: Set all random seeds (PyTorch, NumPy, Python's random) to the master seed for full reproducibility of the sampling process.

6. **Progress tracking**: Print progress every 100 questions showing how many have been processed and how many dropped.

7. **Empty/invalid dataset rows**: The dataset may contain rows where `parse_success=False` or the completion is otherwise unparseable. These rows must be **filtered out before sampling**, not after. Do not attempt to sample questions that already failed to parse in the original data collection run — drop them upfront. This is consistent with how `preprocess_halt.py:validate_row()` filters the data.

---

## Overall algorithm
Yes. For **MCQ**, you can turn semantic entropy into a **certainty score in ([0,1])** by treating each answer option as a semantic category, computing entropy over those categories, then inverting and normalizing it. This is a direct MCQ adaptation of the 2023 semantic entropy idea, whose original definition is entropy over **semantic classes** rather than raw strings. ([arXiv][1])

### High-level idea

Sample the model multiple times for the same question. Each sampled completion is mapped to a final answer option such as A, B, C, or D. In MCQ, that final option is the semantic category. If the samples mostly land on one option, entropy is low and certainty is high. If they are spread across several options, entropy is high and certainty is low. This follows the paper’s core principle: uncertainty should be measured over meanings, after grouping outputs that have the same meaning. ([arXiv][1])

### Step 1: sample answers

For a question (x), draw (N) samples from the model. In your case, set

[
N = 10.
]

Let the sampled outputs be

[
s^{(1)}, s^{(2)}, \dots, s^{(10)}.
]

Extract the final MCQ answer from each sample with a parser:

[
a^{(i)} \in {1,2,\dots,K},
]

where (K) is the number of answer choices.

### Step 2: define the semantic categories

In the original paper, a semantic class is a set of outputs with the same meaning. In MCQ, you do not need a separate semantic judge if your target is the final selected option. Just define one semantic class per answer option:

[
c_k = {, s : \text{final answer of } s \text{ is option } k ,}.
]

So each option is one semantic category. This is the MCQ simplification of the paper’s clustering step. ([arXiv][1])

### Step 3: estimate the category probabilities

Use the 10 samples to estimate how much mass each answer option gets.

The simplest estimator is frequency:

[
n_k = \sum_{i=1}^{10} \mathbf{1}[a^{(i)} = k]
]

[
\hat p_k = \frac{n_k}{10}.
]

These (\hat p_k) values form the estimated distribution over semantic categories.

Example: if the 10 sampled answers are

[
A, A, A, A, A, B, A, C, A, A
]

then for a 4-choice question,

[
\hat p_A = 0.8,\quad \hat p_B = 0.1,\quad \hat p_C = 0.1,\quad \hat p_D = 0.
]

### Step 4: compute semantic entropy

The original semantic entropy is ordinary entropy over semantic classes:

[
SE(x) = -\sum_{c} p(c\mid x)\log p(c\mid x).
]

In MCQ, this becomes entropy over answer options:

[
SE_{\text{MCQ}}(x) = -\sum_{k=1}^{K} \hat p_k \log \hat p_k.
]

Use the convention

[
0 \log 0 = 0.
]

This score is lowest when all 10 samples pick the same option, and highest when the sampled answers are spread evenly across all (K) options. That is exactly the uncertainty behavior you want. ([arXiv][1])

### Step 5: normalize entropy to ([0,1])

Raw entropy is not automatically between 0 and 1. Its maximum is

[
\log K
]

when the distribution is uniform across the (K) answer options.

So normalize it:

[
SE_{\text{norm}}(x) = \frac{SE_{\text{MCQ}}(x)}{\log K}.
]

Now

[
0 \le SE_{\text{norm}}(x) \le 1.
]

Interpretation:

* (0): no uncertainty
* (1): maximum uncertainty

### Step 6: convert uncertainty into certainty

Since you want **certainty**, not uncertainty, invert the normalized entropy:

[
\text{Certainty}(x) = 1 - SE_{\text{norm}}(x).
]

Expanding that fully:

[
\boxed{
\text{Certainty}(x)
===================

1 - \frac{-\sum_{k=1}^{K} \hat p_k \log \hat p_k}{\log K}
}
]

This is your final score, and it is guaranteed to lie in

[
[0,1].
]

Interpretation:

* **1** means all samples chose the same answer
* **0** means the sampled answers were maximally spread out
* values in between reflect partial agreement

### End-to-end implementation recipe

For each MCQ item:

1. Sample the model 10 times.
2. Parse the final answer choice from each sample.
3. Count how many times each option appears.
4. Convert counts to probabilities.
5. Compute entropy over those probabilities.
6. Divide by (\log K).
7. Return certainty as (1) minus that normalized entropy.

In compact form:

[
n_k = \sum_{i=1}^{10} \mathbf{1}[a^{(i)} = k]
]

[
\hat p_k = \frac{n_k}{10}
]

[
H = -\sum_{k=1}^{K} \hat p_k \log \hat p_k
]

[
\text{Certainty} = 1 - \frac{H}{\log K}.
]

### Example

Suppose (K=4), and the 10 sampled answers are distributed as

[
n_A=7,\quad n_B=2,\quad n_C=1,\quad n_D=0.
]

Then

[
\hat p = [0.7, 0.2, 0.1, 0].
]

Entropy:

[
H = -(0.7\log 0.7 + 0.2\log 0.2 + 0.1\log 0.1).
]

Normalized entropy:

[
H_{\text{norm}} = \frac{H}{\log 4}.
]

Certainty:

[
\text{Certainty} = 1 - H_{\text{norm}}.
]

This yields a certainty a bit above the midpoint: fairly confident, but not unanimous.

### Practical note

This is the cleanest MCQ adaptation of semantic entropy. It is slightly simpler than the original paper because the paper has to cluster free-form outputs into semantic groups using semantic equivalence, while in MCQ your semantic groups are already given by the answer options themselves. The original method is “entropy over meanings”; your version is “entropy over answer choices, treating each choice as a meaning-class.” ([arXiv][1])

### Source

Original paper:
**Kuhn, Gal, Farquhar (2023), “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.”** ([arXiv][1])

If you want, I can next turn this into 10 lines of pseudocode or Python.

[1]: https://arxiv.org/abs/2302.09664?utm_source=chatgpt.com "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation"
