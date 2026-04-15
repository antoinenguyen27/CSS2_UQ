"""
HALT preprocessing pipeline.

Loads the CSS2_UQ dataset from HuggingFace, validates and filters rows,
and computes (N, T_max, 25) feature sequences with binary is_correct labels.

Called from training code, e.g.:
    from UQ.halt.preprocessing.preprocess_halt import preprocess
    features, labels = preprocess()

Returns:
    features: float32 array of shape (N, MAX_LEN, 25)
    labels:   float32 array of shape (N,) — binary is_correct (0 or 1)
"""

import numpy as np
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET  = "antoine3/CSS2_UQ"
MAX_LEN     = 192
TOP_K       = 20
FEATURE_DIM = 5 + TOP_K   # 5 engineered + 20 raw log-probs

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_step_features(logprobs_step: np.ndarray):
    """
    Compute 5 engineered uncertainty features for one token step.

    Args:
        logprobs_step: (20,) top-20 log-probs; index 0 = selected token

    Returns:
        feats: (5,) [avg_logprob, rank_proxy, h_overall, h_alts, h_dec]
        h_dec: scalar decision entropy (needed externally for delta)
    """
    eps = 1e-10

    # Numerically stable truncated softmax
    m = np.max(logprobs_step)
    probs = np.exp(logprobs_step - m)
    probs = probs / (probs.sum() + eps)

    # 1. Average log-probability
    avg_logprob = float(np.mean(logprobs_step))

    # 2. Rank proxy of selected token
    rank_proxy = float(1 + np.sum(logprobs_step[1:] > logprobs_step[0]))

    # 3. Overall entropy over top-k distribution
    h_overall = float(-np.sum(probs * np.log(probs + eps)))

    # 4. Alternatives-only entropy
    alts_probs = probs[1:]
    alts_probs = alts_probs / (alts_probs.sum() + eps)
    h_alts = float(-np.sum(alts_probs * np.log(alts_probs + eps)))

    # 5. Binary decision entropy (log-domain for numerical stability)
    best_alt_lp = float(np.max(logprobs_step[1:]))
    log_sum = np.logaddexp(float(logprobs_step[0]), best_alt_lp)
    log_pc = float(logprobs_step[0]) - log_sum
    log_1_pc = best_alt_lp - log_sum
    pc = float(np.clip(np.exp(log_pc), eps, 1 - eps))
    h_dec = float(-(pc * log_pc + (1 - pc) * log_1_pc))

    feats = np.array([avg_logprob, rank_proxy, h_overall, h_alts, h_dec], dtype=np.float32)
    return feats, h_dec


def build_feature_sequence(top20_logprobs) -> np.ndarray | None:
    """
    Build the padded (MAX_LEN, 25) feature matrix for one example.

    Returns None if the sequence is invalid and should be dropped.

    Feature layout per timestep:
        [avg_logprob, rank_proxy, h_overall, h_alts, delta_h_dec, lp_0..lp_19]
    """
    if top20_logprobs is None or len(top20_logprobs) == 0:
        return None

    padded = np.zeros((MAX_LEN, FEATURE_DIM), dtype=np.float32)
    prev_h_dec = 0.0
    valid_steps = 0

    for t, step in enumerate(top20_logprobs):
        if t >= MAX_LEN:
            break
        if step is None:
            continue

        step = np.array(step, dtype=np.float32)

        if step.ndim != 1 or len(step) == 0:
            continue
        if np.any(np.isnan(step)) or np.any(np.isinf(step)):
            continue
        if len(step) < TOP_K:
            step = np.pad(step, (0, TOP_K - len(step)), constant_values=-1e9)
        step = step[:TOP_K]

        stat_feats, h_dec = compute_step_features(step)

        delta_h_dec = h_dec - prev_h_dec
        stat_feats[4] = delta_h_dec
        prev_h_dec = h_dec

        padded[t, :5] = stat_feats
        padded[t, 5:] = step
        valid_steps += 1

    if valid_steps == 0:
        return None

    return padded


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------

def validate_row(row) -> tuple[bool, str]:
    """
    Check a dataset row is usable for training.

    Returns (is_valid, reason_if_invalid).
    """
    if not row.get("parse_success", False):
        return False, "parse_success=False"
    if row.get("top20_token_logprobs") is None or len(row["top20_token_logprobs"]) == 0:
        return False, "empty top20_token_logprobs"
    if row.get("is_correct") is None:
        return False, "missing is_correct"
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess(hf_dataset: str = HF_DATASET) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, validate, and featurize the CSS2_UQ dataset.

    Returns:
        features: float32 array of shape (N, MAX_LEN, 25)
        labels:   float32 array of shape (N,) — binary is_correct (0 or 1)
    """
    print(f"Loading dataset: {hf_dataset}")
    ds = load_dataset(hf_dataset, data_files="examples.parquet")
    df = ds["train"].to_pandas()
    print(f"Total rows: {len(df)}")

    features_list = []
    labels_list   = []
    skipped       = {}

    for _, row in df.iterrows():
        is_valid, reason = validate_row(row)
        if not is_valid:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        features = build_feature_sequence(row["top20_token_logprobs"])
        if features is None:
            skipped["bad feature sequence"] = skipped.get("bad feature sequence", 0) + 1
            continue

        features_list.append(features)
        labels_list.append(float(row["is_correct"]))

    n = len(features_list)
    print(f"Rows kept:    {n}")
    print(f"Rows skipped: {sum(skipped.values())}")
    for reason, count in skipped.items():
        if count > 0:
            print(f"  {reason}: {count}")

    if n == 0:
        raise RuntimeError("No valid rows found — check dataset and filters.")

    features_arr = np.stack(features_list, axis=0)       # (N, MAX_LEN, 25)
    labels_arr   = np.array(labels_list, dtype=np.int8)  # (N,)

    n_correct   = int(labels_arr.sum())
    n_incorrect = n - n_correct
    print(f"Label balance — correct: {n_correct} | incorrect: {n_incorrect}")
    print(f"Feature matrix shape: {features_arr.shape}")

    return features_arr, labels_arr