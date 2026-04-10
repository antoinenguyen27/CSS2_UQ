"""
Test validate_row from modeling.shared.preprocess.

Run from repo root:
    python -m modeling.tests.test_validate_row
"""

from datasets import load_dataset
from modeling.preprocessing.preprocess_halt import validate_row, compute_step_features, preprocess, MAX_LEN, FEATURE_DIM
import numpy as np


def validate_row_test():
    ds = load_dataset("antoine3/CSS2_UQ", data_files="examples.parquet")
    df = ds["train"].to_pandas()
    print(f"Total rows: {len(df)}\n")

    skipped = {}
    valid = 0

    for _, row in df.iterrows():
        is_valid, reason = validate_row(row)
        if is_valid:
            valid += 1
        else:
            skipped[reason] = skipped.get(reason, 0) + 1

    total = len(df)
    print(f"Valid rows:   {valid} ({100 * valid / total:.1f}%)")
    print(f"Skipped rows: {sum(skipped.values())} ({100 * sum(skipped.values()) / total:.1f}%)")
    print("\nBreakdown of skipped:")
    for reason, count in skipped.items():
        print(f"  {reason}: {count} ({100 * count / total:.1f}%)")

def test_avg_logprob():
    """avg_logprob should be the mean of all 20 log-probs."""
    step = np.array([-1.0] * 20, dtype=np.float32)
    feats, _ = compute_step_features(step)
    assert np.isclose(feats[0], -1.0), f"Expected -1.0, got {feats[0]}"
    print("test_avg_logprob passed")


def test_rank_proxy_greedy():
    """If selected token has the highest log-prob, rank proxy should be 1."""
    step = np.array([-0.1] + [-1.0] * 19, dtype=np.float32)
    feats, _ = compute_step_features(step)
    assert feats[1] == 1.0, f"Expected 1.0, got {feats[1]}"
    print("test_rank_proxy_greedy passed")


def test_rank_proxy_non_greedy():
    """If 3 alternatives scored higher than selected token, rank proxy should be 4."""
    step = np.array([-2.0] + [-0.5, -0.6, -0.7] + [-3.0] * 16, dtype=np.float32)
    feats, _ = compute_step_features(step)
    assert feats[1] == 4.0, f"Expected 4.0, got {feats[1]}"
    print("test_rank_proxy_non_greedy passed")


def test_h_overall_uniform():
    """Uniform distribution should have maximum entropy = log(20)."""
    step = np.array([-1.0] * 20, dtype=np.float32)
    feats, _ = compute_step_features(step)
    expected = np.log(20)
    assert np.isclose(feats[2], expected, atol=1e-5), f"Expected {expected:.4f}, got {feats[2]:.4f}"
    print("test_h_overall_uniform passed")


def test_h_overall_peaked():
    """Very peaked distribution should have near-zero entropy."""
    step = np.array([0.0] + [-100.0] * 19, dtype=np.float32)
    feats, _ = compute_step_features(step)
    assert feats[2] < 0.01, f"Expected near-zero entropy, got {feats[2]}"
    print("test_h_overall_peaked passed")


def test_h_alts_uniform():
    """Uniform alternatives should have maximum alternatives entropy = log(19)."""
    step = np.array([-2.0] + [-1.0] * 19, dtype=np.float32)
    feats, _ = compute_step_features(step)
    expected = np.log(19)
    assert np.isclose(feats[3], expected, atol=1e-5), f"Expected {expected:.4f}, got {feats[3]:.4f}"
    print("test_h_alts_uniform passed")


def test_h_dec_certain():
    """If selected token dominates best alternative, h_dec should be near zero."""
    step = np.array([0.0] + [-100.0] * 19, dtype=np.float32)
    _, h_dec = compute_step_features(step)
    assert h_dec < 0.01, f"Expected near-zero h_dec, got {h_dec}"
    print("test_h_dec_certain passed")


def test_h_dec_uncertain():
    """If selected token and best alternative are equal, h_dec should be log(2) ~ 0.693."""
    step = np.array([-1.0, -1.0] + [-100.0] * 18, dtype=np.float32)
    _, h_dec = compute_step_features(step)
    assert np.isclose(h_dec, np.log(2), atol=1e-4), f"Expected {np.log(2):.4f}, got {h_dec:.4f}"
    print("test_h_dec_uncertain passed")


def test_no_nan_or_inf():
    """Output features should never contain nan or inf."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        step = rng.uniform(-10, 0, size=20).astype(np.float32)
        feats, h_dec = compute_step_features(step)
        assert not np.any(np.isnan(feats)), f"nan in feats: {feats}"
        assert not np.any(np.isinf(feats)), f"inf in feats: {feats}"
        assert not np.isnan(h_dec), f"nan in h_dec: {h_dec}"
    print("test_no_nan_or_inf passed")


def test_output_shape():
    """Features should always be shape (5,)."""
    step = np.array([-0.5] * 20, dtype=np.float32)
    feats, h_dec = compute_step_features(step)
    assert feats.shape == (5,), f"Expected shape (5,), got {feats.shape}"
    assert isinstance(h_dec, float), f"Expected float h_dec, got {type(h_dec)}"
    print("test_output_shape passed")


def test_preprocess_shapes():
    features, labels = preprocess()
 
    N = len(labels)
 
    assert features.ndim == 3, f"Expected features to be 3D, got {features.ndim}D"
    assert features.shape[0] == N, f"Expected {N} rows, got {features.shape[0]}"
    assert features.shape[1] == MAX_LEN, f"Expected MAX_LEN={MAX_LEN}, got {features.shape[1]}"
    assert features.shape[2] == FEATURE_DIM, f"Expected FEATURE_DIM={FEATURE_DIM}, got {features.shape[2]}"
    print(f"features shape: {features.shape} ✓")
 
    assert labels.ndim == 1, f"Expected labels to be 1D, got {labels.ndim}D"
    assert labels.shape[0] == N, f"Expected {N} labels, got {labels.shape[0]}"
    print(f"labels shape:   {labels.shape} ✓")
 
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    print(f"features dtype: {features.dtype} ✓")
 
    assert labels.dtype == np.int8, f"Expected int32, got {labels.dtype}"
    print(f"labels dtype:   {labels.dtype} ✓")
 
    assert set(np.unique(labels)).issubset({0, 1}), f"Labels should only contain 0 and 1, got {np.unique(labels)}"
    print(f"labels values:  {np.unique(labels)} ✓")
 

if __name__ == "__main__":
    validate_row_test()
    test_avg_logprob()
    test_rank_proxy_greedy()
    test_rank_proxy_non_greedy()
    test_h_overall_uniform()
    test_h_overall_peaked()
    test_h_alts_uniform()
    test_h_dec_certain()
    test_h_dec_uncertain()
    test_no_nan_or_inf()
    test_output_shape()
    test_preprocess_shapes()
    print("\nAll tests passed.")