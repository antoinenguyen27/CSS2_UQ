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
