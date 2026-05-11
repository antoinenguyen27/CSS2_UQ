from UQ.SE_PRO.config import (
    HF_DATA_FILE,
    HF_DATASET,
    MAX_MODEL_LEN,
    DEFAULT_MAX_TOKENS,
    MAX_NUM_SEQS,
    SMOKE_MAX_NUM_SEQS,
    answer_letters_for_choices,
    build_messages,
    parse_answer,
    runtime_max_num_seqs,
)


def test_default_dataset_uses_mmlu_pro_trace_examples():
    assert HF_DATASET == "auhsoJ69/mmlu-pro-traces"
    assert HF_DATA_FILE == "examples.parquet"


def test_pro_runtime_limits_match_longer_trace_collection():
    assert DEFAULT_MAX_TOKENS == 2048
    assert MAX_MODEL_LEN == 3072


def test_answer_letters_follow_variable_choice_count():
    assert answer_letters_for_choices(["a", "b", "c"]) == ("A", "B", "C")
    assert answer_letters_for_choices(list("abcdefghij")) == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    )


def test_build_messages_shapes_variable_choice_prompt():
    messages = build_messages(
        {
            "subject": "business",
            "question": "Pick one.",
            "choices": ["alpha", "bravo", "charlie", "delta", "echo"],
        }
    )
    content = messages[1]["content"]
    assert "A. alpha" in content
    assert "E. echo" in content
    assert "F." not in content
    assert "Option E: ..." in content
    assert "Option F: ..." not in content


def test_parse_answer_accepts_a_through_j_and_rejects_k():
    for letter in "ABCDEFGHIJ":
        parsed = parse_answer(f"<thinking>ok</thinking><answer>{letter.lower()}</answer>")
        assert parsed.parse_success is True
        assert parsed.predicted_answer == letter

    parsed = parse_answer("<thinking>ok</thinking><answer>K</answer>")
    assert parsed.parse_success is False
    assert parsed.predicted_answer is None


def test_runtime_max_num_seqs_uses_smoke_profile_for_small_eval_request():
    assert runtime_max_num_seqs(32) == SMOKE_MAX_NUM_SEQS
    assert runtime_max_num_seqs(None) == MAX_NUM_SEQS
