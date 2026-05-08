from UQ.SE.config import build_messages, parse_answer, runtime_max_num_seqs, SMOKE_MAX_NUM_SEQS, MAX_NUM_SEQS


def test_build_messages_shapes_four_choice_prompt():
    messages = build_messages(
        {
            "subject": "astronomy",
            "question": "Which planet is closest to the Sun?",
            "choices": ["Mercury", "Venus", "Earth", "Mars"],
        }
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "A. Mercury" in messages[1]["content"]


def test_parse_answer_accepts_lowercase_and_multiple_tags():
    parsed = parse_answer("<thinking>test</thinking><answer>b</answer><answer>A</answer>")
    assert parsed.parse_success is True
    assert parsed.predicted_answer == "B"
    assert parsed.parse_error == "multiple_answer_tags"


def test_runtime_max_num_seqs_uses_smoke_profile_for_small_limit():
    assert runtime_max_num_seqs(32) == SMOKE_MAX_NUM_SEQS
    assert runtime_max_num_seqs(None) == MAX_NUM_SEQS
