import pytest

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
    assert "or J" in system


def test_parse_answer_accepts_a_through_j():
    for letter in "ABCDEFGHIJ":
        parsed = parse_answer(f"<thinking>ok</thinking><answer>{letter}</answer>")
        assert parsed.parse_success is True
        assert parsed.predicted_answer == letter
        assert parsed.parse_error == ""


def test_parse_answer_rejects_k():
    parsed = parse_answer("<thinking>ok</thinking><answer>K</answer>")
    assert parsed.parse_success is False
    assert parsed.predicted_answer is None
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


def test_build_messages_raises_on_too_few_choices():
    example = {"subject": "math", "question": "Q?", "choices": ["a", "b", "c", "d"]}
    with pytest.raises(IndexError):
        build_messages(example)


def test_parse_answer_multiple_tags_uses_first():
    parsed = parse_answer(
        "<thinking>stuff</thinking><answer>C</answer><answer>A</answer>"
    )
    assert parsed.parse_success is True
    assert parsed.predicted_answer == "C"
    assert parsed.parse_error == "multiple_answer_tags"
