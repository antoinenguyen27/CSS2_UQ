from mmlu_trace_eval.prompting import build_messages, parse_answer


def test_parse_answer_success():
    parsed = parse_answer(
        "<thinking>\nCore concept: test\n</thinking>\n<answer>\n b \n</answer>"
    )
    assert parsed.parse_success is True
    assert parsed.predicted_answer == "B"
    assert parsed.parse_error == ""
    assert "Core concept" in parsed.thinking_text
    assert parsed.answer_text == "b"


def test_parse_answer_missing_tag():
    parsed = parse_answer("<thinking>stuff</thinking>")
    assert parsed.parse_success is False
    assert parsed.predicted_answer is None
    assert parsed.parse_error == "missing_answer_tag"


def test_parse_answer_multiple_tags_uses_first():
    parsed = parse_answer(
        "<thinking>stuff</thinking><answer>C</answer><answer>A</answer>"
    )
    assert parsed.parse_success is True
    assert parsed.predicted_answer == "C"
    assert parsed.parse_error == "multiple_answer_tags"


def test_build_messages_shapes_prompt():
    example = {
        "subject": "astronomy",
        "question": "What is the closest planet to the Sun?",
        "choices": ["Mercury", "Venus", "Earth", "Mars"],
    }
    messages = build_messages(example)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Subject: astronomy" in messages[1]["content"]
    assert "A. Mercury" in messages[1]["content"]
