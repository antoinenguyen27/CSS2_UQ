import math

from UQ.SE_PRO.evaluation import build_results_frame, semantic_entropy_from_answers


def test_semantic_entropy_normalizes_by_question_choice_count():
    three_choice = semantic_entropy_from_answers(
        ["A", "B", "C", "A", "B", "C"],
        answer_letters=("A", "B", "C"),
    )
    ten_choice = semantic_entropy_from_answers(
        list("ABCDEFGHIJ"),
        answer_letters=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
    )

    assert math.isclose(three_choice["normalized_entropy"], 1.0, rel_tol=1e-9)
    assert math.isclose(three_choice["certainty"], 0.0, abs_tol=1e-12)
    assert math.isclose(ten_choice["normalized_entropy"], 1.0, rel_tol=1e-9)
    assert math.isclose(ten_choice["certainty"], 0.0, abs_tol=1e-12)


def test_build_results_frame_emits_a_through_j_counts_and_probabilities():
    df = build_results_frame(
        [
            {
                "example_id": "e1",
                "subject": "math",
                "source_split": "test",
                "gold_answer": "J",
                "is_correct": 1,
                "choices": list("abcdefghij"),
                "valid_answers": list("ABCDEFGHIJ"),
                "missing_sample_count": 0,
                "drop_reason": "",
                "sampled_token_logprobs": [-1.0, -2.0],
            }
        ],
        eval_split="test",
    )

    assert df.loc[0, "n_J"] == 1
    assert df.loc[0, "p_J"] == 0.1
    assert df.loc[0, "n_choices"] == 10
