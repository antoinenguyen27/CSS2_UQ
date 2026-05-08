import math

import pandas as pd

from UQ.SE.evaluation import build_results_frame, summarize_partition


def test_build_results_frame_all_agreement_has_max_certainty():
    df = build_results_frame(
        [
            {
                "example_id": "e1",
                "subject": "math",
                "source_split": "test",
                "gold_answer": "A",
                "is_correct": 1,
                "valid_answers": ["A"] * 10,
                "missing_sample_count": 0,
                "drop_reason": "",
                "sampled_token_logprobs": [-1.0, -2.0],
            }
        ],
        eval_split="full",
    )
    assert df.loc[0, "certainty"] == 1.0
    assert df.loc[0, "entropy"] == 0.0


def test_build_results_frame_uniform_answers_has_zero_certainty():
    df = build_results_frame(
        [
            {
                "example_id": "e2",
                "subject": "physics",
                "source_split": "test",
                "gold_answer": "A",
                "is_correct": 0,
                "valid_answers": ["A", "B", "C", "D"] * 2 + ["A", "B"],
                "missing_sample_count": 0,
                "drop_reason": "",
                "sampled_token_logprobs": [-3.0, -3.5],
            }
        ],
        eval_split="full",
    )
    certainty = df.loc[0, "certainty"]
    assert math.isclose(certainty, 0.014524702772665599, rel_tol=1e-9)


def test_summarize_partition_tracks_dropped_questions():
    results_df = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "subject": "math",
                "source_split": "test",
                "eval_split": "test",
                "gold_answer": "A",
                "is_correct": 1,
                "n_valid_samples": 10,
                "n_A": 10,
                "n_B": 0,
                "n_C": 0,
                "n_D": 0,
                "p_A": 1.0,
                "p_B": 0.0,
                "p_C": 0.0,
                "p_D": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "certainty": 1.0,
                "missing_sample_count": 0,
                "drop_reason": "",
                "sampled_token_logprobs": [-1.0, -2.0],
            },
            {
                "example_id": "e2",
                "subject": "history",
                "source_split": "test",
                "eval_split": "test",
                "gold_answer": "B",
                "is_correct": 0,
                "n_valid_samples": 3,
                "n_A": 1,
                "n_B": 1,
                "n_C": 1,
                "n_D": 0,
                "p_A": 1 / 3,
                "p_B": 1 / 3,
                "p_C": 1 / 3,
                "p_D": 0.0,
                "entropy": None,
                "normalized_entropy": None,
                "certainty": None,
                "missing_sample_count": 7,
                "drop_reason": "fewer_than_min_valid_samples",
                "sampled_token_logprobs": [-3.0, -4.0],
            },
        ]
    )
    summary = summarize_partition(results_df)
    assert summary["n_questions_total"] == 2
    assert summary["n_questions_evaluated"] == 1
    assert summary["n_questions_dropped"] == 1
    assert summary["drop_reasons"] == {"fewer_than_min_valid_samples": 1}
