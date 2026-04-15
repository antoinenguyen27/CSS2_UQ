from mmlu_pro_trace_eval.schema import build_example_record
from mmlu_trace_eval.schema import validate_example_record


def test_build_example_record_scores_e_through_j_correctly():
    """Pro build_example_record must correctly score answers E-J (not just A-D)."""
    for letter in "EFGHIJ":
        example = {
            "example_id": f"mmlu_pro__test__00000__math",
            "question_idx": 0,
            "subject": "math",
            "question": "Q?",
            "choices": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "gold_answer": letter,
        }
        record = build_example_record(
            run_id="run-1",
            split="test",
            prompt_variant="reasoning_v1",
            decode_config_json='{"seed":0}',
            model_name="google/gemma-3-12b-it",
            model_revision="rev",
            tokenizer_revision="rev",
            dataset_name="TIGER-Lab/MMLU-Pro",
            dataset_revision="rev",
            prompt_text="prompt",
            example=example,
            completion_text=f"<thinking>ok</thinking><answer>{letter}</answer>",
            sampled_token_ids=[1, 2, 3],
            sampled_token_texts=["a", "b", "c"],
            sampled_token_logprobs=[-0.1, -0.2, -0.3],
            sampled_token_ranks=[1, 1, 1],
            segment_ids=[0, 2, 1],
            segment_names=["thinking", "structure", "answer"],
            cumulative_logprobs=[-0.1, -0.3, -0.6],
            top20_token_ids=[[1] + [-1] * 19, [2] + [-1] * 19, [3] + [-1] * 19],
            top20_token_texts=[["a"] + [""] * 19, ["b"] + [""] * 19, ["c"] + [""] * 19],
            top20_token_logprobs=[
                [-0.1] + [float("-inf")] * 19,
                [-0.2] + [float("-inf")] * 19,
                [-0.3] + [float("-inf")] * 19,
            ],
        )
        assert record["predicted_answer"] == letter, f"Expected {letter}, got {record['predicted_answer']}"
        assert record["is_correct"] is True, f"Expected is_correct=True for gold={letter}"
        assert record["parse_success"] is True
