import math

from mmlu_trace_eval.schema import (
    build_example_record,
    build_token_step_records,
    normalize_topk,
    segment_token_surfaces,
    validate_example_record,
)


class DummyTokenizer:
    def convert_ids_to_tokens(self, token_id):
        return f"tok_{token_id}"


class DummyLogprob:
    def __init__(self, token_id, logprob, rank):
        self.token_id = token_id
        self.logprob = logprob
        self.rank = rank


def test_normalize_topk_pads_and_preserves_sampled_token():
    tokenizer = DummyTokenizer()
    topk = normalize_topk(
        tokenizer,
        {
            7: {"logprob": -0.1, "rank": 1},
            4: {"logprob": -1.2, "rank": 2},
        },
        sampled_token_id=7,
        top_k=4,
    )
    assert topk.token_ids == [7, 4, -1, -1]
    assert topk.token_texts == ["tok_7", "tok_4", "", ""]
    assert topk.sampled_rank == 1
    assert math.isinf(topk.logprobs[2]) and topk.logprobs[2] < 0


def test_normalize_topk_supports_object_iterables():
    tokenizer = DummyTokenizer()
    topk = normalize_topk(
        tokenizer,
        [DummyLogprob(11, -0.05, 1), DummyLogprob(5, -0.5, 2)],
        sampled_token_id=11,
        top_k=3,
    )
    assert topk.token_ids == [11, 5, -1]
    assert topk.sampled_logprob == -0.05


def test_segment_token_surfaces_labels_regions():
    completion = "<thinking>abc</thinking><answer>D</answer>"
    token_surfaces = [
        "<thinking>",
        "ab",
        "c",
        "</thinking>",
        "<answer>",
        "D",
        "</answer>",
    ]
    segment_ids, segment_names = segment_token_surfaces(completion, token_surfaces)
    assert segment_names == [
        "structure",
        "thinking",
        "thinking",
        "structure",
        "structure",
        "answer",
        "structure",
    ]
    assert segment_ids[5] == 1


def test_build_example_record_and_token_steps_are_consistent():
    example = {
        "example_id": "mmlu__test__00000__math",
        "question_idx": 0,
        "subject": "math",
        "question": "2+2?",
        "choices": ["1", "2", "3", "4"],
        "gold_answer": "D",
    }
    record = build_example_record(
        run_id="run-1",
        split="test",
        prompt_variant="reasoning_v1",
        decode_config_json='{"seed":0}',
        model_name="google/gemma-3-12b-it",
        model_revision="rev-model",
        tokenizer_revision="rev-model",
        dataset_name="cais/mmlu",
        dataset_revision="rev-dataset",
        prompt_text="prompt",
        example=example,
        completion_text="<thinking>ok</thinking><answer>D</answer>",
        sampled_token_ids=[1, 2, 3],
        sampled_token_texts=["a", "b", "c"],
        sampled_token_logprobs=[-0.1, -0.2, -0.3],
        sampled_token_ranks=[1, 1, 1],
        segment_ids=[0, 2, 1],
        segment_names=["thinking", "structure", "answer"],
        cumulative_logprobs=[-0.1, -0.3, -0.6],
        top20_token_ids=[[1] + [-1] * 19, [2] + [-1] * 19, [3] + [-1] * 19],
        top20_token_texts=[["a"] + [""] * 19, ["b"] + [""] * 19, ["c"] + [""] * 19],
        top20_token_logprobs=[[-0.1] + [float("-inf")] * 19, [-0.2] + [float("-inf")] * 19, [-0.3] + [float("-inf")] * 19],
    )
    validate_example_record(record)
    assert record["predicted_answer"] == "D"
    assert record["is_correct"] is True
    assert record["answer_start_idx"] == 2
    assert record["answer_end_idx"] == 3

    token_rows = build_token_step_records(record)
    assert len(token_rows) == 3
    assert token_rows[2]["segment_name"] == "answer"
