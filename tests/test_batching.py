from mmlu_trace_eval.batching import PreparedRequest, batch_requests


def _request(index: int, prompt_tokens: int) -> PreparedRequest:
    return PreparedRequest(
        example={"example_id": f"e{index}"},
        prompt_text=f"prompt-{index}",
        prompt_tokens=prompt_tokens,
    )


def test_batch_requests_respects_count_and_budget():
    requests = [_request(0, 10), _request(1, 10), _request(2, 50)]
    batches = batch_requests(
        requests,
        max_num_seqs=2,
        max_num_batched_tokens=100,
        max_output_tokens=20,
    )
    assert [len(batch) for batch in batches] == [2, 1]


def test_batch_requests_singleton_for_oversized_prompt():
    requests = [_request(0, 200), _request(1, 10)]
    batches = batch_requests(
        requests,
        max_num_seqs=4,
        max_num_batched_tokens=100,
        max_output_tokens=20,
    )
    assert [len(batch) for batch in batches] == [1, 1]
    assert batches[0][0].example["example_id"] == "e0"
