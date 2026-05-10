from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedRequest:
    example: dict[str, Any]
    prompt_text: str
    prompt_tokens: int


def batch_requests(
    requests: list[PreparedRequest],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_output_tokens: int,
) -> list[list[PreparedRequest]]:
    batches: list[list[PreparedRequest]] = []
    current: list[PreparedRequest] = []
    current_budget = 0

    for request in requests:
        request_budget = request.prompt_tokens + max_output_tokens
        if request_budget > max_num_batched_tokens:
            if current:
                batches.append(current)
                current = []
                current_budget = 0
            batches.append([request])
            continue

        would_exceed_count = len(current) >= max_num_seqs
        would_exceed_budget = current_budget + request_budget > max_num_batched_tokens

        if current and (would_exceed_count or would_exceed_budget):
            batches.append(current)
            current = []
            current_budget = 0

        current.append(request)
        current_budget += request_budget

    if current:
        batches.append(current)

    return batches
