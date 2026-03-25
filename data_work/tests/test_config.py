from mmlu_trace_eval.config import MAX_NUM_SEQS, SMOKE_MAX_NUM_SEQS, runtime_max_num_seqs


def test_runtime_max_num_seqs_uses_smoke_profile_for_validation():
    assert runtime_max_num_seqs("validation") == SMOKE_MAX_NUM_SEQS
    assert runtime_max_num_seqs("dev") == SMOKE_MAX_NUM_SEQS


def test_runtime_max_num_seqs_keeps_full_profile_for_test():
    assert runtime_max_num_seqs("test") == MAX_NUM_SEQS
