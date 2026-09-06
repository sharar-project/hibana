from copy import deepcopy

from bench.agent_workloads import continuation, validate
from bench.agent_coverage import select_branch, score_branch


def sample():
    return {'error': None, 'meta': {'prompt_tokens': 2, 'completion_tokens': 5, 'spec_verify_ct': 2},
            'output_ids': [10, 11, 12, 13, 14], 'events': [{'tokens': 2, 't_s': 1}, {'tokens': 5, 't_s': 2}]}


def test_decode_rate_excludes_entire_first_stream_burst():
    row = sample()
    validate(row, [1, 2], 5, True)
    assert row['decode_tok_s'] == 3
    assert row['post_prefill_tokens_per_round'] == 2
    assert row['error'] is None


def test_truncated_fixed_stream_is_retained_as_failure():
    row = sample()
    validate(row, [1, 2], 6, True)
    assert 'fixed output budget' in row['error']
    assert len(row['output_ids']) == 5


def test_natural_completion_can_finish_before_budget():
    row = sample()
    validate(row, [1, 2], 256, False)
    assert row['error'] is None


def test_wrong_usage_and_transport_error_are_both_preserved():
    row = sample()
    row['error'] = 'lost stream'
    row['meta']['completion_tokens'] = 6
    validate(row, [1, 2], 6, True)
    assert 'lost stream' in row['error'] and 'reported output count' in row['error']


def test_single_burst_does_not_invent_decode_time():
    row = sample()
    row['events'] = [{'tokens': 5, 't_s': 1}]
    validate(row, [1, 2], 5, True)
    assert row['decode_tok_s'] is None


def test_continuation_preserves_actual_answer_and_branch_source():
    class Tokenizer:
        def encode(self, text):
            assert text.count('<|im_end|>') == 2
            return [90, 91]
    initial, answer = [1, 2], [3, 4, 248046]
    before = deepcopy((initial, answer))
    appended = continuation(Tokenizer(), initial, answer, 'Tool result')
    branch = continuation(Tokenizer(), initial, answer, 'Alternate result')
    assert appended == branch == [1, 2, 3, 4, 90, 91]
    assert (initial, answer) == before


def test_branch_screen_counts_recovered_correction_and_lost_full_chain():
    branch = {'depth': 1, 'token': 42}
    assert score_branch(branch, [7, 42])['potential_token_delta'] == 1
    assert score_branch(branch, [7, 43])['potential_token_delta'] == 0
    assert score_branch(branch, list(range(8)))['potential_token_delta'] == -1


def test_branch_selection_uses_draft_path_mass_and_stable_ties():
    candidates = [[10 + i * 2, 11 + i * 2] for i in range(7)]
    scores = [[[0., 0.], [0., 0.]] for _ in range(7)]
    # Equal first-step choices: choose rank zero for the main chain and rank
    # one for the alternative. Later alternatives have smaller path mass.
    assert select_branch(candidates, scores) == {'depth': 0, 'token': 11, 'draft_path_mass': .5}
