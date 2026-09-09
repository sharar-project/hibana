import copy

import pytest

from bench.gb10_acceptance_replay import prefix_matches, replay_candidates, validate_reference


def reference():
    return {'input_ids': [1, 2, 3], 'output_ids': [10, 11, 12, 13, 14], 'verify_width': 16,
            'rounds': [{'position': 3, 'emitted': [11, 12], 'target_top1': [11, 12]+[0]*14},
                       {'position': 5, 'emitted': [13, 14], 'target_top1': [13, 14]+[0]*14}]}


def test_replay_validates_boundaries_and_reconstructs_the_continuation():
    assert validate_reference(reference()) == [10, 11, 12, 13, 14]
    assert prefix_matches([11, 12, 99, 14], [11, 12, 13, 14]) == 2
    assert prefix_matches([11, 12], [11, 12, 13]) == 2


@pytest.mark.parametrize('token', [0, 1, 248070])
def test_replay_enforces_every_reference_acceptance_length_even_on_padding_tokens(token):
    target = [token] * 16
    for count in range(1, 17):
        candidates = replay_candidates(target, count)
        assert len(candidates) == 15
        assert prefix_matches(candidates, target) + 1 == count


@pytest.mark.parametrize('fault', ['position', 'emitted', 'output_ids', 'width'])
def test_replay_rejects_mismatched_reference_data(fault):
    data = copy.deepcopy(reference())
    if fault == 'position':
        data['rounds'][1]['position'] += 1
    elif fault == 'emitted':
        data['rounds'][0]['emitted'][0] = 999
    elif fault == 'output_ids':
        data['output_ids'][2] = 999
    else:
        data['verify_width'] = 8
    with pytest.raises(ValueError):
        validate_reference(data)
