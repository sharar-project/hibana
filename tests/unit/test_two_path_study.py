import pytest

from bench.two_path_study import coverage, path, summarize_coverage


def test_alternative_follows_its_own_conditional_history():
    candidates = [[10, 11], [20, 21], [30, 31]]
    scores = [[[2, 1], [2, 1]], [[2, 1], [1, 2]], [[2, 1], [1, 2]]]
    assert path(candidates, scores) == [10, 20, 30]
    assert path(candidates, scores, {'depth': 0, 'token': 11}) == [11, 21, 31]


def test_coverage_includes_correction_and_requires_bonus_lookahead():
    assert coverage([1, 2, 3], [1, 8, 9, 10]) == 2
    assert coverage([1, 2, 3], [1, 2, 3, 4]) == 4
    with pytest.raises(ValueError, match='lookahead'):
        coverage([1, 2, 3], [1, 2, 3])


def test_full_trajectory_screen_keeps_late_rounds_and_excludes_truncated_terminal():
    output = [0] * 512 + [10, 11, 12, 13, 14, 15, 16, 17, 18]
    captured = {'output_ids': output, 'rows': [
        {'offset': 512, 'emitted': [10], 'base': [99] * 7, 'alternate': list(range(10, 17))},
        {'offset': 513, 'emitted': list(range(11, 19)), 'base': list(range(11, 18)), 'alternate': [99] * 7},
        {'offset': 514, 'emitted': list(range(12, 19)), 'base': list(range(12, 19)), 'alternate': [99] * 7},
    ]}
    result = summarize_coverage(captured)
    assert result['included_rounds'] == 2 and result['excluded_terminal'] == 1
    assert result['native_tokens'] == 9 and result['candidate_tokens'] == 16
    assert result['after_512'] == {'native': 9, 'candidate': 16}
    captured['rows'][0]['emitted'] = [10, 11]
    with pytest.raises(AssertionError):
        summarize_coverage(captured)
