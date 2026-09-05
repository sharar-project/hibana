import copy

import pytest

from bench.compare_contracts import compare


def report():
    sample = {'output_sha256': 'stable', 'decode_tps': 12.0, 'emitted_tps': 10.0, 'available_gib': 80.0}
    return {'profile': {'repetitions': 3}, 'rows': [
        {'mode': mode, 'input_tokens': length, 'batch': batch,
         'samples': [dict(sample) for _ in range(3)]}
        for mode in ('plain', 'mtp') for length in (128, 2048, 8192) for batch in (1, 4)]}


@pytest.mark.parametrize('fault', ['empty', 'duplicate', 'samples', 'nan', 'floor'])
def test_comparison_rejects_invalid_evidence(fault):
    baseline, candidate = report(), report()
    if fault == 'empty':
        candidate['rows'] = []
    elif fault == 'duplicate':
        candidate['rows'].append(copy.deepcopy(candidate['rows'][0]))
    elif fault == 'samples':
        candidate['rows'][0]['samples'].pop()
    else:
        candidate['rows'][0]['samples'][0].update(
            {'decode_tps': float('nan')} if fault == 'nan' else {'available_gib': 15.9})
    with pytest.raises(ValueError):
        compare(baseline, candidate)


def test_comparison_rejects_output_drift_and_regression():
    baseline, candidate = report(), report()
    assert compare(baseline, candidate)['pass']
    candidate['rows'][0]['samples'][1]['output_sha256'] = 'different'
    assert not compare(baseline, candidate)['pass']
    candidate = report()
    for sample in candidate['rows'][0]['samples']:
        sample['emitted_tps'] *= 0.94
    assert not compare(baseline, candidate)['pass']
