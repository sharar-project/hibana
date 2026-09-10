import math

import pytest

from bench.gb10_family_eval import (compare_reports, first_difference, first_eos, format_comparison,
                                    paired_stats, pre_eos_rounds, select_tasks, task_metrics)


def _result(ids, steps, decode_s, rounds=None, ttft=4.0):
    return {'output_ids': ids, 'emitted_per_step': steps, 'spec_rounds': len(steps) if rounds is None else rounds,
            'decode_s': decode_s, 'total_s': ttft + decode_s, 'ttft_s': ttft}


def test_first_eos_and_pre_eos_rounds():
    ids = [5, 6, 7, 248046, 8, 9]
    assert first_eos(ids) == 3
    assert first_eos([1, 2, 3]) is None
    # prefill gives index 0; step 1 emits indices 1,2; step 2 emits index 3
    assert pre_eos_rounds([2, 1, 3], 3) == 2
    assert pre_eos_rounds([2, 1, 3], None) is None
    assert pre_eos_rounds([1, 1], 5) == 2  # never reached: all steps


def test_task_metrics_arithmetic():
    ids = list(range(1, 11))  # 10 outputs
    m = task_metrics(_result(ids, [3, 3, 3], decode_s=0.3))
    assert m['decode_tps'] == pytest.approx(9 / 0.3)
    assert m['tok_per_round'] == pytest.approx(3.0)
    assert m['round_ms'] == pytest.approx(100.0)
    assert m['first_eos'] is None and m['eos_count'] == 0 and m['plain_steps'] == 0
    m2 = task_metrics(_result([1, 248046, 2, 3], [1, 2], decode_s=0.2, rounds=1))
    assert m2['plain_steps'] == 1 and m2['first_eos'] == 1 and m2['pre_eos_rounds'] == 1
    assert m2['pre_eos_tok_per_round'] == pytest.approx(1.0)


def test_paired_stats_ratio_and_ci():
    rows = [{'baseline': {'x': 10.0}, 'candidate': {'x': 11.0}} for _ in range(6)]
    s = paired_stats(rows, 'x')
    assert s['n'] == 6 and s['ratio'] == pytest.approx(1.1)
    assert s['ci95'][0] == pytest.approx(1.1) and s['ci95'][1] == pytest.approx(1.1)
    assert paired_stats([], 'x') == {'n': 0}
    mixed = [{'baseline': {'x': 10.0}, 'candidate': {'x': v}} for v in (9.0, 11.0, 10.0, 12.0, 8.0, 10.5)]
    s2 = paired_stats(mixed, 'x')
    assert s2['ci95'][0] < s2['ratio'] < s2['ci95'][1]
    assert s2['min'] == pytest.approx(0.8) and s2['max'] == pytest.approx(1.2)


def _report(label, tasks):
    rows = []
    for tid, split, ids, steps, dec in tasks:
        raw = _result(ids, steps, dec)
        rows.append({'id': tid, 'split': split, 'category': 'code', 'repeat': 0, 'metrics': task_metrics(raw), **raw})
    return {'label': label, 'completed': True, 'protocol': {'inputs_sha256': 'x', 'output_tokens': 4}, 'tasks': rows}


def test_compare_reports_groups_and_outputs():
    base = _report('native', [('a', 'train', [1, 2, 3, 4], [1, 2], 0.2), ('b', 'validation', [1, 2, 3, 4], [3], 0.1),
                              ('benchmark_code', 'benchmark', [1, 2, 3, 4], [1, 1, 1], 0.3)])
    cand = _report('cand', [('a', 'train', [1, 2, 3, 4], [1, 2], 0.1), ('b', 'validation', [1, 2, 9, 9], [3], 0.1),
                            ('benchmark_code', 'benchmark', [1, 2, 3, 4], [3], 0.1)])
    cmp_ = compare_reports(base, [cand], baseline_repeat=base)
    fam = cmp_['summary']['families']
    assert fam['count'] == 2 and fam['same_output_count'] == 1
    assert fam['decode_tps']['ratio'] == pytest.approx(math.sqrt(2.0))  # a: 2x, b: 1x
    assert cmp_['summary']['benchmarks']['tok_per_round']['ratio'] == pytest.approx(3.0)
    rows = {r['id']: r for r in cmp_['tasks']}
    assert rows['b']['first_difference'] == 2 and rows['a']['first_difference'] is None
    assert cmp_['noise_floor']['decode_tps']['ratio'] == pytest.approx(1.0)
    text = format_comparison(cmp_)
    assert 'families' in text and 'benchmark_code' in text and 'noise' in text


def test_first_difference_and_select_tasks():
    assert first_difference([1, 2], [1, 2]) is None
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1, 2], [1, 2, 3]) == 2
    frozen = {'tasks': [{'id': 'a', 'split': 'train'}, {'id': 'b', 'split': 'benchmark'}]}
    assert [t['id'] for t in select_tasks(frozen, 'all')] == ['a', 'b']
    assert [t['id'] for t in select_tasks(frozen, 'benchmark')] == ['b']
    assert [t['id'] for t in select_tasks(frozen, 'a')] == ['a']
    with pytest.raises(ValueError):
        select_tasks(frozen, 'zzz')
