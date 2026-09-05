import io
import json

import pytest

from bench.gb10_profile import interval_union, trace_summary
from bench.local_workloads import DEFAULT_MANIFEST, cache_state_error, materialize, quality, stream_request


def test_cache_expectations_include_repeats_too_short_for_an_interior_snapshot():
    assert cache_state_error({'input_ids': [1] * 46}, 'repeat', 0, 64) is None
    assert cache_state_error({'input_ids': [1] * 64}, 'repeat', 0, 64) is None
    assert cache_state_error({'input_ids': [1] * 67}, 'repeat', 64, 64) is None
    assert cache_state_error({'input_ids': [1] * 67}, 'repeat', 0, 64) is not None
    assert cache_state_error({'input_ids': [1] * 67}, 'cold', 64, 64) is not None


def test_quality_checks_behavior_and_rejects_executable_side_effects():
    case = {'enable_thinking': False, 'quality': {'kind': 'clamp'}}
    assert quality(case, 'def clamp(x, lower, upper):\n return lower if x < lower else upper if x > upper else x')['passed']
    assert not quality(case, 'def clamp(x, lower, upper):\n return x')['passed']
    assert not quality(case, 'import os\nos.remove("anything")')['passed']
    assert not quality(case, 'def clamp(x, lower, upper):\n while True: pass')['passed']


def test_quality_separates_reasoning_from_answer_and_reports_manual_review():
    case = {'enable_thinking': True, 'quality': {'kind': 'exact', 'expected': '23'}}
    assert quality(case, '3 * 7 + 2 = 23. </think>23')['passed']
    assert not quality(case, '23 is my thought, but I have not finished.')['passed']
    case['quality'] = {'kind': 'manual', 'rubric': 'Read it.'}
    assert quality(case, '</think>Hello')['passed'] is None
    assert quality(case, '</think>')['passed'] is False


def test_workload_materialization_is_repeatable_and_checks_fixture_integrity(snapshot, tmp_path):
    a = materialize(DEFAULT_MANIFEST, snapshot)
    assert a == materialize(DEFAULT_MANIFEST, snapshot)
    assert len({c['id'] for c in a['cases']}) == len(a['cases'])
    session = next(c for c in a['cases'] if c['id'] == 'session')
    assert session['input_ids'][:len(session['prime_ids'])] == session['prime_ids']
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    manifest['fixtures']['request_state']['path'] = 'bad.txt'
    (tmp_path / 'bad.txt').write_text('changed')
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'):
        materialize(path, snapshot)
    manifest['fixtures']['request_state']['path'] = '../outside.txt'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes'):
        materialize(path, snapshot)


def test_stream_keeps_speculative_bursts_and_requires_terminal_event(monkeypatch):
    from bench import local_workloads
    def data(final):
        events = [{'output_ids': [1], 'meta_info': {}},
                  {'output_ids': [1, 2, 3, 4], 'meta_info': {'finish_reason': {'type': 'length'}} if final else {}}]
        return io.BytesIO((''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n\n').encode())
    monkeypatch.setattr(local_workloads.urllib.request, 'urlopen', lambda *a, **kw: data(True))
    case = {'id': 'case', 'input_ids': [42], 'sampling': {'max_new_tokens': 4}}
    first = []
    result = stream_request('http://example.invalid', case, 'id', on_first_token=lambda: first.append(True))
    assert first == [True]
    assert result['error'] is None
    assert [e['tokens'] for e in result['events']] == [1, 4]
    assert len(result['delivery_gaps_s']) == 1
    monkeypatch.setattr(local_workloads.urllib.request, 'urlopen', lambda *a, **kw: data(False))
    assert 'terminal' in stream_request('http://example.invalid', case, 'id')['error']


def test_trace_activity_uses_union_instead_of_double_counting_streams():
    assert interval_union([(0, 10), (5, 15), (20, 25)]) == 20
    events = [{'name': 'sharar.decode_step', 'ph': 'X', 'ts': 0, 'dur': 100}]
    events += [{'cat': 'kernel', 'name': 'kernel', 'ph': 'X', 'ts': start, 'dur': 60} for start in (0, 40)]
    summary = trace_summary({'traceEvents': events})
    assert summary['device_idle_fraction'] == 0
    assert summary['physical_dram_bytes'] is None
    with pytest.raises(ValueError, match='scopes'):
        trace_summary({'traceEvents': []})
