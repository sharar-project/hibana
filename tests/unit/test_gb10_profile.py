import pytest

from bench.gb10_profile import interval_union, trace_summary


def test_trace_activity_uses_union_instead_of_double_counting_streams():
    assert interval_union([(0, 10), (5, 15), (20, 25)]) == 20
    events = [{'name': 'sharar.decode_step', 'ph': 'X', 'ts': 0, 'dur': 100}]
    events += [{'cat': 'kernel', 'name': 'kernel', 'ph': 'X', 'ts': start, 'dur': 60} for start in (0, 40)]
    summary = trace_summary({'traceEvents': events})
    assert summary['device_idle_fraction'] == 0
    assert summary['physical_dram_bytes'] is None
    with pytest.raises(ValueError, match='scopes'):
        trace_summary({'traceEvents': []})
