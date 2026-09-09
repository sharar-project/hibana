import copy
import json
from pathlib import Path
import subprocess

import pytest

from sharar import benchmark
from sharar.benchmark_aiperf import digest, timings
from sharar.benchmark_workloads import materialize
from sharar.cli import main


def sample(latency, ttft, *, warmup=False, output_hash='same', ids=None):
    ids = ids if ids is not None else [1] * 10000
    return {'case': 'code', 'warmup': warmup, 'latency_s': latency, 'ttft_s': ttft,
            'output_tps': 1000 / latency, 'decode_tps': 999 / (latency - ttft),
            'input_ids_sha256': digest(ids), 'output_ids_sha256': output_hash,
            'usage': {'prompt_tokens': len(ids), 'completion_tokens': 1000}, 'verify_rounds': 200}


def test_tps_uses_total_tokens_over_total_time_and_excludes_warmups():
    rows = [sample(100, 10, warmup=True, output_hash='warmup'), sample(2, .5), sample(4, 1)]
    result = benchmark.summarize(rows)['code']
    assert result['requests'] == 2
    assert result['latency_s'] == 3
    assert result['ttft_s'] == .75
    assert result['output_tps'] == pytest.approx(2000 / 6)
    assert result['decode_tps'] == pytest.approx(1998 / 4.5)
    assert result['emitted_tokens_per_round'] == pytest.approx(1998 / 400)
    assert result['repeatable_output']
    rows[-1]['output_ids_sha256'] = 'different'
    assert not benchmark.summarize(rows)['code']['repeatable_output']


@pytest.mark.parametrize('failure', ['count', 'unit', 'nan', 'ordering'])
def test_invalid_aiperf_metrics_cannot_produce_tps(failure):
    metrics = {'input_sequence_length': {'value': 10, 'unit': 'tokens'},
               'output_sequence_length': {'value': 3, 'unit': 'tokens'},
               'request_latency': {'value': 2000, 'unit': 'ms'},
               'time_to_first_token': {'value': 500, 'unit': 'ms'}}
    if failure == 'count':
        metrics['output_sequence_length']['value'] = 2
    elif failure == 'unit':
        metrics['request_latency']['unit'] = 's'
    elif failure == 'nan':
        metrics['request_latency']['value'] = float('nan')
    else:
        metrics['time_to_first_token']['value'] = 2000
    with pytest.raises(ValueError):
        timings(metrics, 10, 3)


def test_baseline_requires_identical_inputs_and_reports_text_drift():
    baseline = {'schema': 1, 'completed': True, 'protocol': {'model': 'pinned'},
                'cases': benchmark.summarize([sample(4, 1)])}
    current = {'protocol': baseline['protocol'], 'cases': benchmark.summarize([sample(2, .5)])}
    result = benchmark.compare(current, baseline)['code']
    assert result['output_tps_change_pct'] == 100
    assert result['same_output_ids']
    current['cases']['code']['output_ids_sha256'] = ['changed']
    assert not benchmark.compare(current, baseline)['code']['same_output_ids']
    current['cases']['code']['input_ids_sha256'] = 'wrong prompt'
    with pytest.raises(ValueError, match='inputs/token budgets differ'):
        benchmark.compare(current, baseline)


@pytest.mark.parametrize('case,expected', [
    ('code', '04d6e0023e3b6b63d412de5a8bf49f6ee7efe68ef7061da013ec126426b4e195'),
    ('design', '0402378ae156b5906548dcd7bbacf6484421643b6386284e8b8375727df975b1'),
    ('reasoning', 'fced3b9d8b1cafb6d661e1f604d171681751e555530b97b0ad6b4ef66f79ed40'),
])
def test_builtin_workloads_match_frozen_aiperf_baseline(snapshot, case, expected):
    from sharar.tokenizer import SnapshotTokenizer

    ids = materialize(SnapshotTokenizer(snapshot).encode, case, 10000)
    assert len(ids) == 10000
    assert digest(ids) == expected


def test_packaged_source_fixture_stays_frozen():
    fixture = Path(__file__).resolve().parents[2] / 'bench/workloads/request_state_7aebbe6.txt'
    packaged = Path(benchmark.__file__).parent / 'benchmark_data' / fixture.name
    assert packaged.read_bytes() == fixture.read_bytes()


def test_cli_end_to_end_writes_summary_and_reuses_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(benchmark, 'load_config', lambda: None)
    monkeypatch.setattr(benchmark, 'resolve_aiperf', lambda _: Path('/unused/aiperf'))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '0.12.0\n'))
    monkeypatch.setattr(benchmark, 'http_json', lambda *a, **k: {'engine': 'test'})
    monkeypatch.setattr(benchmark, 'materialize', lambda encode, case, length: [1] * length)
    calls = []

    def fake_trial(url, aiperf, ids, output, warmup, case, repetition, **kwargs):
        calls.append((url, list(ids), warmup))
        return sample(100 if warmup else 2 * (repetition + 1), .5, warmup=warmup, ids=ids)

    monkeypatch.setattr(benchmark, 'trial', fake_trial)
    output = tmp_path / 'first'
    assert main(['benchmark', '--repeats', '2', '--output', str(output)]) == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['completed'] and summary['cases']['code']['output_tps'] == pytest.approx(2000 / 6)
    assert [call[2] for call in calls] == [True, False, False]
    assert 'TPS (end-to-end)' in capsys.readouterr().out
    monkeypatch.setattr(benchmark, 'materialize', lambda *a: pytest.fail('must reuse frozen IDs'))
    assert main(['benchmark', '--url', 'http://sglang:30000/v1', '--inputs', str(output / 'frozen-inputs.json'),
                 '--baseline', str(output), '--repeats', '2', '--output', str(tmp_path / 'second')]) == 0
    assert calls[-1][0] == 'http://sglang:30000'
    assert 'end-to-end TPS +0.00%' in capsys.readouterr().out
    count = len(calls)
    assert main(['benchmark', '--inputs', str(output / 'frozen-inputs.json'), '--output', str(output)]) == 1
    assert len(calls) == count  # an existing run is never overwritten


@pytest.mark.parametrize('options', [['--repeats', '0'], ['--warmups', '-1'], ['--output-tokens', '1'],
                                    ['--input-tokens', '32768'], ['--url', 'file:///tmp/server']])
def test_cli_invalid_options_fail_before_launch(options, monkeypatch):
    monkeypatch.setattr(benchmark, 'resolve_aiperf', lambda _: pytest.fail('must reject before launching AIPerf'))
    assert main(['benchmark', *options]) == 1


def test_config_reads_paths_without_executing_shell(tmp_path, monkeypatch):
    config = tmp_path / 'gb10.env'
    marker = tmp_path / 'must-not-exist'
    config.write_text(f'SHARAR_RESULTS="{tmp_path}/with spaces"\nSHARAR_AIPERF="$(touch {marker})"\n')
    monkeypatch.setenv('SHARAR_CONFIG_FILE', str(config))
    monkeypatch.delenv('SHARAR_RESULTS', raising=False)
    monkeypatch.delenv('SHARAR_AIPERF', raising=False)
    benchmark.load_config()
    assert benchmark.results_root() == tmp_path / 'with spaces'
    assert not marker.exists()
    monkeypatch.delenv('SHARAR_AIPERF')
