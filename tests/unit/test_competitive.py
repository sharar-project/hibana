import pytest

from bench import competitive
from bench.competitive import sampling_params, summarize


@pytest.mark.parametrize('reply', ['Cache flushed.\nPlease check backend logs.',
                                 'Cache invalidated; active readers release pinned entries on completion.\n'])
def test_native_cache_flush_success_messages(monkeypatch, reply):
    monkeypatch.setattr(competitive.workloads, 'json_request', lambda *args: reply)
    competitive.flush('http://test')


def test_seed_uses_each_native_api_name_without_changing_the_workload():
    case = {'sampling': {'seed': 123, 'temperature': .7}}
    assert sampling_params(case, 'sharar-dflash2') == case['sampling']
    assert sampling_params(case, 'sglang-dflash2') == {'sampling_seed': 123, 'temperature': .7}
    assert case['sampling']['seed'] == 123


def sample(**changes):
    request = dict(output_ids=[42], meta={'cached_tokens': 0}, error=None,
                   quality={'passed': None}, completion_s=.2, ttft_s=.2, decode_tok_s=None)
    return dict(cell='chat', warmup=False, valid=True, requests=[request], aggregate_tok_s=5., **changes)


def test_single_burst_has_no_invented_decode_rate_or_quality_pass():
    result = summarize([sample()])['chat']
    assert result['aggregate_tok_s'] == 5.
    assert result['decode_tok_s'] is None
    assert result['rubric_pending']


def test_failed_request_suppresses_timing_and_retains_error():
    good, failed = sample(), sample()
    failed['valid'] = False
    failed['requests'][0].update(error='HTTP 500', output_ids=[], ttft_s=None)
    result = summarize([good, failed])['chat']
    assert not result['valid']
    assert result['errors'] == ['HTTP 500']
    assert 'aggregate_tok_s' not in result


def test_warmup_never_changes_reported_speed():
    warmup = sample()
    warmup.update(warmup=True, aggregate_tok_s=1000.)
    assert summarize([warmup]) == {}
    result = summarize([warmup, sample()])['chat']
    assert result['samples'] == 1
    assert result['aggregate_tok_s'] == 5.
