"""Keep useful-row and HTTP completion accounting independent of output text."""
import json

import pytest

from bench.gb10_bs4_engine import work_summary
from bench.gb10_bs4_http import request


def test_work_summary_accounts_for_partial_batch_and_stop_truncation():
    events = [
        {'kind': 'prefill', 'batch': 2, 'seconds': 2.},
        {'kind': 'decode', 'batch': 4, 'seconds': .2, 'physical_rows': 64,
         'real_rows': 64, 'padded_rows': 0, 'emitted': [16, 8, 4, 1]},
        {'kind': 'decode', 'batch': 3, 'seconds': .1, 'physical_rows': 32,
         'real_rows': 24, 'padded_rows': 8, 'emitted': [2, 1, 1]},
    ]
    work = work_summary(events)
    assert work['physical_verify_rows'] == 96
    assert work['padded_rows'] == 8
    assert work['uncommitted_real_rows'] == 55
    assert work['emitted_decode_tokens'] == 33
    assert work['committed_row_fraction'] == 33 / 96
    assert work['full_batch_tps'] == 145
    assert work['full_batch_time_fraction'] == pytest.approx(2 / 3)
    assert work['prefill_host_s'] == 2.


def test_empty_work_has_no_throughput():
    work = work_summary([])
    assert work['decode_rounds'] == work['physical_verify_rows'] == 0
    assert work['full_batch_tps'] is work['committed_row_fraction'] is None


@pytest.mark.parametrize('complete,cached,ok', [(True, 0, True), (False, 0, False), (True, 64, False)])
def test_http_client_requires_complete_uncached_token_stream(monkeypatch, complete, cached, ok):
    from bench import gb10_bs4_http
    class Response:
        status = 200
        def __init__(self):
            value = {'output_ids': [101, 102], 'meta_info': {'completion_tokens': 2, 'cached_tokens': cached}}
            self.lines = iter([b': keepalive\n', b'data: ' + json.dumps(value).encode() + b'\n',
                               *([b'data: [DONE]\n'] if complete else [])])
        def readline(self):
            return next(self.lines, b'')
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(gb10_bs4_http.http.client, 'HTTPConnection', Connection)
    row = request('http://127.0.0.1:1234', {'id': 'test', 'input_ids': [11], 'output_tokens': 2}, 0.)
    assert row['ok'] is ok
    assert row['output_ids'] == [101, 102]
