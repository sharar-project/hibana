"""Fail closed when a benchmark request's raw token evidence is inconsistent."""
import copy
import json

import pytest

from bench.gb10_aiperf import audit_raw, payload_for


def record():
    payload = payload_for([10, 11], budget=3)
    chunks = [
        {'choices': [{'text': 'a', 'token_ids': [5], 'prompt_token_ids': [10, 11]}]},
        {'choices': [{'text': 'bc', 'token_ids': [6, 7], 'finish_reason': 'length'}]},
        {'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}},
    ]
    raw = {'status': 200, 'payload': copy.deepcopy(payload), 'responses': [
        {'packets': [{'name': 'data', 'value': json.dumps(chunk)}]} for chunk in chunks]}
    raw['responses'].append({'packets': [{'name': 'data', 'value': '[DONE]'}]})
    return payload, raw


def change_chunk(raw, index, fn):
    packet = raw['responses'][index]['packets'][0]
    chunk = json.loads(packet['value'])
    fn(chunk)
    packet['value'] = json.dumps(chunk)


def test_accepts_exact_raw_ids_and_server_counts():
    payload, raw = record()
    result = audit_raw(raw, payload)
    assert result['output_ids'] == [5, 6, 7]
    assert result['text'] == 'abc'


@pytest.mark.parametrize('failure', ['payload', 'prompt_ids', 'missing_output_ids', 'usage', 'cache', 'stop', 'stream_error', 'no_done'])
def test_rejects_invalid_evidence(failure):
    payload, raw = record()
    if failure == 'payload':
        raw['payload']['temperature'] = 1
    elif failure == 'prompt_ids':
        change_chunk(raw, 0, lambda c: c['choices'][0].update(prompt_token_ids=[10, 12]))
    elif failure == 'missing_output_ids':
        change_chunk(raw, 1, lambda c: c['choices'][0].pop('token_ids'))
    elif failure == 'usage':
        change_chunk(raw, 2, lambda c: c['usage'].update(completion_tokens=2))
    elif failure == 'cache':
        change_chunk(raw, 2, lambda c: c['usage'].update(prompt_tokens_details={'cached_tokens': 1}))
    elif failure == 'stop':
        change_chunk(raw, 1, lambda c: c['choices'][0].update(finish_reason='stop'))
    elif failure == 'stream_error':
        change_chunk(raw, 1, lambda c: c.update(error='failed'))
    elif failure == 'no_done':
        raw['responses'].pop()
    with pytest.raises(ValueError):
        audit_raw(raw, payload)
