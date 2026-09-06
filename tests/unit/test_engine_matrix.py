"""Boundary checks for comparable-work eligibility, independent of engine code."""
from bench.engine_matrix import validate


def row(count=4, cached=0):
    return {'meta': {'prompt_tokens': 2, 'completion_tokens': count, 'cached_tokens': cached},
            'events': [{'t_s': 1., 'tokens': 2}, {'t_s': 2., 'tokens': count}],
            'output_ids': list(range(count)), 'error': None}


def test_retry_miss_is_a_measured_result():
    value = row()
    validate(value, [1, 2], 4, 'retry')
    assert value['valid'] and not value['cache_hit']
    assert value['decode_tok_s'] == 2  # exclude the whole two-token first burst


def test_cold_hit_is_not_cold_work():
    value = row(cached=1)
    validate(value, [1, 2], 4, 'cold')
    assert not value['valid'] and not value['cache_valid']


def test_natural_short_answer_is_retained_but_not_equal_work():
    value = row(count=3)
    validate(value, [1, 2], 4, 'cold')
    assert not value['valid'] and not value['budget_fulfilled']
    assert value['error'] is None and value['output_tokens'] == 3


def test_unattested_stream_counts_cannot_produce_decode_rate():
    value = row()
    value['output_ids'] = None
    value['text'] = 'answer'
    value['count_error'] = 'missing token events'
    validate(value, [1, 2], 4, 'cold')
    assert not value['valid'] and value['decode_tok_s'] is None


def test_atlas_stream_counts_empty_decoded_tokens(monkeypatch):
    import io
    import json
    from bench.engine_matrix import atlas_stream

    events = [
        {'choices': [{'text': '', 'finish_reason': None}]},
        {'choices': [{'text': 'answer', 'finish_reason': None}]},
        {'choices': [{'text': '', 'finish_reason': 'length'}]},
        {'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 2,
                                 'prompt_tokens_details': {'cached_tokens': 0}}},
    ]
    stream = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
    stream += b'data: [DONE]\n\n'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(stream))
    value = atlas_stream('http://unused', [1, 2], 2, 'request')
    validate(value, [1, 2], 2, 'cold')
    assert value['valid'] and value['text'] == 'answer'
    assert [event['tokens'] for event in value['events']] == [1, 2]
    assert value['decode_tok_s'] > 0


def test_atlas_stream_does_not_invent_hidden_eos_timestamp(monkeypatch):
    import io
    from bench.engine_matrix import atlas_stream

    stream = (
        b'data: {"choices":[{"text":"answer","finish_reason":null}]}\n\n'
        b'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,'
        b'"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
        b'data: [DONE]\n\n'
    )
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: io.BytesIO(stream))
    value = atlas_stream('http://unused', [1, 2], 8, 'request')
    validate(value, [1, 2], 8, 'cold')
    assert value['output_tokens'] == 2 and len(value['events']) == 1
    assert value['count_error'] and value['decode_tok_s'] is None
    assert not value['valid']
