"""Anthropic Messages API: request translation onto the chat path, content
blocks, stop reasons, stream events and errors."""
import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from sharar.api import messages as messages_api
from sharar.api.server import create_app
from sharar.engine.loop import EngineUnavailable
from sharar.engine.scheduler import QueueFull

TOOLS = [{'name': 'get_weather', 'description': 'Weather for a city',
          'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}}}}]
CALL = '<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>'


# One ID per character, except that tags are single tokens as in the real vocabulary.
TAGS = ['<tool_call>', '</tool_call>', '<think>', '</think>']
TAG_SPLIT = re.compile('(' + '|'.join(map(re.escape, TAGS)) + ')')


def encode(text):
    return [0x110000 + TAGS.index(piece) if piece in TAGS else ord(c)
            for piece in TAG_SPLIT.split(text) for c in ([piece] if piece in TAGS else piece)]


def decode(ids, **kwargs):
    return ''.join(TAGS[i - 0x110000] if i >= 0x110000 else chr(i) for i in ids)


def render(messages, enable_thinking=False, tools=None, **kwargs):
    return 'prompt' + ('<think>\n' if enable_thinking else '')


def fake_engine(output='', reason='stop', chunk=3, prefix_len=0):
    queue = asyncio.Queue()
    ids = encode(output)
    parts = [ids[i:i + chunk] for i in range(0, len(ids), chunk)] or [[]]
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        queue.put_nowait((part, last, reason if last else None))
    ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=prefix_len))
    return SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock(), stats=lambda: {},
                           sched=SimpleNamespace(ex=SimpleNamespace(spec=None)))


def fake_app(engine, describe=None):
    tok = SimpleNamespace(encode=Mock(side_effect=encode), decode=decode)
    tmpl = SimpleNamespace(render=Mock(side_effect=render))
    return create_app(engine, tok, tmpl, describe or {}), tmpl


def body(**fields):
    return {'model': 'claude-sonnet-4-5', 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': 'hello'}], **fields}


def post(payload, output='', reason='stop', path='/v1/messages', describe=None, **kwargs):
    async def scenario():
        engine = fake_engine(output, reason, **kwargs)
        app, tmpl = fake_app(engine, describe)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post(path, json=payload)
        return response, engine, tmpl
    return asyncio.run(scenario())


def sse_events(text):
    events = []
    for chunk in text.strip().split('\n\n'):
        name, data = chunk.split('\n')
        assert name.startswith('event: ') and data.startswith('data: ')
        event = json.loads(data[len('data: '):])
        assert event['type'] == name[len('event: '):]
        events.append(event)
    return events


def rendered_messages(tmpl):
    return tmpl.render.call_args.args[0]


# ---- request translation ----
def test_string_and_block_system_prompts():
    _, _, tmpl = post(body(system='Be brief.'))
    assert rendered_messages(tmpl)[0] == {'role': 'system', 'content': 'Be brief.'}
    blocks = [{'type': 'text', 'text': 'You are a CLI.', 'cache_control': {'type': 'ephemeral'}},
              {'type': 'text', 'text': ' Be brief.'}]
    _, _, tmpl = post(body(system=blocks))
    assert rendered_messages(tmpl)[0] == {'role': 'system', 'content': [
        {'type': 'text', 'text': 'You are a CLI.'}, {'type': 'text', 'text': ' Be brief.'}]}
    _, _, tmpl = post(body())
    assert rendered_messages(tmpl)[0]['role'] == 'user'


def test_tool_history_translates_in_order():
    turns = [
        {'role': 'user', 'content': 'weather?'},
        {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'Call the tool.', 'signature': messages_api.SIGNATURE},
            {'type': 'redacted_thinking', 'data': 'opaque'},
            {'type': 'text', 'text': 'Checking.'},
            {'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {'city': 'Paris'}},
            {'type': 'tool_use', 'id': 'toolu_2', 'name': 'get_weather', 'input': {'city': 'Rome'}}]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'sunny'},
            {'type': 'tool_result', 'tool_use_id': 'toolu_2', 'is_error': True,
             'content': [{'type': 'text', 'text': 'no '}, {'type': 'text', 'text': 'data'}]},
            {'type': 'text', 'text': 'and tomorrow?'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'tool_result', 'tool_use_id': 'x'}]},
    ]
    _, _, tmpl = post(body(messages=turns, tools=TOOLS))
    assert rendered_messages(tmpl) == [
        {'role': 'user', 'content': 'weather?'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Checking.'}], 'reasoning_content': 'Call the tool.',
         'tool_calls': [{'id': 'toolu_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}},
                        {'id': 'toolu_2', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Rome'}}}]},
        {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'sunny'},
        {'role': 'tool', 'tool_call_id': 'toolu_2', 'content': [{'type': 'text', 'text': 'no '}, {'type': 'text', 'text': 'data'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'and tomorrow?'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}]},
        {'role': 'tool', 'tool_call_id': 'x', 'content': ''},
    ]
    assert tmpl.render.call_args.kwargs['tools'] == [{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Weather for a city', 'parameters': TOOLS[0]['input_schema']}}]


def test_cache_breakpoints_and_server_tools_are_recorded():
    request = messages_api.parse_request({
        'system': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b', 'cache_control': {'type': 'ephemeral'}}],
        'tools': [{**TOOLS[0], 'cache_control': {'type': 'ephemeral'}}, {'type': 'web_search_20250305', 'name': 'web_search'}],
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'q', 'cache_control': {'type': 'ephemeral'}}]}]})
    assert request.cache_breakpoints == [('system', 1), ('messages', 0, 0), ('tools', 0)]
    assert request.ignored_tools == 1 and len(request.tools) == 1


@pytest.mark.parametrize('thinking,kwargs,expected', [
    ({'type': 'enabled', 'budget_tokens': 1024}, None, True),
    ({'type': 'adaptive'}, None, True),
    ({'type': 'disabled'}, None, False),
    (None, None, True),  # the chat route's default
    (None, {'enable_thinking': False}, False),
    ({'type': 'enabled', 'budget_tokens': 1024}, {'enable_thinking': False, 'reasoning_effort': 'low'}, True),
])
def test_thinking_switch(thinking, kwargs, expected):
    fields = {}
    if thinking is not None:
        fields['thinking'] = thinking
    if kwargs is not None:
        fields['chat_template_kwargs'] = kwargs
    response, _, tmpl = post(body(**fields), 'Hmm.\n</think>\n\nHi.' if expected else 'Hi.')
    assert response.status_code == 200, response.text
    assert tmpl.render.call_args.kwargs['enable_thinking'] is expected
    if kwargs and 'reasoning_effort' in kwargs:
        assert tmpl.render.call_args.kwargs['reasoning_effort'] == 'low'
    content = response.json()['content']
    if expected:
        assert content == [{'type': 'thinking', 'thinking': 'Hmm.', 'signature': messages_api.SIGNATURE},
                           {'type': 'text', 'text': 'Hi.'}]
    else:
        assert content == [{'type': 'text', 'text': 'Hi.'}]


def test_unknown_fields_headers_and_query_are_ignored():
    async def scenario():
        engine = fake_engine('ok')
        app, _ = fake_app(engine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/messages?beta=true', json=body(
                thinking={'type': 'disabled'}, metadata={'user_id': 'u'}, betas=['x'], context_management={'edits': []}, service_tier='auto',
                tool_choice={'type': 'auto', 'disable_parallel_tool_use': True}),
                headers={'anthropic-version': '2023-06-01', 'anthropic-beta': 'fine-grained-tool-streaming-2025-05-14',
                         'x-api-key': 'unused'})
        assert response.status_code == 200, response.text
        assert response.json()['content'] == [{'type': 'text', 'text': 'ok'}]
    asyncio.run(scenario())


def test_sampling_fields_and_extensions_reach_the_engine():
    response, engine, _ = post(body(temperature=0.5, top_p=0.9, top_k=40, stop_sequences=['END'], trace=False))
    assert response.status_code == 200
    params = engine.submit.await_args.args[1]
    assert (params.temperature, params.top_p, params.top_k, params.max_new_tokens, params.trace) == (0.5, 0.9, 40, 256, False)
    _, engine, _ = post(body(temperature=0, accept_top_k=3, accept_margin=0.5))
    params = engine.submit.await_args.args[1]
    assert (params.accept_top_k, params.accept_margin, params.trace) == (3, 0.5, True)


def test_max_tokens_is_capped_by_the_context():
    _, engine, _ = post(body(max_tokens=32000), describe={'context_length': 100})
    assert engine.submit.await_args.args[1].max_new_tokens == 100 - len(encode(render([], enable_thinking=True)))
    response, engine, _ = post(body(max_tokens=10), describe={'context_length': 8})
    assert response.status_code == 400
    assert response.json()['error']['message'] == 'prompt is too long: 8 tokens > 7 maximum'
    engine.submit.assert_not_awaited()


def test_model_is_echoed():
    response, _, _ = post(body(model='claude-opus-4-1'), 'ok')
    assert response.json()['model'] == 'claude-opus-4-1'
    response, _, _ = post({'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'hi'}]}, 'ok')
    assert response.json()['model'] == 'RadixArk/Qwen3.8-27B-NVFP4'


# ---- rejections ----
@pytest.mark.parametrize('payload,message', [
    (body(max_tokens=None), 'max_tokens'),
    ({k: v for k, v in body().items() if k != 'max_tokens'}, 'max_tokens'),
    (body(max_tokens=0), 'max_new_tokens'),
    (body(messages=[]), 'messages'),
    (body(messages=[{'role': 'system', 'content': 'x'}]), 'role'),
    (body(messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]), 'text-only'),
    (body(messages=[{'role': 'user', 'content': [{'type': 'document', 'source': {}}]}]), 'text-only'),
    (body(messages=[{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a',
                                                  'content': [{'type': 'image', 'source': {}}]}]}]), 'text-only'),
    (body(system=[{'type': 'image', 'source': {}}]), 'text-only'),
    (body(messages=[{'role': 'user', 'content': [{'type': 'search_result'}]}]), "'search_result'"),
    (body(messages=[{'role': 'user', 'content': [{'type': 'tool_use', 'name': 'x', 'input': {}}]}]), "'tool_use'"),
    (body(thinking={'type': 'sometimes'}), 'thinking'),
    (body(tools=[{'name': 'x'}]), 'input_schema'),
    (body(tool_choice={'type': 'required'}), 'tool_choice'),
    (body(stop_sequences=['']), 'stop'),
    (body(temperature=-1), 'temperature'),
])
@pytest.mark.parametrize('stream', [False, True])
def test_invalid_requests_return_anthropic_errors(payload, message, stream):
    response, engine, _ = post({**payload, 'stream': stream})
    assert response.status_code == 400
    assert response.json()['type'] == 'error'
    assert response.json()['error']['type'] == 'invalid_request_error'
    assert message in response.json()['error']['message']
    engine.submit.assert_not_awaited()


def test_non_object_body_is_rejected():
    async def scenario():
        app, _ = fake_app(fake_engine())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            for content in (b'not json', b'[1, 2]'):
                response = await client.post('/v1/messages', content=content)
                assert response.status_code == 400
                assert response.json()['error']['type'] == 'invalid_request_error'
    asyncio.run(scenario())


def test_forced_tool_choice_is_rejected_and_counted():
    async def scenario():
        engine = fake_engine('ok')
        app, tmpl = fake_app(engine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            for choice in ({'type': 'any'}, {'type': 'tool', 'name': 'get_weather'}):
                response = await client.post('/v1/messages', json=body(tools=TOOLS, tool_choice=choice))
                assert response.status_code == 400
                error = response.json()['error']
                assert error['type'] == 'invalid_request_error' and 'not supported' in error['message']
            tmpl.render.assert_not_called()
            engine.submit.assert_not_awaited()
            info = (await client.get('/get_server_info')).json()
            assert info['messages_api']['forced_tool_choice_rejected'] == 2
            assert 'sharar_messages_forced_tool_choice_rejected 2' in (await client.get('/metrics')).text
            response = await client.post('/v1/messages', json=body(tools=[*TOOLS, {'type': 'web_search_20250305', 'name': 'web_search'}]))
            assert response.status_code == 200
            assert tmpl.render.call_args.kwargs['tools'][0]['function']['name'] == 'get_weather'
            assert len(tmpl.render.call_args.kwargs['tools']) == 1
            assert (await client.get('/get_server_info')).json()['messages_api']['server_tools_ignored'] == 1
    asyncio.run(scenario())


def test_tool_choice_none_drops_tool_schemas():
    response, _, tmpl = post(body(tools=TOOLS, tool_choice={'type': 'none'}, thinking={'type': 'disabled'}), 'Sure. ' + CALL)
    assert tmpl.render.call_args.kwargs['tools'] is None
    assert response.json()['stop_reason'] == 'end_turn'
    assert [b['type'] for b in response.json()['content']] == ['text']


@pytest.mark.parametrize('stream', [False, True])
def test_queue_full_and_engine_unavailable(stream):
    for error, status, kind in ((QueueFull('queue full'), 429, 'rate_limit_error'),
                                (EngineUnavailable('engine unavailable'), 503, 'overloaded_error')):
        async def scenario():
            engine = fake_engine()
            engine.submit = AsyncMock(side_effect=error)
            app, _ = fake_app(engine)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                return await client.post('/v1/messages', json=body(stream=stream))
        response = asyncio.run(scenario())
        assert response.status_code == status
        assert response.json() == {'type': 'error', 'error': {'type': kind, 'message': str(error)}}


def test_engine_failure_mid_stream_sends_an_error_event():
    events = sse_events(post(body(stream=True), 'partial', reason='error', chunk=100)[0].text)
    assert [e['type'] for e in events] == ['message_start', 'ping', 'error']
    assert events[-1]['error']['type'] == 'overloaded_error'


# ---- responses ----
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('output,reason,stop_reason,stop_sequence,fields', [
    ('Hello.', 'stop', 'end_turn', None, {}),
    ('Hello there', 'length', 'max_tokens', None, {}),
    ('alpha END beta', 'stop', 'stop_sequence', 'END', {'stop_sequences': ['NEVER', 'END']}),
    ('Checking.\n\n' + CALL, 'stop', 'tool_use', None, {'tools': TOOLS}),
])
def test_stop_reasons(stream, output, reason, stop_reason, stop_sequence, fields):
    response, engine, _ = post(body(stream=stream, thinking={'type': 'disabled'}, **fields), output, reason, prefix_len=2)
    assert response.status_code == 200
    usage = {'input_tokens': len('prompt') - 2, 'output_tokens': len(encode(output)) if reason != 'stop' or not stop_sequence
             else len('alpha END'), 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 2}
    if stream:
        events = sse_events(response.text)
        delta = events[-2]
        assert delta['type'] == 'message_delta'
        assert delta['delta'] == {'stop_reason': stop_reason, 'stop_sequence': stop_sequence}
        assert delta['usage']['cache_read_input_tokens'] == 2 and delta['usage']['input_tokens'] == usage['input_tokens']
        text = ''.join(e['delta']['text'] for e in events if e['type'] == 'content_block_delta' and e['delta']['type'] == 'text_delta')
    else:
        result = response.json()
        assert (result['stop_reason'], result['stop_sequence']) == (stop_reason, stop_sequence)
        assert result['usage'] == usage
        text = ''.join(b['text'] for b in result['content'] if b['type'] == 'text')
    assert text == {'tool_use': 'Checking.', 'stop_sequence': 'alpha'}.get(stop_reason, output)
    if stop_sequence:
        engine.cancel.assert_called()


def test_tool_use_block():
    result = post(body(tools=TOOLS, thinking={'type': 'disabled'}), CALL)[0].json()
    assert result['stop_reason'] == 'tool_use'
    [block] = result['content']
    assert block['type'] == 'tool_use' and block['id'].startswith('toolu_') and block['name'] == 'get_weather'
    assert block['input'] == {'city': 'Paris', 'days': 2}
    assert result['id'].startswith('msg_') and result['type'] == 'message' and result['role'] == 'assistant'


@pytest.mark.parametrize('chunk', [1, 4, 1000])
def test_stream_event_sequence(chunk):
    output = '\nLet me look.\n</think>\n\nChecking now.\n\n' + CALL
    response, _, _ = post(body(stream=True, tools=TOOLS, thinking={'type': 'enabled', 'budget_tokens': 1024}),
                          output, chunk=chunk, prefix_len=1)
    assert response.headers['content-type'].startswith('text/event-stream')
    events = sse_events(response.text)
    kinds = [e['type'] if e['type'] != 'content_block_delta' else e['delta']['type'] for e in events]
    squeezed = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
    assert squeezed == ['message_start', 'ping',
                        'content_block_start', 'thinking_delta', 'signature_delta', 'content_block_stop',
                        'content_block_start', 'text_delta', 'content_block_stop',
                        'content_block_start', 'input_json_delta', 'content_block_stop',
                        'message_delta', 'message_stop']
    start = events[0]['message']
    assert start['content'] == [] and start['model'] == 'claude-sonnet-4-5' and start['stop_reason'] is None
    assert start['usage']['input_tokens'] == len(encode(render([], enable_thinking=True))) and start['usage']['output_tokens'] == 0
    starts = [e for e in events if e['type'] == 'content_block_start']
    assert [s['index'] for s in starts] == [0, 1, 2]
    assert starts[0]['content_block'] == {'type': 'thinking', 'thinking': '', 'signature': ''}
    assert starts[1]['content_block'] == {'type': 'text', 'text': ''}
    assert starts[2]['content_block']['type'] == 'tool_use' and starts[2]['content_block']['input'] == {}
    deltas = [e['delta'] for e in events if e['type'] == 'content_block_delta']
    assert ''.join(d.get('thinking', '') for d in deltas) == 'Let me look.'
    assert ''.join(d.get('text', '') for d in deltas) == 'Checking now.'
    assert [d['signature'] for d in deltas if d['type'] == 'signature_delta'] == [messages_api.SIGNATURE]
    assert json.loads(''.join(d.get('partial_json', '') for d in deltas)) == {'city': 'Paris', 'days': 2}
    assert events[-2]['delta']['stop_reason'] == 'tool_use' and events[-2]['usage']['output_tokens'] == len(encode(output))


def test_whitespace_only_output_has_no_blocks():
    result = post(body(thinking={'type': 'disabled'}), '\n\n')[0].json()
    assert result['content'] == [] and result['stop_reason'] == 'end_turn'
    events = sse_events(post(body(stream=True, thinking={'type': 'disabled'}), '\n\n')[0].text)
    assert [e['type'] for e in events] == ['message_start', 'ping', 'message_delta', 'message_stop']


# ---- count_tokens ----
def test_count_tokens_matches_the_rendered_prompt():
    payload = {'model': 'claude-haiku', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': TOOLS,
               'tool_choice': {'type': 'any'}, 'thinking': {'type': 'enabled', 'budget_tokens': 10}}
    response, engine, tmpl = post(payload, path='/v1/messages/count_tokens')
    assert response.json() == {'input_tokens': len(encode(render([], enable_thinking=True)))}
    assert tmpl.render.call_args.kwargs['tools'] is not None
    engine.submit.assert_not_awaited()
    response, _, _ = post({**payload, 'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}, path='/v1/messages/count_tokens')
    assert response.status_code == 400 and response.json()['error']['type'] == 'invalid_request_error'


# ---- cancellation ----
@pytest.mark.parametrize('stream', [False, True])
def test_client_disconnect_cancels_the_request(stream):
    async def scenario():
        queue = asyncio.Queue()  # the engine never produces output
        ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))
        engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock())
        app, _ = fake_app(engine)
        payload = json.dumps(body(stream=stream)).encode()
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': payload, 'more_body': False}
            return {'type': 'http.disconnect'}

        async def send(message):
            pass

        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                 'scheme': 'http', 'path': '/v1/messages', 'raw_path': b'/v1/messages', 'query_string': b'',
                 'root_path': '', 'headers': [(b'content-type', b'application/json')],
                 'client': ('test', 1), 'server': ('test', 80)}
        await asyncio.wait_for(app(scope, receive, send), 5)
        engine.cancel.assert_called_with(ticket)
    asyncio.run(scenario())


# ---- the real template: identical to the equivalent OpenAI chat request ----
def test_rendering_matches_the_openai_route(snapshot):
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer

    tok, tmpl = SnapshotTokenizer(snapshot), ChatTemplate(snapshot)
    anthropic = {
        'model': 'claude-sonnet-4-5', 'max_tokens': 64, 'thinking': {'type': 'enabled', 'budget_tokens': 32},
        'system': [{'type': 'text', 'text': 'You are a helper.', 'cache_control': {'type': 'ephemeral'}},
                   {'type': 'text', 'text': ' Be brief.'}],
        'tools': TOOLS,
        'messages': [
            {'role': 'user', 'content': "What's the weather in Paris?"},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'I should call the tool.', 'signature': messages_api.SIGNATURE},
                {'type': 'text', 'text': 'Checking.'},
                {'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {'city': 'Paris', 'days': 2}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'sunny'},
                {'type': 'text', 'text': 'Thanks, and tomorrow?'}]}]}
    openai = {
        'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': True},
        'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city',
                                                    'parameters': TOOLS[0]['input_schema']}}],
        'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helper.'}, {'type': 'text', 'text': ' Be brief.'}]},
            {'role': 'user', 'content': "What's the weather in Paris?"},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Checking.'}],
             'reasoning_content': 'I should call the tool.',
             'tool_calls': [{'id': 'toolu_1', 'type': 'function',
                             'function': {'name': 'get_weather', 'arguments': {'city': 'Paris', 'days': 2}}}]},
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'sunny'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Thanks, and tomorrow?'}]}]}

    async def scenario():
        prompts = {}
        for path, payload in (('/v1/messages', anthropic), ('/v1/chat/completions', openai)):
            engine = fake_engine('ok')
            app = create_app(engine, tok, tmpl, {})
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await client.post(path, json=payload)
                assert response.status_code == 200, response.text
                if path == '/v1/messages':
                    count = (await client.post('/v1/messages/count_tokens', json=anthropic)).json()['input_tokens']
            prompts[path] = engine.submit.await_args.args[0]
        return prompts, count

    prompts, count = asyncio.run(scenario())
    ids = prompts['/v1/messages']
    assert ids == prompts['/v1/chat/completions'] and count == len(ids)
    text = tok.decode(ids, skip_special_tokens=False)
    assert 'You are a helper. Be brief.' in text
    assert '<think>\nI should call the tool.\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis' in text
    assert '<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>user\nThanks, and tomorrow?<|im_end|>' in text
    assert text.endswith('<|im_start|>assistant\n<think>\n')


def test_openai_string_tool_arguments_render_like_objects(snapshot):
    # OpenAI clients (OpenCode, Cline, Aider) send earlier calls' arguments as a JSON string.
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer

    tok, tmpl = SnapshotTokenizer(snapshot), ChatTemplate(snapshot)

    def history(arguments):
        return {'max_tokens': 16, 'tools': [{'type': 'function', 'function': {
                    'name': 'get_weather', 'description': 'Weather for a city', 'parameters': TOOLS[0]['input_schema']}}],
                'messages': [{'role': 'user', 'content': 'Weather in Paris?'},
                             {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
                                 'name': 'get_weather', 'arguments': arguments}}]},
                             {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'sunny'}]}

    async def scenario():
        prompts, statuses = [], []
        for arguments in ({'city': 'Paris', 'days': 2}, '{"city": "Paris", "days": 2}', '{"city": '):
            engine = fake_engine('ok')
            async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine, tok, tmpl, {})), base_url='http://test') as client:
                response = await client.post('/v1/chat/completions', json=history(arguments))
            statuses.append(response.status_code)
            prompts.append(engine.submit.await_args.args[0] if response.status_code == 200 else None)
        return prompts, statuses

    prompts, statuses = asyncio.run(scenario())
    assert statuses == [200, 200, 400]
    assert prompts[0] == prompts[1]
