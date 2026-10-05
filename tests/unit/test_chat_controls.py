"""Unsupported output constraints fail before prompt work or engine admission."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from sharar.api.server import create_app


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('field,value', [
    ('tool_choice', 'required'),
    ('tool_choice', {'type': 'function', 'function': {'name': 'run'}}),
    ('tool_choice', {}), ('tool_choice', []), ('tool_choice', False),
    ('tool_choice', 1), ('tool_choice', 'AUTO'),
    ('response_format', {'type': 'json_schema', 'json_schema': {'name': 'result'}}),
    ('response_format', {'type': 'text', 'schema': {}}),
    ('response_format', {}), ('response_format', []),
    ('response_format', False), ('response_format', 'text'),
])
def test_malformed_controls_rejected_before_render_and_submit(stream, field, value):
    """Malformed choices and formats, and a forced tool_choice without tools."""
    async def scenario():
        engine = SimpleNamespace(submit=AsyncMock())
        tok, template = SimpleNamespace(encode=Mock()), SimpleNamespace(render=Mock())
        app = create_app(engine, tok, template, {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream, field: value})
        assert response.status_code == 400
        assert response.json()['error']['type'] == 'invalid_request_error'
        assert field in response.json()['error']['message']
        template.render.assert_not_called()
        tok.encode.assert_not_called()
        engine.submit.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('controls', [{}, {'tool_choice': None, 'response_format': None},
    {'tool_choice': 'auto', 'response_format': {'type': 'text'}}, {'tool_choice': 'none'}])
@pytest.mark.parametrize('tools', [None, [{'type': 'function', 'function': {
    'name': 'run', 'parameters': {'type': 'object', 'properties': {}}}}]])
def test_unconstrained_chat_still_submits_and_returns_output(stream, controls, tools):
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([2], True, 'stop'))
        engine = SimpleNamespace(submit=AsyncMock(return_value=SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[1]), decode=lambda ids, **kwargs: 'ok')
        template = SimpleNamespace(render=Mock(return_value='prompt'))
        app = create_app(engine, tok, template, {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream,
                'tools': tools, **controls})
        assert response.status_code == 200
        engine.submit.assert_awaited_once()
        expected_tools = None if controls.get('tool_choice') == 'none' else tools
        assert template.render.call_args.kwargs['tools'] == expected_tools
        if stream:
            assert '"content": "ok"' in response.text and '[DONE]' in response.text
        else:
            assert response.json()['choices'][0]['message']['content'] == 'ok'
    asyncio.run(scenario())


@pytest.mark.parametrize('body,expected', [({}, 4096 - 3), ({'max_tokens': 100}, 100), ({'max_completion_tokens': 200}, 200)])
def test_chat_without_max_tokens_may_use_the_rest_of_the_context(body, expected):
    """As in the OpenAI API: an unset reply budget is the context left after the prompt, not a fixed small default."""
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([2], True, 'stop'))
        engine = SimpleNamespace(submit=AsyncMock(return_value=SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[1, 1, 1]), decode=lambda ids, **kwargs: 'ok')
        template = SimpleNamespace(render=Mock(return_value='prompt'))
        app = create_app(engine, tok, template, {'context_length': 4096})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}], **body})
        assert response.status_code == 200
        params = engine.submit.await_args.args[1]
        assert params.max_new_tokens == expected
    asyncio.run(scenario())


@pytest.mark.parametrize('controls,kind', [
    ({'response_format': {'type': 'json_object'}}, 'json_object'),
    ({'response_format': {'type': 'json_schema', 'json_schema': {'name': 'w', 'schema': {
        'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}}, 'json_schema'),
    ({'tool_choice': 'required'}, 'tool_call'),
    ({'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}}}, 'tool_call'),
])
def test_constrained_output_requests_carry_their_grammar(controls, kind):
    from tests.unit.test_grammar import PIECES, STOP, ID, V
    from tests.unit.test_messages_api import post
    from sharar.engine.grammar import Grammars
    grammars = Grammars(PIECES, V, [STOP], ID['</think>'], byte_level=False)
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]
    payload = {'messages': [{'role': 'user', 'content': 'hi'}], 'tools': tools, **controls}
    response, engine, _ = post({**payload, 'chat_template_kwargs': {'enable_thinking': False}}, output='{}',
                               path='/v1/chat/completions', grammars=grammars)
    assert response.status_code == 200
    constraint = engine.submit.await_args.args[1].constraint
    assert constraint.kind == kind and constraint.armed is True  # thinking off: the grammar applies from the first token
    response, engine, _ = post(payload, output='{}', path='/v1/chat/completions', grammars=grammars)
    assert engine.submit.await_args.args[1].constraint.armed is False  # thinking (the default): free until </think>


def test_forced_tool_choice_and_json_format_cannot_combine_and_bad_schemas_are_refused():
    from tests.unit.test_grammar import PIECES, STOP, ID, V
    from tests.unit.test_messages_api import post
    from sharar.engine.grammar import Grammars
    grammars = Grammars(PIECES, V, [STOP], ID['</think>'], byte_level=False)
    tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object', 'properties': {}}}}]
    response, engine, _ = post({'messages': [{'role': 'user', 'content': 'hi'}], 'tools': tools, 'tool_choice': 'required',
                                'response_format': {'type': 'json_object'}}, path='/v1/chat/completions', grammars=grammars)
    assert response.status_code == 400 and 'cannot be combined' in response.json()['error']['message']
    response, engine, _ = post({'messages': [{'role': 'user', 'content': 'hi'}], 'response_format': {
        'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object', 'properties': {'a': {'type': 'not-a-type'}}}}}},
        path='/v1/chat/completions', grammars=grammars)
    assert response.status_code == 400 and 'schema' in response.json()['error']['message']
    engine.submit.assert_not_awaited()
