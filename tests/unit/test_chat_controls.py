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
    ('response_format', {'type': 'json_object'}),
    ('response_format', {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': {}}}),
    ('response_format', {'type': 'text', 'schema': {}}),
    ('response_format', {}), ('response_format', []),
    ('response_format', False), ('response_format', 'text'),
])
def test_unsupported_controls_rejected_before_render_and_submit(stream, field, value):
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
    {'tool_choice': 'auto', 'response_format': {'type': 'text'}}])
@pytest.mark.parametrize('tools', [None, [{'type': 'function', 'function': {
    'name': 'run', 'parameters': {'type': 'object', 'properties': {}}}}]])
def test_unconstrained_chat_still_submits_and_returns_output(stream, controls, tools):
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([2], True, 'stop'))
        engine = SimpleNamespace(submit=AsyncMock(return_value=SimpleNamespace(queue=queue)), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[1]), decode=lambda ids, **kwargs: 'ok')
        template = SimpleNamespace(render=Mock(return_value='prompt'))
        app = create_app(engine, tok, template, {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream,
                'tools': tools, **controls})
        assert response.status_code == 200
        engine.submit.assert_awaited_once()
        assert template.render.call_args.kwargs['tools'] == tools
        if stream:
            assert '"content": "ok"' in response.text and '[DONE]' in response.text
        else:
            assert response.json()['choices'][0]['message']['content'] == 'ok'
    asyncio.run(scenario())
