"""Completion reports preserve engine token IDs across speculative chunks."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from hibana.api.server import create_app


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('report_ids', [False, True])
@pytest.mark.parametrize('report_usage', [False, True])
def test_completion_reports_exact_ids_and_usage(stream, report_ids, report_usage):
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([5], False, None))
        queue.put_nowait(([6, 7], True, 'length'))
        ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=2))
        engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(), decode=lambda ids, **kw: ''.join(map(str, ids)))
        app = create_app(engine, tok, SimpleNamespace(), {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/completions', json={
                'prompt': [10, 11, 12], 'stream': stream, 'max_tokens': 3,
                'temperature': 0, 'ignore_eos': True, 'return_token_ids': report_ids,
                'stream_options': {'include_usage': report_usage}})
        assert response.status_code == 200
        tok.encode.assert_not_called()
        assert engine.submit.call_args.args[0] == [10, 11, 12]
        if stream:
            events = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
            assert events[-1] == '[DONE]'
            chunks = [json.loads(event) for event in events[:-1]]
            choices = [choice for chunk in chunks for choice in chunk['choices']]
            assert ''.join(choice['text'] for choice in choices) == '567'
            assert choices[-1]['finish_reason'] == 'length'
            usages = [chunk['usage'] for chunk in chunks if 'usage' in chunk]
            assert len(usages) == int(report_usage)
            if report_ids:
                assert [choice['token_ids'] for choice in choices] == [[5], [6, 7]]
                assert choices[0]['prompt_token_ids'] == [10, 11, 12]
                assert 'prompt_token_ids' not in choices[1]
            else:
                assert all('token_ids' not in choice and 'prompt_token_ids' not in choice for choice in choices)
        else:
            result = response.json()
            choice = result['choices'][0]
            assert choice['text'] == '567'
            usages = [result['usage']]
            if report_ids:
                assert choice['token_ids'] == [5, 6, 7]
                assert choice['prompt_token_ids'] == [10, 11, 12]
            else:
                assert 'token_ids' not in choice and 'prompt_token_ids' not in choice
        for usage in usages:
            assert usage == {'prompt_tokens': 3, 'completion_tokens': 3, 'total_tokens': 6,
                             'prompt_tokens_details': {'cached_tokens': 2}}
    asyncio.run(scenario())


@pytest.mark.parametrize('stream', [False, True])
def test_chat_usage_reports_cached_prompt_tokens(stream):
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([5, 6], True, 'stop'))
        ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=2))
        engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[10, 11, 12]), decode=lambda ids, **kw: 'ok')
        template = SimpleNamespace(render=Mock(return_value='prompt'))
        app = create_app(engine, tok, template, {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream,
                'chat_template_kwargs': {'enable_thinking': False}, 'stream_options': {'include_usage': True}})
        assert response.status_code == 200
        if stream:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            usage = [chunk['usage'] for chunk in chunks if 'usage' in chunk][-1]
        else:
            usage = response.json()['usage']
        assert usage['prompt_tokens'] == 3 and usage['completion_tokens'] == 2
        assert usage['prompt_tokens_details'] == {'cached_tokens': 2}
    asyncio.run(scenario())


def test_streamed_chat_keeps_text_before_a_call_in_the_same_update():
    async def scenario():
        text = 'Sure.<tool_call>\n<function=run>\n</function>\n</tool_call>'
        queue = asyncio.Queue()
        queue.put_nowait(([ord(c) for c in 'I'], False, None))
        queue.put_nowait(([ord(c) for c in text], True, 'stop'))  # several tokens in one speculative round
        ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0))
        engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[1]), decode=lambda ids, **kw: ''.join(map(chr, ids)))
        app = create_app(engine, tok, SimpleNamespace(render=Mock(return_value='prompt')), {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True,
                'chat_template_kwargs': {'enable_thinking': False},
                'tools': [{'type': 'function', 'function': {'name': 'run', 'parameters': {'type': 'object'}}}]})
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        deltas = [chunk['choices'][0]['delta'] for chunk in chunks]
        assert ''.join(d.get('content', '') for d in deltas) == 'ISure.'
        assert [c['function']['name'] for d in deltas for c in d.get('tool_calls', [])] == ['run']
        assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    asyncio.run(scenario())
