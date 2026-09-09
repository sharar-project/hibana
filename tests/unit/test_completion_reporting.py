"""Completion reports preserve engine token IDs across speculative chunks."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from sharar.api.server import create_app


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
