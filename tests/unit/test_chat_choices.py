"""n > 1 on the chat route: one primer prefills and publishes the prompt, then
n samples with their own seeds, returned as n choices."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from sharar.api.server import create_app
from tests.unit.test_messages_api import decode, encode, render


def engine_answering(outputs):
    """A fake engine whose k-th submission streams outputs[k]."""
    submitted = []

    async def submit(ids, params):
        k = len(submitted)
        submitted.append((ids, params))
        queue = asyncio.Queue()
        queue.put_nowait((encode(outputs[min(k, len(outputs) - 1)]), True, "stop"))
        return SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=7 if k == 0 else 1000))

    engine = SimpleNamespace(submit=AsyncMock(side_effect=submit), cancel=Mock(), stats=lambda: {},
                             sched=SimpleNamespace(ex=SimpleNamespace(spec=None)))
    return engine, submitted


def app_for(engine):
    tok = SimpleNamespace(encode=encode, decode=decode)
    tmpl = SimpleNamespace(render=render)
    return create_app(engine, tok, tmpl, {})


def post(engine, body):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app_for(engine)), base_url="http://test") as client:
            return await client.post("/v1/chat/completions", json=body)
    return asyncio.run(go())


BODY = {"messages": [{"role": "user", "content": "count"}], "max_tokens": 16, "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False}}


def test_n_choices_share_one_prefill_and_get_distinct_seeds():
    engine, submitted = engine_answering(["x", "one", "two", "three"])
    response = post(engine, {**BODY, "n": 3, "seed": 40})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [c["message"]["content"] for c in data["choices"]] == ["one", "two", "three"]
    assert [c["index"] for c in data["choices"]] == [0, 1, 2]
    primer, *samples = [params for _, params in submitted]
    assert primer.max_new_tokens == 1 and primer.trace is False and primer.seed is None
    assert [p.seed for p in samples] == [40, 41, 42] and all(p.max_new_tokens == 16 for p in samples)
    assert len({tuple(ids) for ids, _ in submitted}) == 1  # the same prompt every time
    assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 7  # what the primer reused


@pytest.mark.parametrize("extra", [{"n": 2, "stream": True}, {"n": 0}, {"n": 17}, {"n": 2.0}])
def test_invalid_n_is_rejected(extra):
    engine, submitted = engine_answering(["x"])
    response = post(engine, {**BODY, **extra})
    assert response.status_code == 400 and not submitted


def test_n_one_is_the_ordinary_path():
    engine, submitted = engine_answering(["only"])
    response = post(engine, {**BODY, "n": 1})
    assert response.status_code == 200 and len(submitted) == 1 and response.json()["choices"][0]["message"]["content"] == "only"
