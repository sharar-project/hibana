import asyncio
import threading
import multiprocessing
import os
import signal
import socket
import time
from types import SimpleNamespace

import pytest
import torch
import httpx

from sharar.engine.loop import EngineLoop, EngineUnavailable
from sharar.api.validation import _sampling_from
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.engine.contracts import ProposerContext
from sharar.proposers.mtp import MtpProposer
from tests.unit.test_runtime_safety import tiny_executor


def _shutdown_server(pipe):
    import uvicorn
    from contextlib import asynccontextmanager
    from sharar.api.server import create_app
    with pytest.MonkeyPatch.context() as patches:
        ex = tiny_executor(patches)
        original = ex.decode_step
        def slow(reqs):
            time.sleep(.01)
            return original(reqs)
        ex.decode_step = slow
        engine = EngineLoop(Scheduler(ex))
        engine.start()
        tok = SimpleNamespace(encode=lambda text: [1], decode=lambda ids, **kwargs: 'x' * len(ids))
        app = create_app(engine, tok, None, {})
        original_lifespan = app.router.lifespan_context
        @asynccontextmanager
        async def observe_shutdown(app):
            async with original_lifespan(app):
                yield
            pipe.send((engine.failure, engine.sched.active, ex.kv_pool.free_count))
        app.router.lifespan_context = observe_shutdown
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        pipe.send(sock.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', timeout_graceful_shutdown=5))
        try:
            server.run(sockets=[sock])
        finally:
            engine.drain(2)
            engine.thread.join(3)


@pytest.mark.skipif(os.name != 'posix', reason='SIGTERM lifecycle on the serving platform')
def test_sigterm_drains_an_inflight_http_stream():
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_shutdown_server, args=(child,))
    process.start()
    try:
        assert parent.poll(10)
        url = f'http://127.0.0.1:{parent.recv()}'
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(url + '/health').status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                time.sleep(.05)
            with client.stream('POST', url + '/v1/completions', json={
                    'prompt': [1, 2], 'max_tokens': 128.0, 'temperature': 0, 'stream': True}) as response:
                assert response.status_code == 200
                lines = response.iter_lines()
                first = next(line for line in lines if line.startswith('data:'))
                os.kill(process.pid, signal.SIGTERM)
                text = first + '\n' + '\n'.join(lines)
                assert '"finish_reason": "length"' in text and '[DONE]' in text
                assert '"error"' not in text
        assert parent.poll(10)
        assert parent.recv() == (None, 0, 1024)
        process.join(5)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()
        child.close()


def test_http_integral_numbers_keep_internal_integer_contract():
    params = _sampling_from({'max_tokens': 128.0, 'top_k': 20.0, 'seed': 7.0})
    assert type(params.max_new_tokens) is int and params.max_new_tokens == 128
    assert type(params.top_k) is int and type(params.seed) is int
    with pytest.raises(ValueError):
        SamplingParams(max_new_tokens=128.0).validate()


def test_engine_thread_owns_inference_context_for_startup_tensors(monkeypatch):
    async def scenario():
        with torch.inference_mode():
            ex = tiny_executor(monkeypatch)
        engine = EngineLoop(Scheduler(ex))
        engine.start()
        try:
            ticket = await engine.submit([1, 2], SamplingParams(max_new_tokens=3))
            while True:
                _, done, reason = await asyncio.wait_for(ticket.queue.get(), 2)
                if done:
                    assert reason == 'length' and len(ticket.req.output_ids) == 3
                    break
            assert engine.failure is None
        finally:
            engine.drain(1)
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('deadline', [0, 2])
def test_drain_finishes_accepted_work_or_bounds_it(monkeypatch, deadline):
    async def scenario():
        ex = tiny_executor(monkeypatch)
        engine = EngineLoop(Scheduler(ex))
        entered, proceed = threading.Event(), threading.Event()
        original = ex.decode_step
        def pause(reqs):
            entered.set()
            assert proceed.wait(2)
            return original(reqs)
        ex.decode_step = pause
        engine.start()
        try:
            ticket = await engine.submit([1, 2], SamplingParams(max_new_tokens=8))
            assert await asyncio.to_thread(entered.wait, 2)
            engine.drain(deadline)
            assert not engine.healthy
            with pytest.raises(EngineUnavailable):
                await engine.submit([1], SamplingParams())
            proceed.set()
            tokens, reason = [], None
            while True:
                new, done, reason = await asyncio.wait_for(ticket.queue.get(), 2)
                tokens += new
                if done:
                    break
            await asyncio.to_thread(engine.thread.join, 2)
            assert not engine.thread.is_alive() and engine.sched.active == 0
            assert len(ex.kv_pool._free) == 1024
            if deadline:
                assert len(tokens) == 8 and reason == 'length' and engine.failure is None
            else:
                assert reason == 'error' and 'deadline' in engine.failure
        finally:
            proceed.set()
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


def test_mtp_bonus_and_cached_prefix_never_read_unwritten_draft_kv(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    kv = torch.full((1025,), float('nan'))
    writes = []
    def write(tokens, hidden, positions, pages):
        assert len(tokens) == len(hidden) == len(pages)
        kv[pages.long()] = hidden[:, 0]
        writes.extend(pages.tolist())
    def forward(tokens, hidden, positions, pages, qo_lens, kv_pages):
        write(tokens, hidden, positions, pages)
        assert all(torch.isfinite(kv[p.long()]).all() for p in kv_pages)
        return hidden
    def logits(hidden):
        out = torch.zeros((len(hidden), 128))
        out[:, 7] = 10
        return out
    ex.model.logits = logits
    from sharar.proposers.draft_head import TargetHead
    head = SimpleNamespace(bytes=0, write_kv=write, forward=forward, logits=logits, draft_head=TargetHead(logits), head_input=lambda d: d)
    proposer = MtpProposer(head, ProposerContext(ex.device, 4, 512))
    ex.set_proposer(proposer)
    for _ in range(2):
        r = ex.admit([7] * 70, SamplingParams(max_new_tokens=13))
        ex.prefill(r)
        while not r.finished:
            ex.decode_step([r])
            assert torch.isfinite(kv[r.pages[1:r.committed_len].long()]).all()
        assert r.spec_emitted == 12
        ex.release(r)
    assert proposer.cache_hits == 1 and proposer.cache_fallbacks == 0
    # Missing/foreign schema keeps the target hit but uses ordinary decode.
    match = ex.cache.match([7] * 69)
    match.node.auxiliary.clear()
    r = ex.admit([7] * 70, SamplingParams(max_new_tokens=3))
    ex.prefill(r)
    assert r.prefix_len > 0 and not proposer.ready(r)
    while not r.finished:
        ex.decode_step([r])
    assert r.spec_rounds == 0 and r.output_ids == [7] * 3
    ex.release(r)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024


def test_prefill_checkpoints_align_to_each_forward_not_absolute_positions(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    assert ex._checkpoint_positions(1910, 2172, 2172) == [2102, 2166]
    assert ex._checkpoint_positions(0, 2172, 2172) == [2048, 2112]
    for start in range(1900, 2050):
        positions = ex._checkpoint_positions(start, 2172, 2172)
        assert all(start < p < 2172 and (p - start) % 64 == 0 for p in positions)
        assert not positions or 2172 - positions[-1] <= 64
