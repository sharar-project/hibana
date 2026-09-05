"""Small CPU model, real executor/control paths; no checkpoint or GPU needed."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import torch

from sharar import config
from sharar.api.server import EngineLoop, EngineUnavailable, _sampling_from, create_app
from sharar.engine.executor import Executor
from sharar.engine.graphs import capture_sizes, check_context_capacity
from sharar.engine.request import Request
from sharar.engine.sampler import InvalidRequest, Sampler, SamplingParams
from sharar.engine.scheduler import Scheduler, SchedulerConfig
from sharar.engine.spec import SpecDecoder
from sharar.engine.contracts import ProposerContext
from sharar.proposers.mtp import MtpProposer, MtpState
from sharar.memory.request_state import StateManager
from sharar.memory.radix import RadixCache
from sharar.memory.state_pool import StatePool
from tests.unit.test_scheduler import FakeExecutor, FakePool


def tiny_executor(monkeypatch, cache=False):
    ex = Executor.__new__(Executor)
    ex.device = torch.device('cpu')
    ex.max_context, ex.max_batch = 512, 4
    ex.kv_pool = FakePool(1025)
    ex.kv_pool._free.remove(0)
    ex.kv_pool.num_pages, ex.kv_pool.page_size = 1025, 1
    ex.state_pool = StatePool.__new__(StatePool)
    ex.state_pool.ssm_all = torch.zeros((1, 16, 1))
    ex.state_pool.conv_all = torch.zeros((1, 16, 1))
    ex.state_pool._free = list(range(15, 0, -1))
    ex.state_pool.layer_ids = [0]
    ex.state = StateManager(ex.kv_pool, ex.state_pool)
    ex.cache = RadixCache(ex.kv_pool, ex.state_pool, ex.device) if cache else None
    ex.spec, ex.graphs, ex.use_graphs = None, None, False
    ex.sampler = Sampler(ex.device)
    ex._paged_prefill = SimpleNamespace(plan=lambda *a, **kw: None)
    ex._decode_wrapper = SimpleNamespace(plan=lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    def forward(ids, ctx):
        if ctx.mode == 'decode':
            groups = [(i, i + 1, slot) for i, slot in enumerate(ctx.slots.tolist())]
        elif ctx.cu_seqlens is not None:
            cu = ctx.cu_seqlens.tolist()
            groups = [(cu[i], cu[i + 1], slot) for i, slot in enumerate(ctx.seq_slots.tolist())]
        else:
            groups = [(0, len(ids), ctx.slot)]
        norms = []
        for gi, (start, end, slot) in enumerate(groups):
            s = ex.state_pool.ssm_all[0, slot, 0].item()
            for i in range(start, end):
                # Exact integer recurrence makes replay/state corruption visible.
                s = (s * 31 + ids[i].item()) % 97
                if ctx.mode == 'verify':
                    ctx.inter_ssm[0][gi, i - start] = s
                    ctx.verify_conv_windows[0][gi, i - start] = s
                else:
                    ex.state_pool.ssm_all[0, slot, 0] = s
                    ex.state_pool.conv_all[0, slot, 0] = s
                norms.append(s)
        return {'norm': torch.tensor(norms).reshape(-1, 1)}

    def logits(norm):
        out = torch.zeros((norm.shape[0], 128))
        out.scatter_(1, norm.long() + 1, 6)
        return out

    ex.model = SimpleNamespace(forward=forward, logits=logits)
    return ex


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('sampled', [False, True])
def test_retraction_replays_without_changing_delivered_output_or_rng(monkeypatch, batched, sampled):
    ex = tiny_executor(monkeypatch)
    params = SamplingParams(max_new_tokens=12, temperature=1.0 if sampled else 0.0, seed=42)
    baseline = ex.generate([1, 2, 3], params).output_ids
    r = ex.admit([1, 2, 3], params)
    ex.prefill(r)
    ex.decode_step([r])
    delivered = r.new_tokens()
    first_time, generator = r.t_first_token, r.generator
    ex.retract(r)
    assert r.output_ids == delivered and r.new_tokens() == []
    assert r.generator is generator and r.t_first_token == first_time
    ex.admit(r.prompt_ids, r.params, r)
    if batched:
        other = ex.admit([5, 6], SamplingParams(max_new_tokens=1))
        ex.prefill_batch([(r, r.prefill_len), (other, other.prefill_len)])
        ex.release(other)
    else:
        ex.prefill(r, max_tokens=2)
        ex.prefill(r)
    while not r.finished:
        ex.decode_step([r])
        delivered += r.new_tokens()
    assert delivered == r.output_ids == baseline
    assert r.committed_len == r.context_len - 1
    ex.release(r)
    assert len(ex.kv_pool._free) == 1024 and len(ex.state_pool._free) == 15


def test_seeded_requests_are_independent_of_interleaving():
    sampler = Sampler(torch.device('cpu'))
    params = SamplingParams(temperature=1, top_k=-1, top_p=1, seed=77)
    logits = torch.zeros((1, 100))
    expected_gen = sampler.generator(params.seed)
    expected = [sampler.sample(logits, [params], [expected_gen]).item() for _ in range(20)]
    a, b = sampler.generator(params.seed), sampler.generator(params.seed)
    assert a is not b
    actual_a, actual_b = [], []
    for i in range(20):
        if i % 2:
            actual_a.append(sampler.sample(logits, [params], [a]).item())
            actual_b.append(sampler.sample(logits, [params], [b]).item())
        else:
            both = sampler.sample(logits.expand(2, -1), [params, params], [b, a]).tolist()
            actual_b.append(both[0])
            actual_a.append(both[1])
    assert actual_a == actual_b == expected


@pytest.mark.parametrize('temperature', [0, 1])
def test_real_scheduler_retraction_under_pool_pressure_preserves_stream(monkeypatch, temperature):
    ex = tiny_executor(monkeypatch)
    items = [([1, 2, 3], SamplingParams(max_new_tokens=40, temperature=temperature, seed=1)),
             ([5, 6, 7], SamplingParams(max_new_tokens=45, temperature=temperature, seed=1))]
    expected = [ex.generate(ids, params).output_ids for ids, params in items]
    ex.kv_pool.num_pages, ex.kv_pool._free = 61, list(range(60, 0, -1))
    sched = Scheduler(ex, SchedulerConfig(reserve_tokens=0))
    requests = [sched.submit(ids, params) for ids, params in items]
    streamed = [[], []]
    def publish():
        for i, r in enumerate(requests):
            streamed[i].extend(r.new_tokens())
    sched.on_progress = publish
    for _ in range(150):
        if not sched.active:
            break
        sched.step()
    assert sched.active == 0 and sched.metrics.retractions > 0
    assert streamed == [r.output_ids for r in requests] == expected
    assert all(r.released for r in requests)
    assert len(ex.kv_pool._free) == len(set(ex.kv_pool._free)) == 60


def test_cache_rejects_mislabeled_committed_state(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    r = ex.admit([1, 2, 3], SamplingParams(max_new_tokens=1))
    ex.prefill(r)
    r.committed_len += 1
    ex.release(r)
    assert ex.cache.cached_tokens == 0
    assert len(ex.kv_pool._free) == 1024 and len(ex.state_pool._free) == 15


def test_mixed_spec_plain_memory_failure_does_not_advance_any_row(monkeypatch):
    ex = tiny_executor(monkeypatch)
    a = ex.admit([1], SamplingParams(max_new_tokens=10))
    b = ex.admit([2], SamplingParams(max_new_tokens=10, temperature=1))
    ex.prefill_batch([(a, 1), (b, 1)])
    called = []
    ex.spec = SimpleNamespace(steps=4, ready=lambda r: True, round=lambda rs: called.append(rs))
    ex.kv_pool._free = ex.kv_pool._free[:4]  # enough for spec alone, not both paths
    before = [a.output_ids[:], b.output_ids[:]]
    with pytest.raises(MemoryError):
        ex.decode_step([a, b])
    assert not called and [a.output_ids, b.output_ids] == before


def test_failed_spec_round_releases_uncommitted_reservations(monkeypatch):
    ex = tiny_executor(monkeypatch)
    r = ex.admit([1, 2, 3], SamplingParams(max_new_tokens=8))
    ex.prefill(r)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex, spec.device, spec.k, spec.steps, spec.timing = ex, ex.device, 3, 4, False
    spec._host_rows = torch.empty((3, 4), dtype=torch.int64)
    def allocate(n):
        pages = ex.kv_pool.alloc(n)
        return pages, pages.tolist()
    def fail(*args):
        raise RuntimeError('draft failure after reservation')
    ex.kv_pool.alloc_with_ids = allocate
    ex.kv_pool.free_ids = ex.kv_pool._free.extend
    spec.proposer = MtpProposer(SimpleNamespace(forward=fail, bytes=0), ProposerContext(ex.device, 4, 512), 3)
    r.proposer_state = MtpState(draft_pos=r.committed_len, pending_token=r.output_ids[-1], pending_hidden=torch.zeros((1, 1)))
    with pytest.raises(RuntimeError, match='draft failure'):
        spec.round([r])
    assert len(r.reservation.ids) == 4
    r.cacheable = False
    ex.release(r)
    ex.release(r)
    assert len(ex.kv_pool._free) == len(set(ex.kv_pool._free)) == 1024
    assert len(ex.state_pool._free) == 15


@pytest.mark.parametrize('body', [
    {'max_tokens': 0}, {'max_tokens': -1}, {'max_tokens': 1.5}, {'max_tokens': True},
    {'temperature': -1}, {'temperature': float('nan')}, {'temperature': float('inf')},
    {'top_p': 0}, {'top_p': 1.1}, {'top_k': 1.5}, {'seed': '1'}, {'seed': 2**64},
    {'ignore_eos': 'false'}, {'sampling_params': [1]}, {'sampling_params': []},
])
def test_invalid_sampling_is_rejected(body):
    with pytest.raises(InvalidRequest):
        _sampling_from(body)


@pytest.mark.parametrize('ids,budget', [([], 1), ([-1], 1), ([config.VOCAB_SIZE], 1),
    ([True], 1), ([1.5], 1), ([1] * 512, 1), ([1], 512)])
def test_invalid_prompt_rejected_before_admission(monkeypatch, ids, budget):
    ex = tiny_executor(monkeypatch)
    sched = Scheduler(ex)
    with pytest.raises(InvalidRequest):
        sched.submit(ids, SamplingParams(max_new_tokens=budget))
    assert sched.active == 0 and len(ex.kv_pool._free) == 1024


def test_profile_and_verify_capacity(monkeypatch):
    ex = tiny_executor(monkeypatch)
    ex.validate_request([1], SamplingParams(max_new_tokens=511))
    ex.spec = SimpleNamespace(k=3)
    ex.kv_pool.num_pages = 513
    with pytest.raises(InvalidRequest, match='scratch'):
        ex.validate_request([1], SamplingParams(max_new_tokens=511))
    assert capture_sizes(3) == (1, 2, 3)
    check_context_capacity([512, 1], 512)
    with pytest.raises(ValueError):
        check_context_capacity([513, 1], 512)


def test_api_validation_does_not_kill_engine(monkeypatch):
    async def scenario():
        ex = tiny_executor(monkeypatch)
        engine = EngineLoop(Scheduler(ex))
        engine.start()
        tok = SimpleNamespace(encode=lambda text: list(range(len(text))), decode=lambda ids, **kw: 'ok')
        app = create_app(engine, tok, None, {})
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                for body in ({'input_ids': []}, {'input_ids': [-1]}, {'input_ids': [True]},
                             {'input_ids': [1.5]}, {'input_ids': [config.VOCAB_SIZE]},
                             {'input_ids': [1], 'max_tokens': 0},
                             {'input_ids': [1], 'temperature': -1},
                             {'input_ids': [1], 'top_p': 0},
                             {'input_ids': [1] * 512, 'max_tokens': 1}):
                    response = await client.post('/generate', json=body)
                    assert response.status_code == 400, response.text
                    assert engine.healthy and engine.sched.active == 0
                    assert len(ex.kv_pool._free) == 1024
                response = await client.post('/v1/completions', json={'prompt': 'x' * 513, 'max_tokens': 1})
                assert response.status_code == 400
                response = await client.post('/generate', json={'input_ids': [1], 'max_tokens': 1})
                assert response.status_code == 200
                assert (await client.get('/health')).status_code == 200
        finally:
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


def test_scheduler_counts_speculative_outputs():
    ex = FakeExecutor(pages=100)
    def decode(reqs):
        for r in reqs:
            for _ in range(4):
                r.push(1001)
    ex.decode_step = decode
    sched = Scheduler(ex)
    r = sched.submit([1], SamplingParams(max_new_tokens=4))
    sched.run_until_idle()
    assert len(r.output_ids) == 4 and sched.metrics.decode_tokens == 3


@pytest.mark.parametrize('phase', ['admit', 'prefill', 'decode', 'flush'])
def test_engine_failure_terminates_tickets_and_rejects_new_work(monkeypatch, phase):
    async def scenario():
        ex = tiny_executor(monkeypatch)
        engine = EngineLoop(Scheduler(ex))
        called = False
        def fail(*args):
            nonlocal called
            called = True
            raise RuntimeError('injected failure')
        if phase == 'admit':
            original = ex.admit
            def partial_admit(*args):
                original(*args)  # resources acquired before an unexpected failure
                fail()
            ex.admit = partial_admit
        elif phase == 'prefill':
            ex.prefill_batch = fail
        elif phase == 'decode':
            ex.decode_step = fail
        else:
            ex.flush_cache = fail
        engine.start()
        try:
            if phase == 'flush':
                with pytest.raises(EngineUnavailable):
                    await asyncio.wait_for(engine.flush_cache(), 2)
            else:
                async def client():
                    try:
                        ticket = await engine.submit([1, 2], SamplingParams(max_new_tokens=10))
                    except EngineUnavailable:
                        return
                    while True:
                        _, finished, reason = await ticket.queue.get()
                        if finished:
                            assert reason == 'error'
                            return
                await asyncio.wait_for(asyncio.gather(*(client() for _ in range(8))), 2)
            await asyncio.to_thread(engine.thread.join, 2)
            assert called and not engine.healthy and engine.failure
            assert not engine.tickets and engine.sched.active == 0
            assert len(ex.kv_pool._free) == 1024 and len(ex.state_pool._free) == 15
            with pytest.raises(EngineUnavailable):
                await engine.submit([1], SamplingParams())
            with pytest.raises(EngineUnavailable):
                await engine.flush_cache()
            app = create_app(engine, None, None, {})
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                assert (await client.get('/health')).status_code == 503
                assert (await client.get('/ready')).status_code == 503
        finally:
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('endpoint', ['/generate', '/v1/completions', '/v1/chat/completions'])
def test_http_error_finishes_stream_or_returns_503(monkeypatch, stream, endpoint):
    async def scenario():
        ex = tiny_executor(monkeypatch)
        def fail(*args):
            raise RuntimeError('injected decode failure')
        ex.decode_step = fail
        engine = EngineLoop(Scheduler(ex))
        tok = SimpleNamespace(encode=lambda s: [1, 2], decode=lambda ids, **kw: ''.join(chr(65 + t % 26) for t in ids))
        tmpl = SimpleNamespace(render=lambda *a, **kw: 'prompt')
        engine.start()
        try:
            app = create_app(engine, tok, tmpl, {})
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                response = await asyncio.wait_for(client.post(endpoint, json={'prompt': 'p', 'text': 'p', 'messages': [{'role': 'user', 'content': 'p'}], 'max_tokens': 10, 'stream': stream}), 3)
                if stream:
                    assert response.status_code == 200
                    assert '"type": "server_error"' in response.text and '[DONE]' in response.text
                else:
                    assert response.status_code == 503
        finally:
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())
