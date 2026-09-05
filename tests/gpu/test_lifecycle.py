"""Opt-in GB10 continuation/ownership checks with the pinned real checkpoint.

Run with SHARAR_RUN_GPU=1 and tools/memory-guard.sh watching the test container.
Pools are deliberately bounded so the test does not size itself to all RAM.
"""

import os
from pathlib import Path

import pytest
import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.memory.radix import RadixCache
from sharar.parity.prompts import build_prompts, random_ids
from sharar.tokenizer import SnapshotTokenizer

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 checkpoint tests')


def test_cache_restore_matches_manual_restore_at_every_layer(setup, tmp_path):
    from sharar.parity.cache_trace import run
    configure, _ = setup
    ex, _ = configure(True, 'cold')
    spec = ex.spec
    ex.spec = None
    try:
        ids = SnapshotTokenizer(config.snapshot_dir()).encode('hello ' * 128)
        report = run(ex, ids, tmp_path / 'cache-trace.json')
        assert report['boundary'] == 128
        assert len(report['comparisons']['manual_vs_radix']) == 64
        assert all(row['output']['equal'] for row in report['comparisons']['manual_vs_radix'])
    finally:
        ex.spec = spec


@pytest.fixture(scope='module')
def runtime():
    from sharar.model.qwen38 import Qwen38
    config.check()
    torch.cuda.empty_cache()
    model = Qwen38(config.snapshot_dir(), log=print)
    torch.cuda.empty_cache()
    ex = Executor(model, num_pages=35000, num_slots=12, log=print)
    ex.enable_spec(3, log=print)
    torch.cuda.empty_cache()
    print('before capture:', ex.ledger.describe(), flush=True)
    ex.capture_graphs(log=print)
    torch.cuda.empty_cache()
    print('after capture:', ex.ledger.describe(), flush=True)
    ex.warmup(log=print)
    torch.cuda.empty_cache()
    return ex, build_prompts(config.snapshot_dir())


@pytest.fixture
def setup(runtime):
    ex, prompts = runtime
    spec = ex.spec
    vg, dg = spec.verify_graphs, spec.proposer.graphs
    owned = []

    def configure(graphs, cache='off'):
        ex.use_graphs = graphs
        spec.verify_graphs, spec.proposer.graphs = (vg, dg) if graphs else (None, None)
        ex.cache = RadixCache(ex.kv_pool, ex.state_pool, ex.device) if cache != 'off' else None
        if cache == 'hit':
            ids = prompts['canary_json']
            ex.generate(ids[:len(ids) // 2], SamplingParams(max_new_tokens=1))
        spec.reset_stats()
        ex.ledger.check_floor()
        return ex, prompts

    def admit(ids, params):
        r = ex.admit(ids, params)
        owned.append(r)
        return r

    yield configure, admit
    for r in owned:
        ex.release(r)
    ex.flush_cache()
    spec.verify_graphs, spec.proposer.graphs = vg, dg
    ex.use_graphs = True
    assert len(ex.kv_pool._free) == len(set(ex.kv_pool._free)) == ex.kv_pool.num_pages - 1
    assert len(ex.state_pool._free) == len(set(ex.state_pool._free)) == ex.state_pool.num_slots - 1
    ex.ledger.check_floor()


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('cache', ['off', 'cold', 'hit'])
@pytest.mark.parametrize('stop', ['length', 'eos'])
def test_speculative_stops_commit_exact_snapshots_and_resume(setup, monkeypatch, graphs, batch, cache, stop):
    configure, admit = setup
    ex, prompts = configure(graphs, cache)
    ids = prompts['canary_json']
    reqs = [admit(ids, SamplingParams(max_new_tokens=2 + i if stop == 'length' else 8, ignore_eos=stop == 'length')) for i in range(batch)]
    assert all((r.prefix_len > 0) == (cache == 'hit') for r in reqs)
    for r in reqs:
        ex.prefill(r)
    original = ex.spec._accept
    checks = []

    def checked_accept(rs, y, z, features, t0, t1):
        y_cpu, z_cpu = y.tolist(), z.tolist()
        before = [len(r.output_ids) for r in rs]
        for i, r in enumerate(rs):
            if stop == 'eos':
                a = 0
                while a < ex.spec.k and y_cpu[i][a] == z_cpu[i][a]:
                    a += 1
                candidates = y_cpu[i][:a] + [z_cpu[i][a]]
                r.params.stop_token_ids = (candidates[min(i, len(candidates) - 1)],)
        original(rs, y, z, features, t0, t1)
        for i, r in enumerate(rs):
            count = len(r.output_ids) - before[i]
            assert torch.equal(ex.state_pool.ssm_all[:, r.slot], ex.spec.inter_ssm_all[:, i, count - 1])
            assert torch.equal(ex.state_pool.conv_all[:, r.slot], ex.spec.conv_windows_all[:, i, count - 1])
            assert len(r.pages) == r.committed_len == r.context_len - 1
            assert not set(r.pages.tolist()) & set(ex.kv_pool._free)
            checks.append(count)

    monkeypatch.setattr(ex.spec, '_accept', checked_accept)
    while active := [r for r in reqs if not r.finished]:
        ex.decode_step(active)
    assert checks and ex.spec.emitted == sum(r.spec_emitted for r in reqs) == sum(len(r.output_ids) - 1 for r in reqs)
    monkeypatch.setattr(ex.spec, '_accept', original)
    prefixes = []
    for r in reqs:
        assert r.finish_reason == ('stop' if stop == 'eos' else 'length')
        # Compare a live continuation against the same committed prefix after
        # ownership transfers to the radix cache. Both must produce one token.
        prefix = r.prompt_ids + r.output_ids
        prefixes.append(prefix)
        slot = ex.state_pool.alloc()
        ex.state_pool.copy_slot(r.slot, slot)
        try:
            ex.release(r)
            if ex.cache is not None:
                match = ex.cache.match(prefix[:-1])
                assert match.length == len(prefix) - 1
                assert torch.equal(ex.state_pool.ssm_all[:, match.snapshot_slot], ex.state_pool.ssm_all[:, slot])
                assert torch.equal(ex.state_pool.conv_all[:, match.snapshot_slot], ex.state_pool.conv_all[:, slot])
        finally:
            ex.state_pool.free(slot)
    # Check ownership before generating continuations: a continuation can
    # insert another request's prefix, for which the cache retains its first
    # valid snapshot (prefill vs verify need not round identically).
    for prefix in prefixes:
        continuation = ex.generate(prefix, SamplingParams(max_new_tokens=1))
        if ex.cache is not None:
            assert continuation.prefix_len == len(prefix) - 1
        # Cold teacher-forced continuation: high-margin structured canary.
        saved = ex.cache
        ex.cache = None
        cold = ex.generate(prefix, SamplingParams(max_new_tokens=1))
        ex.cache = saved
        assert continuation.output_ids == cold.output_ids


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 3, 4])
def test_flush_with_active_shared_prefix_and_continuation(setup, graphs, batch):
    configure, admit = setup
    ex, prompts = configure(graphs, 'hit')
    ids = prompts['canary_json']
    reqs = [admit(ids, SamplingParams(max_new_tokens=8, ignore_eos=True)) for _ in range(batch)]
    for r in reqs:
        ex.prefill(r)
    pinned = set(reqs[0].pages[:reqs[0].prefix_len].tolist())
    assert pinned
    ex.flush_cache()
    assert ex.cache.flushing and not pinned & set(ex.kv_pool._free)
    assert ex.cache.match(ids).length == 0
    fresh_pages = ex.kv_pool.alloc(64)
    assert not pinned & set(fresh_pages.tolist())
    ex.kv_pool.free(fresh_pages)
    while active := [r for r in reqs if not r.finished]:
        ex.decode_step(active)
    outputs = [r.output_ids[:] for r in reqs]
    for i, r in enumerate(reqs):
        ex.release(r)
        assert ex.cache.flushing == (i != len(reqs) - 1)
    assert pinned <= set(ex.kv_pool._free)
    cold = ex.generate_batch([(ids, SamplingParams(max_new_tokens=8, ignore_eos=True)) for _ in reqs])
    assert outputs == [r.output_ids for r in cold]


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 3, 4])
@pytest.mark.parametrize('cache', ['off', 'cold', 'hit'])
def test_retract_after_streaming_preserves_history_and_continues(setup, graphs, batch, cache):
    configure, admit = setup
    ex, prompts = configure(graphs, cache)
    ids = prompts['canary_json']
    reqs = [admit(ids, SamplingParams(max_new_tokens=8, ignore_eos=True)) for _ in range(batch)]
    for r in reqs:
        ex.prefill(r)
    ex.decode_step(reqs)
    delivered = [r.new_tokens() for r in reqs]
    for r in reqs:
        ex.retract(r)
        assert r.new_tokens() == []
        ex.admit(r.prompt_ids, r.params, r)
    ex.prefill_batch([(r, r.prefill_len - r.prefill_pos) for r in reqs])
    while active := [r for r in reqs if not r.finished]:
        ex.decode_step(active)
    for i, r in enumerate(reqs):
        delivered[i] += r.new_tokens()
        assert delivered[i] == r.output_ids and len(r.output_ids) == 8
        assert r.committed_len == r.context_len - 1
    outputs = [r.output_ids[:] for r in reqs]
    for r in reqs:
        ex.release(r)
    saved = ex.cache
    ex.cache = None
    baseline = ex.generate_batch([(ids, SamplingParams(max_new_tokens=8, ignore_eos=True)) for _ in reqs])
    ex.cache = saved
    assert outputs == [r.output_ids for r in baseline]


@pytest.mark.parametrize('graphs', [False, True])
def test_scheduler_short_budgets_and_profile_boundary(setup, graphs):
    configure, _ = setup
    ex, prompts = configure(graphs, 'cold')
    sched = Scheduler(ex)
    rs = [sched.submit(prompts['canary_json'], SamplingParams(max_new_tokens=n, ignore_eos=True)) for n in range(1, 6)]
    sched.run_until_idle()
    assert [len(r.output_ids) for r in rs] == list(range(1, 6))
    assert all(r.released for r in rs)
    assert sched.metrics.decode_tokens == sum(len(r.output_ids) - 1 for r in rs)
    ids = random_ids(config.CONTEXT_LENGTH - 2, 700)
    with pytest.raises(InvalidRequest):
        sched.submit(ids, SamplingParams(max_new_tokens=3))
    r = sched.submit(ids, SamplingParams(max_new_tokens=2, ignore_eos=True))
    sched.run_until_idle()
    assert r.released and r.finish_reason == 'length' and len(r.output_ids) == 2


@pytest.mark.parametrize('graphs', [False, True])
def test_frozen_canaries(setup, graphs):
    import json
    from sharar.engine.run import _canary_checks, _compare_continuation

    configure, _ = setup
    ex, prompts = configure(graphs)
    tok = SnapshotTokenizer(config.snapshot_dir())
    reference = Path(os.environ['SHARAR_RESULTS']) / 'parity/reference/http'
    for name, check in _canary_checks().items():
        r = ex.generate(prompts[name], SamplingParams(max_new_tokens=64))
        text = tok.decode(r.output_ids, skip_special_tokens=True)
        assert check(text), (name, text)
        frozen = json.loads((reference / f'{name}.json').read_text())
        _, ok = _compare_continuation(name, r.output_ids, frozen)
        assert ok


@pytest.mark.parametrize('graphs', [False, True])
@pytest.mark.parametrize('features', [False, True])
def test_ngram_real_target_cache_and_feature_lifetimes(runtime, graphs, features):
    from sharar.engine.contracts import FeatureRequest, ProposerContext, ProposerRequirements
    from sharar.engine.spec import SpecDecoder
    from sharar.proposers.ngram import NGramProposer

    ex, prompts = runtime
    previous, old_graphs = ex.spec, ex.use_graphs
    observed = []
    class ObservedNgram(NGramProposer):
        def accepted(self, req, emitted, result, row):
            if features:
                assert result.norm.shape[1:] == (4, config.HIDDEN_SIZE)
                assert set(result.layers) == {0, 63}
                for tensor in (result.norm, *result.layers.values()):
                    assert tensor.dtype == torch.bfloat16
                    observed.append((tensor, tensor.clone()))
            else:
                assert result.norm is None and result.layers is None
                observed.append(None)
    prop = ObservedNgram(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3, max_ngram=1)
    if features:
        prop.requirements = ProposerRequirements('observed-ngram', FeatureRequest(True, (0, 63)))
    # The frozen structured canary has a stable continuation across cold and
    # paged prefill. Its punctuation/keys also offer real suffix matches.
    ids = prompts['canary_json']
    params = SamplingParams(max_new_tokens=16, ignore_eos=True)
    try:
        ex.flush_cache()
        ex.cache = RadixCache(ex.kv_pool, ex.state_pool, ex.device)
        ex.spec = None
        ex.use_graphs = graphs
        baseline = ex.generate(ids, params).output_ids
        ex.flush_cache()
        ex.spec = SpecDecoder(ex, prop)
        if graphs:
            ex.spec.capture_graphs(log=print)
        # Both a fresh prefill and a hit must supply valid proposer inputs.
        cold = ex.generate(ids, params)
        ex.flush_cache()
        ex.generate(ids[:len(ids) // 2], SamplingParams(max_new_tokens=1))
        warm = ex.generate(ids, params)
        assert cold.output_ids == warm.output_ids == baseline, (cold.output_ids, warm.output_ids, baseline)
        assert warm.prefix_len > 0 and ex.spec.rounds > 1
        assert observed
        if features:
            # Replay has happened several times: retained callbacks must not
            # alias graph output storage or later in-place residual updates.
            assert all(torch.equal(tensor, snapshot) for tensor, snapshot in observed)
        else:
            assert prop.requirements.persistent_bytes == 0
            if graphs:
                assert all(st['features'] is None for st in ex.spec.verify_graphs._st.values())
    finally:
        ex.flush_cache()
        ex.spec, ex.use_graphs = previous, old_graphs
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
    ex.ledger.check_floor()


def test_selected_features_are_post_block_residual_streams(runtime):
    from sharar.engine.context import scratch_prefill_context
    from sharar.engine.contracts import FeatureRequest

    ex, prompts = runtime
    ids = torch.tensor(prompts['rand8'], device=ex.device)
    ctx = scratch_prefill_context(len(ids), ex.device)
    ctx.feature_request = FeatureRequest(layers=(0, 31, 63))
    out = ex.model.forward(ids, ctx, capture=True)
    assert set(out['features']) == {0, 31, 63}
    for index, feature in out['features'].items():
        hidden, residual = out['layers'][index]
        assert torch.equal(feature, hidden + residual)
    expected = {i: value.clone() for i, value in out['features'].items()}
    fresh = ex.model.forward(ids, scratch_prefill_context(len(ids), ex.device))
    assert 'features' not in fresh
    assert all(torch.equal(value, expected[i]) for i, value in out['features'].items())
