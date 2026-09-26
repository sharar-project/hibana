"""Behavioral checks for the state/proposer boundaries, with no GPU backend."""

import subprocess
import sys

import pytest
import torch

from sharar.engine.contracts import FeatureRequest, ProposerContext, ProposerRequirements
from sharar.engine.profile import ServingProfile
from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.proposers.ngram import NGramProposer
from tests.unit.test_runtime_safety import tiny_executor


@pytest.mark.parametrize('cache', [False, True])
@pytest.mark.parametrize('budget', [1, 2, 3, 4, 5, 10])
def test_feature_free_proposer_uses_real_verify_commit_and_scheduler(monkeypatch, cache, budget):
    ex = tiny_executor(monkeypatch, cache)
    def logits(norm):
        out = torch.zeros((len(norm), 128))
        out[:, 7] = 5
        return out
    ex.model.logits = logits
    params = SamplingParams(max_new_tokens=budget)
    expected = ex.generate([7] * 8, params).output_ids
    calls = []
    class CheckedNgram(NGramProposer):
        def accepted_batch(self, requests, emitted, features):
            assert features.norm is None and features.layers is None
            assert all(r.reservation is None and r.committed_len == r.context_len - 1 for r in requests)
            calls.extend(map(len, emitted))
    proposer = CheckedNgram(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3)
    ex.set_proposer(proposer)
    sched = Scheduler(ex)
    requests = [sched.submit([7] * 8, params) for _ in range(4)]
    sched.run_until_idle()
    assert all(r.output_ids == expected and r.released for r in requests)
    assert sum(calls) == 4 * (budget - 1)
    assert ex.spec.emitted == sched.metrics.decode_tokens == sum(calls)
    assert not hasattr(ex.spec, 'mtp') and not hasattr(ex.spec, 'draft_graphs')
    ex.flush_cache()
    assert len(ex.kv_pool._free) == 1024 and len(ex.state_pool._free) == 15


def test_ngram_match_and_window_are_explicit():
    proposer = NGramProposer(ProposerContext(torch.device('cpu'), 4, 512), k=2, max_ngram=2, window=8)
    req = Request([1, 2, 3, 4, 1, 2], SamplingParams())
    assert proposer.candidates(req) == [3, 4]
    req.prompt_ids = [1, 2, 3, 4] + [99] * 8 + [1, 2]
    assert proposer.candidates(req) is None


def test_ngram_fallback_keeps_sampled_rng_semantics(monkeypatch):
    ex = tiny_executor(monkeypatch)
    params = SamplingParams(max_new_tokens=10, temperature=1, seed=14)
    expected = ex.generate([1, 2, 3], params).output_ids
    proposer = NGramProposer(ProposerContext(ex.device, 4, 512))
    def forbidden(*args, **kwargs):
        raise AssertionError('sampled requests must use ordinary decode')
    proposer.propose = forbidden
    ex.set_proposer(proposer)
    assert ex.generate([1, 2, 3], params).output_ids == expected


def test_reservation_rollback_keeps_committed_state_and_reclaims_pages(monkeypatch):
    ex = tiny_executor(monkeypatch)
    req = ex.admit([1, 2, 3], SamplingParams())
    ex.prefill(req)
    state = ex.state_pool.ssm_all[:, req.slot].clone()
    pages = req.pages.clone()
    free = ex.kv_pool.free_count
    reserved = ex.state.reserve([req], 4)
    assert ex.state.step_pages(req).shape[0] == len(pages) + 4
    with pytest.raises(ValueError, match='overlapping'):
        ex.state.reserve([req], 1)
    ex.state.rollback(req)
    ex.state.rollback(req)
    assert torch.equal(ex.state_pool.ssm_all[:, req.slot], state)
    assert torch.equal(req.pages, pages) and req.committed_len == 3
    assert ex.kv_pool.free_count == free
    assert set(reserved.tolist()) <= set(ex.kv_pool._free)
    ex.release(req)


def test_admission_failure_returns_slot_and_cache_loan(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    ex.generate([1, 2], SamplingParams(max_new_tokens=1))
    before = ex.state_pool.free_count
    def fail(n):
        raise MemoryError('injected allocation failure')
    monkeypatch.setattr(ex.kv_pool, 'alloc', fail)
    req = Request([1, 2, 3], SamplingParams())
    with pytest.raises(MemoryError):
        ex.admit(req.prompt_ids, req.params, req)
    assert req.slot == -1 and req.pages is None and req.match is None
    assert ex.state_pool.free_count == before
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_release_runs_adapter_cleanup_once_and_abandoned_state_is_not_cached(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    cleaned = []
    prop = NGramProposer(ProposerContext(ex.device, 4, 512))
    prop.release = lambda req: cleaned.append(req)
    ex.set_proposer(prop)
    req = ex.admit([1, 2, 3], SamplingParams())
    ex.prefill(req)
    ex.state.reserve([req], 1)
    # Simulate a partially advanced in-place forward, then abandon its pages.
    ex.state_pool.ssm_all[:, req.slot].fill_(99)
    ex.state.rollback(req)
    ex.release(req)
    ex.release(req)
    assert cleaned == [req]
    assert ex.cache.match([1, 2, 3]).length == 0
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_feature_and_context_requirements_are_checked_before_buffers(monkeypatch):
    ex = tiny_executor(monkeypatch)
    prop = NGramProposer(ProposerContext(ex.device, 4, 512))
    prop.requirements = ProposerRequirements('bad', FeatureRequest(layers=(64,)))
    with pytest.raises(ValueError, match='feature layer'):
        ex.set_proposer(prop)
    assert ex.spec is None
    prop.requirements = NGramProposer.requirements
    prop.context = ProposerContext(ex.device, 4, 1024)
    with pytest.raises(ValueError, match='context'):
        ex.set_proposer(prop)


def test_profile_limits_and_load_budget(monkeypatch):
    from sharar.cache import ledger
    profile = ServingProfile(max_batch=3, max_context=512)
    assert profile.graph_sizes == (1, 2, 3)
    # Startup validates the proposer's width; the serving profile owns only
    # context, batch and scheduling limits.
    from sharar.cache.spec_budget import SpecConfig
    assert SpecConfig('dflash2', 15).depth == 15
    for name, depth in [('dflash2', 16), ('dspark', 8), ('mtp', 9)]:
        with pytest.raises(ValueError, match='proposer/depth'):
            SpecConfig(name, depth)
    monkeypatch.setattr(ledger, 'mem_available_bytes', lambda: 63 * 2**30)
    with pytest.raises(MemoryError, match='model load'):
        ledger.check_load_headroom(48 * 2**30)
    monkeypatch.setattr(ledger, 'mem_available_bytes', lambda: 64 * 2**30)
    ledger.check_load_headroom(48 * 2**30)


def test_control_modules_import_without_gpu_backends():
    code = '''
import builtins, importlib.abc, sys, torch
# CUDA PyTorch probes Triton during its own initialization. Check Sharar's
# imports after that bootstrap, including imports already in sys.modules.
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'flashinfer', 'triton', 'sglang'}:
        raise AssertionError(f'CPU control path imported {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
class NoGpuBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'flashinfer', 'triton', 'sglang'}:
            raise AssertionError(f'CPU control path imported {fullname}')
sys.meta_path.insert(0, NoGpuBackend())
from sharar.engine.executor import Executor
from sharar.engine.spec import SpecDecoder
from sharar.engine.scheduler import Scheduler
from sharar.proposers.ngram import NGramProposer
from sharar.api.server import EngineLoop
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('budget', [2, 4, 8])
def test_short_verify_releases_entire_unverified_reservation(monkeypatch, depth, budget):
    ex = tiny_executor(monkeypatch)
    expected = ex.generate([7, 7, 7], SamplingParams(max_new_tokens=budget)).output_ids
    ex.set_proposer(NGramProposer(ProposerContext(ex.device, 4, 512), k=3))
    ex.spec.verify_depths = (1, 3)
    ex.spec.verify_depth = depth
    requests = ex.generate_batch([([7, 7, 7], SamplingParams(max_new_tokens=budget)) for _ in range(4)])
    assert all(r.output_ids == expected and r.committed_len == r.context_len - 1 for r in requests)
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15
