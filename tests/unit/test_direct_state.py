"""StateManager authorizes the entire batch before any in-place recurrence."""
from types import SimpleNamespace

import pytest
import torch

from hibana.engine.sampler import SamplingParams
from tests.unit.test_runtime_safety import tiny_executor


def setup(monkeypatch, batch=4):
    ex = tiny_executor(monkeypatch, cache=True)
    ex.state_pool.num_slots = 16
    reqs = [ex.admit([1, row + 2, 3], SamplingParams()) for row in range(batch)]
    for req in reqs:
        ex.prefill(req)
    # Nonordered private destinations, distinct source markers, mixed counts.
    reqs = list(reversed(reqs))
    counts = [1, 4, 2, 3][:batch]
    ex.state.reserve(reqs, 4)
    for req in reqs:
        ex.state.step_pages(req)
    windows = torch.arange(batch * 4).reshape(1, batch, 4, 1).float() + 100
    calls = []
    def run(pool, slots, lengths):
        calls.append((slots.tolist(), lengths.tolist()))
        assert all(r.reservation is not None and r.committed_len == 3 for r in reqs)
        for row, (slot, count) in enumerate(zip(slots, lengths)):
            pool.ssm_all[:, slot] += row * 10 + count
    replay = SimpleNamespace(batch=4, steps=4, run=run)
    return ex, reqs, counts, windows, replay, calls


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
def test_direct_commit_keeps_exact_pages_and_private_state(monkeypatch, batch):
    ex, reqs, counts, windows, replay, calls = setup(monkeypatch, batch)
    before = ex.state_pool.ssm_all.clone()
    expected = before.clone()
    retained = [torch.cat([r.pages, r.reservation.pages[:n]]) for r, n in zip(reqs, counts)]
    tails = [r.reservation.ids[n:] for r, n in zip(reqs, counts)]
    for row, (req, count) in enumerate(zip(reqs, counts)):
        expected[:, req.slot] += row * 10 + count
    ex.state.commit_reconstructed(reqs, counts, replay, windows, verified_steps=4)
    assert calls == [([r.slot for r in reqs], counts)]
    assert torch.equal(ex.state_pool.ssm_all, expected)
    for row, (req, count) in enumerate(zip(reqs, counts)):
        assert req.reservation is None and req.committed_len == 3 + count
        assert req.cacheable and torch.equal(req.pages, retained[row])
        assert torch.equal(ex.state_pool.conv_all[:, req.slot], windows[:, row, count - 1])
        assert set(tails[row]) <= set(ex.kv_pool._free)
        assert not set(req.pages.tolist()) & set(ex.kv_pool._free)
        req.cacheable = False  # no synthetic emitted history for cache insertion
        ex.release(req)
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


@pytest.mark.parametrize('bad', ['zero', 'oversize', 'float', 'last_row', 'duplicate_slot', 'dummy',
                                'outside_pool', 'released', 'no_reservation', 'prefix', 'short_conv', 'replay_width', 'unverified_tail'])
def test_invalid_row_prevents_every_write_and_publication(monkeypatch, bad):
    ex, reqs, counts, windows, replay, calls = setup(monkeypatch)
    if bad == 'zero': counts[0] = 0
    elif bad == 'oversize': counts[0] = 5
    elif bad == 'float': counts[0] = 1.0
    elif bad == 'last_row': counts[-1] = 5
    elif bad == 'duplicate_slot': reqs[-1].slot = reqs[0].slot
    elif bad == 'dummy': reqs[-1].slot = 0
    elif bad == 'outside_pool': reqs[-1].slot = 16
    elif bad == 'released': reqs[-1].released = True
    elif bad == 'no_reservation': reqs[-1].reservation = None
    elif bad == 'prefix': reqs[-1].committed_len -= 1
    elif bad == 'short_conv': windows = windows[:, :, :2]
    elif bad == 'replay_width': replay.steps = 2
    before = ex.state_pool.ssm_all.clone(), ex.state_pool.conv_all.clone()
    ownership = [(r.pages, r.committed_len, r.reservation) for r in reqs]
    free = ex.kv_pool._free[:]
    with pytest.raises(ValueError):
        ex.state.commit_reconstructed(reqs, counts, replay, windows, verified_steps=2 if bad == 'unverified_tail' else 4)
    assert not calls and ex.kv_pool._free == free
    assert all(torch.equal(a, b) for a, b in zip(before, (ex.state_pool.ssm_all, ex.state_pool.conv_all)))
    assert all(r.pages is pages and r.committed_len == count and r.reservation is reservation
               for r, (pages, count, reservation) in zip(reqs, ownership))


@pytest.mark.parametrize('failure', [RuntimeError, MemoryError])
def test_partial_write_failure_cannot_retry_or_enter_prefix_cache(monkeypatch, failure):
    ex, reqs, counts, windows, replay, _ = setup(monkeypatch)
    def fail(pool, slots, counts):
        pool.ssm_all[:, slots[0]] = -999
        raise failure('injected post-write failure')
    replay.run = fail
    with pytest.raises(RuntimeError):
        ex.state.commit_reconstructed(reqs, counts, replay, windows, verified_steps=4)
    assert all(not r.cacheable and r.reservation is not None and r.committed_len == 3 for r in reqs)
    for req in reqs:
        ex.state.rollback(req)
        ex.release(req)
        ex.release(req)
    assert ex.cache.cached_tokens == 0
    assert ex.kv_pool.free_count == len(set(ex.kv_pool._free)) == 1024
    assert ex.state_pool.free_count == len(set(ex.state_pool._free)) == 15
