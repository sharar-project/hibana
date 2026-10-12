import torch
from types import SimpleNamespace

from sharar.model.parallel_draft import convolve, ParallelDraftModel
from sharar.proposers.parallel import FeatureCache
from sharar.cache.spec_budget import verify_bytes


def test_convolution_resets_at_each_block_and_uses_both_dynamic_taps():
    torch.manual_seed(12)
    x = torch.randn(2, 4, 32)
    delta = torch.randn(2, 4, 2, 2)
    base = torch.randn(2, 32)
    expected = torch.zeros_like(x)
    for b in range(2):
        for t in range(4):
            for c in range(32):
                for tap in range(2):
                    if t >= tap:
                        expected[b, t, c] += (base[tap, c] + delta[b, t, tap, c // 16]) * x[b, t - tap, c]
    torch.testing.assert_close(convolve(x, delta, base), expected, rtol=1e-6, atol=1e-6)


def test_feature_cache_bounds_memory_and_invalidates_evicted_handles():
    cache = FeatureCache(64)
    a = cache.put(0, 4, torch.ones(4, 2))
    b = cache.put(4, 8, torch.ones(4, 2) * 2)
    assert cache.bytes == 64
    assert cache.get(a) is not None  # a becomes most recent
    c = cache.put(8, 12, torch.ones(4, 2) * 3)
    assert cache.bytes == 64 and cache.get(b) is None
    assert cache.get(a)[1] == 4 and cache.get(c)[1] == 12
    assert cache.put(0, 100, torch.ones(100)) is None


def test_reconstruction_budget_includes_all_retained_inputs_and_conv():
    full = verify_bytes(4, 4, 'snapshots')
    compact = verify_bytes(4, 4, 'reconstruct')
    assert full == 2.25 * 2**30 + 45 * 2**20
    assert compact == 45 * 2**20 + 48 * 4 * 4 * ((16 * 128 + 48 * 128) * 2 + 48 * 2 * 4)
    assert compact < .03 * full


def test_selector_lattice_conditions_on_anchor_then_previous_candidate():
    torch.manual_seed(71)
    hidden = torch.randn(2, 3, 5120)
    logits = torch.randn(6, 32)
    weights = {'candidate_selector.hidden_projection.weight': torch.randn(4, 5120),
               'candidate_selector.predecessor_codebook': torch.randn(32, 4),
               'candidate_selector.successor_codebook': torch.randn(32, 4)}
    from sharar.proposers.draft_head import TargetHead
    model = SimpleNamespace(target=SimpleNamespace(logits=lambda x: logits), draft_head=TargetHead(lambda x: logits), weights=weights,
                            cfg={'dflash_config': {'selector_top_k': 4}}, hidden=hidden.shape[-1])
    anchor = torch.tensor([2, 9])
    candidates, scores = ParallelDraftModel.selector(model, hidden, anchor)
    gate = hidden @ weights['candidate_selector.hidden_projection.weight'].T
    for b in range(2):
        for t in range(3):
            for i in range(4):
                previous = anchor[b] if t == 0 else candidates[b, t - 1, i]
                for j in range(4):
                    token = candidates[b, t, j]
                    pair = (weights['candidate_selector.predecessor_codebook'][previous]
                            * gate[b, t] * weights['candidate_selector.successor_codebook'][token]).sum()
                    torch.testing.assert_close(scores[b, t, i, j], logits[b * 3 + t, token] + pair, rtol=1e-5, atol=1e-4)


def test_prospective_draft_budget_reduces_optional_cache_before_allocation(monkeypatch):
    from sharar.cache import ledger
    monkeypatch.setattr(ledger, 'mem_available_bytes', lambda: 64 * 2**30)
    book = ledger.Ledger()
    control = book.plan(state_slot_target=1000)
    draft = book.plan(future_bytes=10 * 2**30, extra_page_bytes=4096, state_slot_target=1000)
    assert draft.kv_tokens < control.kv_tokens and draft.state_slots < control.state_slots
    assert draft.reserved_bytes == control.reserved_bytes + 10 * 2**30
