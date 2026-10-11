from types import SimpleNamespace

import pytest
import torch

from sharar.engine.context import ForwardContext
from sharar.model.gdn_plan import prepare_gdn_prefill


def context(**kwargs):
    return ForwardContext(mode='prefill', kv_pool=None, state_pool=None,
                          positions=torch.empty(0, dtype=torch.int64), **kwargs)


def test_single_plan_preserves_cold_attention_and_private_slot(monkeypatch):
    ctx = context(seq_len=129, slot=9, checkpoint_positions=[64, 128], checkpoint_slots=[2, 3])
    # Metadata already lives on the CPU; no tensor reads are needed to plan.
    def forbidden(*args, **kwargs):
        raise AssertionError('planning read a tensor')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'tolist', forbidden)
        plan = prepare_gdn_prefill(ctx, 129, 'cpu')
        assert prepare_gdn_prefill(ctx, 129, 'cpu') is plan
    assert not ctx.batched and ctx.cu_seqlens is None
    assert plan.cache_indices.tolist() == [0]  # standalone parity owns slot zero
    assert plan.query_start_loc.tolist() == [0, 129]
    assert plan.chunk_indices.tolist() == [[0, 0], [0, 1], [0, 2]]
    assert plan.chunk_offsets.tolist() == [0, 3]
    assert plan.checkpoints == ((2, 64, 1), (3, 128, 2))


def test_uneven_batched_plan_uses_cpu_lengths_and_preserves_slot_order(monkeypatch):
    lengths = [1, 3, 63, 64, 65, 128, 2048]
    cu = torch.tensor([0, 1, 4, 67, 131, 196, 324, 2372], dtype=torch.int32)
    slots = torch.tensor([7, 1, 6, 2, 5, 3, 4], dtype=torch.int32)
    ctx = context(cu_seqlens=cu, seq_slots=slots, seq_lens=lengths,
                  seq_prefix_lens=[0, 64, 128, 0, 64, 64, 0],
                  checkpoint_positions=[128, 128, 1984], checkpoint_slots=[8, 9, 10],
                  checkpoint_seqs=[4, 5, 6])
    def forbidden(*args, **kwargs):
        raise AssertionError('planning read a tensor')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'tolist', forbidden)
        plan = prepare_gdn_prefill(ctx, sum(lengths), 'cpu')
    assert plan.query_start_loc is cu and plan.cache_indices is slots
    assert plan.chunk_offsets.dtype == torch.int64
    assert plan.chunk_offsets.tolist() == [0, 1, 2, 3, 4, 6, 8, 40]
    assert plan.has_initial_state.tolist() == [False, True, True, False, True, True, False]
    assert plan.checkpoints == ((8, 195, 5), (9, 260, 7), (10, 2308, 39))
    assert ctx.batched


def test_cached_single_keeps_its_committed_slot():
    ctx = context(slot=5, prefix_len=128, seq_len=65)
    ctx.state_pool = SimpleNamespace()
    plan = prepare_gdn_prefill(ctx, 65, 'cpu')
    assert plan.cache_indices.tolist() == [5]
    assert plan.has_initial_state.tolist() == [True]
    assert not ctx.batched


def test_cute_checkpoint_plan_reuses_cpu_metadata():
    ctx = context(seq_len=257, checkpoint_positions=[192, 64], checkpoint_slots=[3, 2])
    plan = prepare_gdn_prefill(ctx, 257, 'cpu')
    assert plan.cute_cu_seqlens.dtype == torch.int64
    assert plan.cute_cu_seqlens.tolist() == [0, 257]
    assert plan.checkpoint_interval == 64
    assert plan.checkpoint_count == 2
    assert plan.checkpoint_indices.tolist() == [1, 0]
    assert plan.checkpoint_cu_starts.tolist() == [0, 4, 0, -1, 1, -1]
    assert plan.checkpoint_destinations.tolist() == [3, 2]


def test_sparse_checkpoint_plan_uses_largest_interval():
    ctx = context(seq_len=8192, checkpoint_positions=[6144, 2048], checkpoint_slots=[3, 2])
    plan = prepare_gdn_prefill(ctx, 8192, 'cpu')
    assert plan.checkpoint_interval == 2048
    assert plan.checkpoint_count == 2
    assert plan.checkpoint_indices.tolist() == [1, 0]


@pytest.mark.parametrize('position', [0, 1, 65, 128])
def test_checkpoint_must_be_interior_chunk_boundary(position):
    ctx = context(seq_len=128, checkpoint_positions=[position], checkpoint_slots=[1])
    with pytest.raises(ValueError, match='interior'):
        prepare_gdn_prefill(ctx, 128, 'cpu')
