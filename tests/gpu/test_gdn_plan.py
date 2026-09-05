"""Prepared metadata must preserve exact chunk output and recurrent state."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 kernels')


@pytest.mark.parametrize('lengths', [[1], [3], [16], [17], [63], [64], [65], [129], [1, 3], [63, 65, 129]])
@torch.inference_mode()
def test_prepared_chunk_metadata_is_bit_exact(lengths, monkeypatch):
    from sharar.engine.context import ForwardContext
    from sharar.model.gdn_plan import prepare_gdn_prefill
    from sharar.kernels.fla.chunk import chunk_gated_delta_rule
    from sharar.kernels.fla.index import prepare_chunk_indices, prepare_chunk_offsets
    from sharar.kernels.fla import chunk, chunk_delta_h, chunk_o, chunk_fwd, cumsum
    from itertools import accumulate
    torch.manual_seed(913)
    t, n = sum(lengths), len(lengths)
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device='cuda')
    slots = torch.arange(n, 0, -1, dtype=torch.int32, device='cuda')
    ctx = ForwardContext(mode='prefill', kv_pool=None, state_pool=None,
                         positions=torch.arange(t, device='cuda'), cu_seqlens=cu, seq_slots=slots,
                         seq_lens=lengths, seq_prefix_lens=[64] * n)
    plan = prepare_gdn_prefill(ctx, t, 'cuda')
    assert torch.equal(plan.chunk_indices, prepare_chunk_indices(cu, 64))
    assert torch.equal(plan.chunk_offsets, prepare_chunk_offsets(cu, 64))
    q, k = [torch.randn(1, t, 2, 128, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    v = torch.randn(1, t, 4, 128, dtype=torch.bfloat16, device='cuda')
    g = -torch.rand(1, t, 4, dtype=torch.float32, device='cuda')
    beta = torch.rand(1, t, 4, dtype=torch.bfloat16, device='cuda')
    initial = torch.randn(n + 1, 4, 128, 128, dtype=torch.float32, device='cuda')
    old, new = initial.clone(), initial.clone()
    args = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state_indices=slots,
                cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    expected, _, expected_h = chunk_gated_delta_rule(**args, initial_state=old)
    def forbidden(*args, **kwargs):
        raise AssertionError('a layer rebuilt prefill metadata')
    for module in (chunk, chunk_delta_h, chunk_o, chunk_fwd, cumsum):
        monkeypatch.setattr(module, 'prepare_chunk_indices', forbidden)
    monkeypatch.setattr(chunk_delta_h, 'prepare_chunk_offsets', forbidden)
    actual, _, actual_h = chunk_gated_delta_rule(**args, initial_state=new,
                        chunk_indices=plan.chunk_indices, chunk_offsets=plan.chunk_offsets)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_h, actual_h)
    assert torch.equal(old, new)
    assert torch.equal(initial[0], new[0])  # unowned state was untouched
