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


@torch.inference_mode()
def test_reordered_checkpoints_preserve_core_and_exact_snapshots():
    """Capacity-prioritized destinations must equal the ascending kernel plan."""
    from types import SimpleNamespace
    from sharar import config
    from sharar.engine.context import ForwardContext
    from sharar.model.gdn import GatedDeltaNet
    from sharar.memory.state_pool import CONV_DIM, CONV_STATE_LEN

    torch.manual_seed(72)
    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    layer.layer_id = 0
    layer.num_k_heads, layer.num_v_heads = config.LINEAR_NUM_KEY_HEADS, config.LINEAR_NUM_VALUE_HEADS
    layer.head_k_dim, layer.head_v_dim = config.LINEAR_KEY_HEAD_DIM, config.LINEAR_VALUE_HEAD_DIM
    layer.key_dim, layer.value_dim = layer.num_k_heads * layer.head_k_dim, layer.num_v_heads * layer.head_v_dim
    layer.conv_weight = torch.randn(CONV_DIM, 4, dtype=torch.bfloat16, device='cuda') * .1
    layer.A_log = torch.zeros(layer.num_v_heads, device='cuda')
    layer.dt_bias = torch.zeros(layer.num_v_heads, dtype=torch.bfloat16, device='cuda')
    lengths, slots = [257, 193], [1, 2]
    t = sum(lengths)
    qkv = torch.randn(t, CONV_DIM, dtype=torch.bfloat16, device='cuda') * .1
    ba = [torch.randn(t, layer.num_v_heads, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    conv = torch.zeros(9, CONV_DIM, CONV_STATE_LEN, dtype=torch.bfloat16, device='cuda')
    ssm = torch.zeros(9, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim, device='cuda')
    pool = SimpleNamespace(conv={0: conv}, ssm={0: ssm})
    def context(positions, destinations, rows):
        return ForwardContext(mode='prefill', kv_pool=None, state_pool=pool,
                              positions=torch.arange(t, device='cuda'),
                              cu_seqlens=torch.tensor([0, 257, t], dtype=torch.int32, device='cuda'),
                              seq_slots=torch.tensor(slots, dtype=torch.int32, device='cuda'),
                              seq_lens=lengths, seq_prefix_lens=[0, 0],
                              checkpoint_positions=positions, checkpoint_slots=destinations, checkpoint_seqs=rows)
    expected = layer._prefill_core(qkv, *ba, context([64, 256, 64, 192], [3, 4, 5, 6], [0, 0, 1, 1])).clone()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    ssm.zero_(); conv.zero_()
    actual = layer._prefill_core(qkv, *ba, context([256, 192, 64, 64], [7, 8, 3, 5], [0, 1, 0, 1]))
    assert torch.equal(expected, actual)
    for old, new in ((1, 1), (2, 2), (3, 3), (4, 7), (5, 5), (6, 8)):
        assert torch.equal(old_ssm[old], ssm[new])
        assert torch.equal(old_conv[old], conv[new])
    assert not ssm[0].count_nonzero() and not conv[0].count_nonzero()
