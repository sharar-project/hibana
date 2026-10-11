"""Merge routing regression; importing the model requires the GPU runtime.

The kernels are mocked, but the real GDN module imports FlashInfer and Triton.
Keep this in the gated GPU suite, not dependency-light CPU unit CI.
"""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("quantized", [False, True])
def test_prefill_routes_fp32_and_quantized_pool_states(monkeypatch, quantized):
    import sharar.model.gdn as module

    layer = module.GatedDeltaNet.__new__(module.GatedDeltaNet)
    layer.layer_id = 0
    layer.num_k_heads, layer.num_v_heads = 2, 6
    layer.head_k_dim = layer.head_v_dim = 4
    layer.key_dim, layer.value_dim = 8, 24
    layer.conv_dim = 40
    layer.conv_weight = torch.empty(40, 4)
    layer.A_log = layer.dt_bias = torch.empty(6)
    state = torch.zeros(9, 6, 4, 4)
    conv = torch.zeros(9, 40, 3)
    packed = torch.zeros(2, 6, 4, 4)
    history = torch.full((1, 3, 6, 4, 4), 7.)
    result = torch.empty(1, 130, 6, 4)
    slots = torch.tensor([3, 1], dtype=torch.int32)
    plan = SimpleNamespace(cache_indices=slots, checkpoints=((7, 64, 1),),
                           query_start_loc=torch.tensor([0, 65, 130], dtype=torch.int32),
                           chunk_indices=torch.empty(4, 2, dtype=torch.int32),
                           chunk_offsets=torch.tensor([0, 2, 4]),
                           has_initial_state=torch.tensor([True, True]), seq_lens=(65, 65))
    views = object()
    window = SimpleNamespace(layout=object(), layer={0: views}) if quantized else None
    ctx = SimpleNamespace(state_pool=SimpleNamespace(window=window))
    layer._states = lambda ctx: (conv, state)
    monkeypatch.setattr(module, "prepare_gdn_prefill", lambda *a: plan)
    monkeypatch.setattr(module, "causal_conv1d_fn", lambda x, *a, **kw: x)
    monkeypatch.setattr(module, "fused_gdn_gating", lambda *a: (torch.empty(1, 130, 6), torch.empty(1, 130, 6)))
    calls, commits = [], []
    def materialize(layout, src, indices):
        assert quantized and src is views and indices is slots
        return packed
    def dispatcher(**kw):
        assert not quantized and kw['state'] is state and kw['plan'] is plan
        calls.append('dispatcher')
        return result
    def triton(**kw):
        assert quantized and kw['initial_state'] is packed
        assert kw['initial_state_indices'].tolist() == [0, 1]
        assert kw['cu_seqlens'] is plan.query_start_loc
        calls.append('triton')
        return result, None, history
    def quantize(layout, dst, values, indices):
        assert quantized and dst is views
        commits.append((values.clone(), indices.clone()))
    monkeypatch.setattr(module, "materialize", materialize)
    monkeypatch.setattr(module, "gdn_prefill", dispatcher)
    monkeypatch.setattr(module, "chunk_gated_delta_rule", triton)
    monkeypatch.setattr(module, "quantize", quantize)
    mixed = torch.randn(130, 40)
    assert layer._prefill_core(mixed, None, None, ctx) is result
    assert torch.equal(conv[7], mixed[61:64].T)
    assert calls == ['triton' if quantized else 'dispatcher']
    if quantized:
        assert len(commits) == 2
        assert torch.equal(commits[0][0], packed) and torch.equal(commits[0][1], slots)
        assert torch.equal(commits[1][0], history[0, [1]])
        assert commits[1][1].tolist() == [7]
    else:
        assert not commits
