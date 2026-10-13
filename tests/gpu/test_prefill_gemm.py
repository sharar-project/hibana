"""Prefill-sized NVFP4 MLP GEMMs with the chosen CUTLASS tactics, and the consuming residual norm: both bitwise equal
to the default dispatch on real weights."""

import pytest
import torch


@pytest.fixture(scope='module', params=['gate_up', 'down'])
def projection(request):
    from sharar import config
    from sharar.weights.modelopt import load_nvfp4_linear
    from sharar.weights.safetensors import SnapshotWeights
    w = SnapshotWeights(config.snapshot_dir())
    prefix = 'model.language_model.layers.32.mlp.'
    try:
        names = ('gate_proj', 'up_proj') if request.param == 'gate_up' else ('down_proj',)
        linear = load_nvfp4_linear(w, [prefix + name for name in names], 'cuda')
    finally:
        w.close()
    yield request.param, linear


@pytest.mark.parametrize('rows', [512, 1024, 1536, 2048, 3000, 4096])
@torch.inference_mode()
def test_prefill_tactics_match_the_default_dispatch(projection, rows, monkeypatch):
    from sharar.kernels import ops, prefill_gemm
    name, linear = projection
    tuned = prefill_gemm.tactic(rows, linear.out_features, linear.in_features)
    assert (tuned is not None) == (rows >= {'gate_up': 1024, 'down': 1536}[name])
    for seed, amplitude in [(41, 1.), (42, .01), (43, 4.)]:
        torch.manual_seed(seed)
        x = torch.randn(rows, linear.in_features, device='cuda', dtype=torch.bfloat16) * amplitude
        out = ops.nvfp4_linear(x, linear)
        monkeypatch.setattr(prefill_gemm, 'TACTICS', {})
        reference = ops.nvfp4_linear(x, linear)
        monkeypatch.undo()
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (name, rows, seed)


@pytest.mark.parametrize('rows', [1, 2048])
@torch.inference_mode()
def test_consuming_norm_matches_the_copying_norm(rows):
    from sharar.model.norm import GemmaRMSNorm
    torch.manual_seed(7)
    norm = GemmaRMSNorm(torch.randn(5120, device='cuda', dtype=torch.bfloat16) * 0.1, 1e-6)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    scale = torch.tensor([0.05], device='cuda')
    want, want_q, want_residual = norm.fused_add_quant(x.clone(), residual.clone(), scale, inplace=True)
    given = x.clone()
    got, got_q, got_residual = norm.fused_add_quant(given, residual.clone(), scale, inplace=True, consume_input=True)
    assert got.data_ptr() == given.data_ptr()  # no copy
    assert torch.equal(got, want) and torch.equal(got_q.view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(got_residual, want_residual)
    want, want_residual = norm.fused_add(x.clone(), residual.clone(), inplace=True)
    got, got_residual = norm.fused_add(x.clone(), residual.clone(), inplace=True, consume_input=True)
    assert torch.equal(got, want) and torch.equal(got_residual, want_residual)
