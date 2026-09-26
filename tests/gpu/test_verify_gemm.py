"""Real-weight verifier GEMMs: reference parity and changed graph inputs."""

import pytest
import torch



@pytest.fixture(scope='module', params=['qkv', 'gate_up', 'down'])
def projection(request):
    from sharar import config
    from sharar.weights.modelopt import load_fp8_linear, load_nvfp4_linear
    from sharar.weights.safetensors import SnapshotWeights
    w = SnapshotWeights(config.snapshot_dir())
    prefix = 'model.language_model.layers.'
    try:
        if request.param == 'qkv':
            linear = load_fp8_linear(w, [prefix + '3.self_attn.' + name for name in ('q_proj', 'k_proj', 'v_proj')], 'cuda')
        elif request.param == 'gate_up':
            linear = load_nvfp4_linear(w, [prefix + '32.mlp.' + name for name in ('gate_proj', 'up_proj')], 'cuda')
        else:
            linear = load_nvfp4_linear(w, [prefix + '32.mlp.down_proj'], 'cuda')
    finally:
        w.close()
    yield request.param, linear


@pytest.mark.parametrize('rows', [16, 32, 64])
@torch.inference_mode()
def test_eager_and_graph_replay_match_original_with_new_inputs(projection, rows, monkeypatch):
    from sharar.kernels import ops
    name, linear = projection
    k = linear.weight.shape[1] if name == 'qkv' else linear.in_features
    x = torch.empty(rows, k, device='cuda', dtype=torch.bfloat16)
    call = lambda: ops.fp8_linear(x, linear) if name == 'qkv' else ops.nvfp4_linear(x, linear)
    monkeypatch.setattr(ops, 'VERIFY_GEMM_ENABLED', True)
    x.normal_()
    for _ in range(2):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for seed, amplitude in [(41, 1.), (42, .01), (43, 4.), (44, 0.)]:
        torch.manual_seed(seed)
        x.normal_().mul_(amplitude)
        before = x.clone()
        graph.replay()
        eager = call()
        monkeypatch.setattr(ops, 'VERIFY_GEMM_ENABLED', False)
        reference = call()
        monkeypatch.setattr(ops, 'VERIFY_GEMM_ENABLED', True)
        assert torch.equal(eager, reference), (name, rows, seed, 'eager')
        assert torch.equal(output, reference), (name, rows, seed, 'graph')
        assert torch.equal(x, before)
        assert output.data_ptr() != eager.data_ptr()
