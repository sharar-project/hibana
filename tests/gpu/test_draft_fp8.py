"""FP8 packing must match the eager pilot, including graph replay inputs."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 tests')


def reference(x):
    scale = (x.float().abs().amax() / 448).clamp_min(1e-12).reshape(1, 1)
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize('shape', [(8, 5120), (16, 4096), (64, 17408), (1, 128)])
@pytest.mark.parametrize('amplitude', [0., 1., 1e-10, 1000.])
def test_dynamic_pack_matches_reference(shape, amplitude):
    from sharar.kernels.draft_fp8 import quant_dynamic
    torch.manual_seed(43)
    x = (torch.randn(shape, device='cuda') * amplitude).bfloat16()
    packed, scale = quant_dynamic(x)
    expected, expected_scale = reference(x)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))


def test_replay_refreshes_scale_and_linear_output():
    from sharar.model.draft_linear import DraftFp8Linear
    torch.manual_seed(48)
    weight = torch.randn(1024, 5120, device='cuda', dtype=torch.bfloat16)
    linear = DraftFp8Linear.from_bf16(weight)
    x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
    for _ in range(2):
        linear(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = linear(x)
    for amplitude in (1., .001, 100.):
        x.normal_().mul_(amplitude)
        graph.replay()
        q, scale = reference(x)
        expected = torch._scaled_mm(q, linear.weight.T, scale_a=scale, scale_b=linear.scale,
                                    out_dtype=torch.bfloat16)
        assert torch.equal(output, expected)
