"""Exact E2M1 tie and signed-zero regression checks against CUDA quantization."""

import pytest
import torch



@pytest.mark.parametrize('scale', [0.5, 1.0, 2.0, 16.0])
@pytest.mark.parametrize('rows', [1, 37, 128])
def test_nvfp4_ties_and_neighbors(scale, rows):
    from sharar.kernels.nvfp4_quant import nvfp4_quantize
    from sharar.kernels.ops import fp4_quantize

    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        device='cuda', dtype=torch.bfloat16)
    below = torch.nextafter(ties, torch.zeros_like(ties))
    above = torch.nextafter(ties, torch.full_like(ties, float('inf')))
    values = torch.cat([below, ties, above, -below, -ties, -above,
                        torch.tensor([0.0, -0.0], device='cuda', dtype=torch.bfloat16)])
    # One tested value per block; amax=6*scale makes each block scale exact.
    blocks = torch.zeros((len(values), 16), device='cuda', dtype=torch.bfloat16)
    blocks[:, 0] = values * scale
    blocks[:, 1] = 6 * scale
    x = blocks.flatten().repeat(rows, 1)
    global_scale = torch.ones((), device='cuda')
    actual, _ = nvfp4_quantize(x, global_scale)
    expected, _ = fp4_quantize(x, global_scale)
    assert torch.equal(actual, expected), (actual != expected).sum().item()
    # Explicit integer oracle makes the quarter tie regression visible even
    # if the external reference changes. Positive/negative zero use 0/8.
    codes = actual.view(rows, len(values), 8)[:, :, 0] & 15
    tie_codes = torch.tensor([0, 2, 2, 4, 4, 6, 6], device='cuda', dtype=torch.uint8)
    assert torch.equal(codes[0, 7:14], tie_codes)
    assert torch.equal(codes[0, 28:35], tie_codes | 8)
    assert codes[0, -2:].tolist() == [0, 8]
