"""Check BF16 rounding and block boundaries against the unfused implementation."""

import pytest
import torch



@pytest.mark.parametrize('batch,steps', [(1, 8), (1, 16), (3, 16), (4, 8), (4, 16)])
@pytest.mark.parametrize('amplitude', [.01, 1., 100.])
def test_fused_conv_matches_each_bf16_round_and_resets_blocks(batch, steps, amplitude):
    from sharar.model.parallel_draft import convolve, convolve_reference
    torch.manual_seed(912)
    x = (torch.randn(batch, steps, 5120, device='cuda') * amplitude).bfloat16()
    coefficients = torch.randn(batch, steps, 2, 2, 320, device='cuda', dtype=torch.bfloat16)
    base = torch.randn(2, 2, 5120, device='cuda', dtype=torch.bfloat16)
    for side in range(2):
        c = coefficients[:, :, side]
        # The public wrapper also accepts strided base coefficients.
        selected_base = base[side] if side == 0 else base.transpose(1, 2).contiguous().transpose(1, 2)[side]
        expected = convolve_reference(x, c, selected_base)
        actual = convolve(x, c, selected_base)
        assert torch.equal(actual, expected)
