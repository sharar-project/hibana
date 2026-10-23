import pytest
import torch

from sharar.kernels.kv_fp4 import nvfp4_round, fp8_round, _GRID


def _reference(x):
    """Brute force: per block of 16, scale = e4m3(amax / 6); each value the grid point nearest x / scale (ties to the
    even code, codes 0..7 in grid order), saturating at 6."""
    out = torch.empty_like(x, dtype=torch.float32)
    flat = x.float().reshape(-1, 16)
    res = out.reshape(-1, 16)
    for b, block in enumerate(flat):
        s = float((block.abs().max() / 6).to(torch.float8_e4m3fn).float())
        for i, v in enumerate(block.tolist()):
            if s == 0:
                res[b, i] = 0.0
                continue
            a = min(abs(v / s), 6.0)
            best = min(range(8), key=lambda c: (abs(_GRID[c] - a), c % 2))  # nearest, then the even code
            res[b, i] = (_GRID[best] if v >= 0 else -_GRID[best]) * s
    return out


def test_nvfp4_round_matches_the_brute_force_rule():
    torch.manual_seed(0)
    x = torch.randn(8, 4, 256) * torch.logspace(-3, 2, 8).view(8, 1, 1)
    x[0, 0, :16] = 0  # a zero block
    x[1, 1, 3] = 50.0  # an outlier sets its block's scale
    got = nvfp4_round(x.bfloat16())
    assert torch.equal(got.float(), _reference(x.bfloat16()).bfloat16().float())
    assert torch.equal(got, got.float().bfloat16())  # every value is a BF16
    assert bool((got[0, 0, :16] == 0).all())


def test_ties_go_to_the_even_code():
    # a block whose amax is 6 * 2^k has scale 2^k exactly: values at grid midpoints tie
    s = 0.5
    x = torch.tensor([6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.75, -5, 0, 0, 0, 0, 0, 0], dtype=torch.float32) * s
    want = torch.tensor([6, 0, 1, 1, 2, 2, 4, 4, -1, -4, 0, 0, 0, 0, 0, 0], dtype=torch.float32) * s
    assert torch.equal(nvfp4_round(x), want)


def test_fp8_round_is_the_pool_rounding():
    x = torch.randn(64, 256).bfloat16()
    assert torch.equal(fp8_round(x), x.to(torch.float8_e4m3fn).bfloat16())
    with pytest.raises(ValueError):
        nvfp4_round(torch.zeros(3, 10))
