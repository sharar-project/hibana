"""Simulated 3-bit routed experts (weights/expert_sim3.py): the output is valid NVFP4 whose values lie in the
3-bit format (eight levels times one E4M3 scale per group), and the error is what that format costs."""
import pytest
import torch

from sharar.weights.expert_sim3 import LEVELS, _decode, requantize
from sharar.weights.modelopt import unpack_fp4


def _nvfp4(e, r, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (e, r, k // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand(e, r, k // 16, generator=g) * 0.02 + 0.001).to(torch.float8_e4m3fn)
    return packed, scales


def test_decode_matches_the_modelopt_unpacker():
    packed, scales = _nvfp4(1, 8, 64)
    assert torch.equal(_decode(packed[0], scales[0]), unpack_fp4(packed[0], scales[0]))


@pytest.mark.parametrize("group", [16, 32])
def test_output_is_the_three_bit_format(group):
    packed, scales = _nvfp4(3, 16, 128)
    p, s, err = requantize(packed, scales, group, chunk=2)
    assert p.shape == packed.shape and p.dtype == torch.uint8 and s.shape == scales.shape and s.dtype == torch.float8_e4m3fn
    for i in range(3):
        values = _decode(p[i], s[i])
        blocks = s[i].float().reshape(16, 128 // group, group // 16)
        assert torch.equal(blocks, blocks[..., :1].expand_as(blocks))  # one scale per group
        scale = s[i].float().repeat_interleave(16, -1)
        mags = torch.where(scale > 0, values.abs() / scale, torch.zeros_like(values))
        assert set(mags[scale > 0].unique().tolist()) <= set(LEVELS)
    assert 0 < err < 0.5


def test_group_16_is_closer_than_group_32():
    packed, scales = _nvfp4(2, 32, 256, seed=3)
    assert requantize(packed, scales, 16)[2] < requantize(packed, scales, 32)[2]


def test_invalid_group():
    packed, scales = _nvfp4(1, 16, 64)
    with pytest.raises(ValueError):
        requantize(packed, scales, 8)
