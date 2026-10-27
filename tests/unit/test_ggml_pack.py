"""Native GGML packing preserves checkpoint bytes and exact FP8 components."""
import hashlib
import struct

import pytest
import torch

from sharar.kernels.ggml import _execution_lut, pack_weight
from sharar.weights.ggml import FORMATS, dequantize
from sharar.weights.iq2_grid import GRID
from sharar.weights.iq2_xs_grid import GRID_XS


def blocks(quant, n=136, k=768):
    block, size = FORMATS[quant]
    raw = torch.randint(0, 256, (n, k // block, size), dtype=torch.uint8,
                        generator=torch.Generator().manual_seed(92))
    bases = torch.tensor([0., -0., 2**-24, -0.0625, 0.125, 1.], dtype=torch.float16)
    raw[..., :2] = bases[torch.arange(n * (k // block)).reshape(n, k // block) % len(bases)][..., None].view(torch.uint8)
    return raw


@pytest.mark.parametrize("quant", FORMATS)
def test_pack_is_lossless_owned_and_compact(quant):
    raw = blocks(quant)
    expected = raw.clone()
    weight = pack_weight(raw, quant)
    assert weight.bytes == raw.numel()
    assert (weight.out_features, weight.in_features) == (136, 768)
    assert torch.equal(weight.blocks(), expected)
    assert torch.equal(weight.dequantize(), dequantize(expected, quant))
    raw.zero_()
    assert torch.equal(weight.blocks(), expected)


@pytest.mark.parametrize("quant", FORMATS)
def test_pack_strided_block_views(quant):
    raw = blocks(quant, n=16, k=1024)[::2, ::2]
    assert not raw.is_contiguous()
    assert torch.equal(pack_weight(raw, quant).blocks(), raw)


@pytest.mark.parametrize("quant", FORMATS)
def test_execution_table_exact_integer_decomposition(quant):
    table = _execution_lut(quant, torch.device("cpu"))
    assert table.numel() == (512 if quant == "IQ2_XS" else 256)
    if quant == "Q8_0":
        raw = table.to(torch.int64)
        high = (raw & 255).to(torch.uint8).view(torch.float8_e4m3fn).float()
        low = ((raw >> 8) & 255).to(torch.uint8).view(torch.float8_e4m3fn).float()
        expected = torch.arange(256).to(torch.uint8).view(torch.int8).float()
    else:
        selectors = (table.to(torch.int64)[:, None] >> (4 * torch.arange(8))) & 15
        high = torch.tensor([8., 24., 44.])[selectors]
        low = torch.tensor([0., 1., -1.])[selectors]
        expected = torch.tensor(GRID_XS if quant == "IQ2_XS" else GRID)
    assert torch.equal(high + low, expected)
    assert (high != expected).any()  # A single rounded FP8 component would lose native weights.


def test_xs_grid_matches_pinned_source():
    assert len(GRID_XS) == 512 and all(len(row) == 8 for row in GRID_XS)
    assert hashlib.sha256(bytes(x for row in GRID_XS for x in row)).hexdigest() == "06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de"


def test_xs_scales_change_every_sixteen_and_signs_include_parity():
    descriptors = [((j % 128) << 9) | (j % 2) for j in range(32)]
    scale_bytes = bytes(lo | ((15 - lo) << 4) for lo in range(8))
    raw = struct.pack("<e32H", -0.125, *descriptors) + scale_bytes
    packed = torch.tensor(list(raw), dtype=torch.uint8).reshape(1, 1, 74)
    expected = []
    for j, descriptor in enumerate(descriptors):
        signs = descriptor >> 9
        mask = signs | ((signs.bit_count() & 1) << 7)
        subscale = (scale_bytes[j // 4] >> (4 * (j // 2 % 2))) & 15
        vector = [8] * 8 if descriptor & 511 == 0 else [43] + [8] * 7
        expected.extend(-0.125 * (subscale + 0.5) * 0.25 * value * (-1 if mask & (1 << bit) else 1)
                        for bit, value in enumerate(vector))
    assert torch.equal(dequantize(packed, "IQ2_XS"), torch.tensor([expected]))


@pytest.mark.parametrize("quant", FORMATS)
@pytest.mark.parametrize("base", [float("inf"), float("nan")])
def test_nonfinite_bases_rejected(quant, base):
    raw = blocks(quant, n=8, k=256)
    raw[0, 0, :2] = torch.tensor([base], dtype=torch.float16).view(torch.uint8)
    with pytest.raises(ValueError, match="finite"):
        pack_weight(raw, quant)


@pytest.mark.parametrize("n", [0, 1, 7, 9])
def test_native_requires_n8(n):
    with pytest.raises(ValueError, match="divisible"):
        pack_weight(torch.zeros(n, 1, 66, dtype=torch.uint8), "IQ2_XXS")
