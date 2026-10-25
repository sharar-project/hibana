"""Byte-level GGML fixtures, independent of the decoder's vectorized operations."""

import hashlib
import struct

import pytest
import torch

from sharar.weights.ggml import GgmlLinear, dequantize
from sharar.weights.iq2_grid import GRID
from sharar.weights.linear import Linear


def test_q8_signed_values_half_scales_and_multiple_rows():
    values = list(range(-128, -112)) + list(range(112, 128))
    blocks = [struct.pack("<e32b", scale, *values) for scale in (0., 0.5, -2., 1.25)]
    packed = torch.tensor(list(b"".join(blocks)), dtype=torch.uint8).reshape(2, 2, 34)
    expected = torch.tensor([[scale * q for q in values] for scale in (0., 0.5, -2., 1.25)]).reshape(2, 64)
    assert torch.equal(dequantize(packed, "Q8_0"), expected)
    # A row slice remains a valid encoding.
    assert torch.equal(dequantize(packed[1:], "Q8_0"), expected[1:])


def _iq2_fixture(scale):
    """Hand-build eight 32-value groups from known codebook vectors and sign parities."""
    vectors = {0: [8] * 8, 1: [43] + [8] * 7, 255: [8, 25, 8, 8, 25, 43, 43, 43]}
    grids = [0, 1, 255, 0]
    sign_ids = [0, 1, 3, 127]
    signs = [0, 129, 3, 255]  # High sign bit makes the eight-bit parity even.
    payload = bytearray(struct.pack("<e", scale))
    expected = []
    for subscale in (0, 1, 2, 3, 4, 7, 14, 15):
        control = subscale << 28
        for group, sign in enumerate(sign_ids):
            control |= sign << (7 * group)
        payload.extend(bytes(grids) + struct.pack("<I", control))
        for grid, sign in zip(grids, signs):
            expected.extend(scale * (0.5 + subscale) * 0.25 * v * (-1 if sign & (1 << bit) else 1)
                            for bit, v in enumerate(vectors[grid]))
    return bytes(payload), expected


def test_iq2_grid_matches_pinned_ggml_codebook():
    assert len(GRID) == 256 and all(len(row) == 8 for row in GRID)
    assert hashlib.sha256(bytes(x for row in GRID for x in row)).hexdigest() == "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"


def test_iq2_subscales_signs_and_codebook_indices():
    cases = [_iq2_fixture(scale) for scale in (0., 0.125, -0.5, 2.)]
    packed = torch.tensor(list(b"".join(p for p, _ in cases)), dtype=torch.uint8).reshape(2, 2, 66)
    expected = torch.tensor([v for _, v in cases]).reshape(2, 512)
    assert torch.equal(dequantize(packed, "IQ2_XXS"), expected)
    # Non-contiguous row/column-block views are decoded in logical order.
    assert torch.equal(dequantize(packed[:, ::2], "IQ2_XXS"), expected[:, :256])


@pytest.mark.parametrize("quant", ["Q8_0", "IQ2_XXS"])
def test_reference_linear_contract(quant):
    if quant == "Q8_0":
        raw = struct.pack("<e32b", 0.125, *range(32)) * 6
        packed = torch.tensor(list(raw), dtype=torch.uint8).reshape(3, 2, 34)
    else:
        raw = _iq2_fixture(0.125)[0] * 6
        packed = torch.tensor(list(raw), dtype=torch.uint8).reshape(3, 2, 66)
    linear = GgmlLinear(packed, quant)
    assert isinstance(linear, Linear)
    assert linear.bytes == len(raw) and linear.out_features == 3
    assert linear.in_features == (64 if quant == "Q8_0" else 512)
    assert linear.w3 is None
    x = torch.randn(2, 4, linear.in_features)
    torch.testing.assert_close(linear(x), x @ linear.dequantize().T)


@pytest.mark.parametrize("packed,quant", [
    (torch.zeros(2, 3, 66), "IQ2_XXS"),
    (torch.zeros(2, 3, 65, dtype=torch.uint8), "IQ2_XXS"),
    (torch.zeros(2, 3, 66, dtype=torch.uint8), "IQ2_XS"),
    (torch.zeros(2, 34, dtype=torch.uint8), "Q8_0"),
])
def test_rejects_other_encodings_and_shapes(packed, quant):
    with pytest.raises(ValueError):
        dequantize(packed, quant)
