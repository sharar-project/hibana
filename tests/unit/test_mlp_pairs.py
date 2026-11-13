"""An MLP's gate|up interleaved in blocks (`modelopt.interleave_pairs`): the NVFP4 rows and their 3-bit copy agree with
the concatenated layout (SiLU-mul over the pairs: tests/gpu/test_mlp_fusion.py)."""
import torch

from hibana.kernels import w3
from hibana.weights.modelopt import Nvfp4Linear, interleave_pairs, pair_order, swizzle_block_scales

BLOCK = 64


def linear(n: int, k: int) -> Nvfp4Linear:
    g = torch.Generator().manual_seed(n + k)
    scales = (torch.rand((n, k // 16), generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    return Nvfp4Linear(weight=torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g),
                       weight_scale_swizzled=swizzle_block_scales(scales.view(torch.uint8)).view(torch.float8_e4m3fn),
                       alpha=torch.tensor(0.01), input_scale_inv=torch.tensor(2.0), out_features=n, in_features=k)


def test_pair_order_alternates_gate_and_up_blocks():
    order = pair_order(512, BLOCK).tolist()
    assert order[:BLOCK] == list(range(BLOCK)) and order[BLOCK:2 * BLOCK] == list(range(256, 256 + BLOCK))
    assert sorted(order) == list(range(512))


def test_interleaved_weights_and_their_3bit_copy():
    natural = linear(512, 256)
    reference = natural.dequantize()
    paired = interleave_pairs(linear(512, 256), BLOCK)
    order = pair_order(512, BLOCK)
    assert paired.pairs == BLOCK and torch.equal(paired.dequantize(), reference[order])
    # a calibrated copy is stored concatenated and interleaved at load; a nearest-level one packs the interleaved rows
    moved, packed = w3.interleave_rows(w3.pack(reference), BLOCK), w3.pack(paired.dequantize())
    assert all(torch.equal(getattr(moved, f), getattr(packed, f)) for f in ("planes", "scales", "row_scale"))
