"""Three-bit routed experts (kernels/w3_moe.py): the sim3 experts convert to the W3 format without rounding again,
fc1's row order pairs every up row with its gate row inside a 32-row block, and the scale unswizzle inverts moe.py's."""
import pytest
import torch

from sharar.kernels import w3, w3_moe
from sharar.model.flashnext.moe import swizzle_block_scales_batched
from sharar.weights.expert_sim3 import _decode, requantize

E, I, H = 2, 128, 256  # small shapes the packing accepts (rows and k multiples of 128, I of 64)


def _sim3(rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (E, rows, k // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand(E, rows, k // 16, generator=g) * 0.02 + 0.001).to(torch.float8_e4m3fn)
    p, s, _ = requantize(packed, scales, 32)
    return p, s


def test_fc1_order_pairs_up_and_gate_rows():
    order = w3_moe.fc1_order(I)
    assert sorted(order.tolist()) == list(range(2 * I))
    blocks = order.reshape(-1, 32)
    assert torch.equal(blocks[:, 16:], blocks[:, :16] + I)  # gate row of the same intermediate column


def test_sim3_experts_pack_exactly():
    w13, s13 = _sim3(2 * I, H, 0)
    w2, s2 = _sim3(H, I, 1)
    g13, g2 = torch.tensor([0.5, 0.25]), torch.tensor([0.125, 2.0])
    ex = w3_moe.from_sim3(w13, s13, g13, w2, s2, g2, chunk=1)
    inverse = torch.argsort(w3_moe.fc1_order(I))
    assert ex.planes13.shape == (E, 2 * I // 32, H // 128, 3, 32, 4) and ex.planes2.shape == (E, H // 32, I // 128, 3, 32, 4)
    for e in range(E):
        fc1 = w3.dequantize(ex.matrix(e, True))[inverse]
        fc2 = w3.dequantize(ex.matrix(e, False))
        torch.testing.assert_close(fc1, _decode(w13[e], s13[e]) * g13[e], rtol=1e-6, atol=0)
        torch.testing.assert_close(fc2, _decode(w2[e], s2[e]) * g2[e], rtol=1e-6, atol=0)
    per_weight = 8 * ex.bytes / (E * 3 * I * H)
    assert 3.25 < per_weight < 3.5  # codes and group scales, plus one FP32 multiplier a row


def test_plain_nvfp4_is_refused():
    g = torch.Generator().manual_seed(2)
    packed = torch.randint(0, 256, (4, 64), generator=g, dtype=torch.uint8)
    scales = torch.full((4, 8), 0.01).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        w3_moe.sim3_codes(packed, scales)


def test_unswizzle_inverts_the_layer_swizzle():
    g = torch.Generator().manual_seed(3)
    s = (torch.rand(E, 2 * I, H // 16, generator=g) + 0.01).to(torch.float8_e4m3fn)
    back = w3_moe.unswizzle_batched(swizzle_block_scales_batched(s), 2 * I, H // 16)
    assert torch.equal(back.view(torch.uint8), s.view(torch.uint8))
