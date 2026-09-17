"""3-bit fragment-order packing round-trips through the reference dequantizer (CPU)."""
import struct

import pytest
import torch

from sharar.kernels import w3


@pytest.mark.parametrize("codebook", ["uniform", "lloyd"])
def test_pack_dequantize_round_trip_and_size(codebook):
    torch.manual_seed(0)
    weight = torch.randn(256, 384) * 0.05
    packed = w3.pack(weight, codebook)
    assert packed.planes.shape == (8, 3, 3, 32, 4) and packed.planes.dtype == torch.uint32  # [N/32, K/128, load, lane, word]
    assert packed.scales.shape == (8, 3, 4, 32) and packed.scales.dtype == torch.uint8
    assert packed.bytes == 256 * 384 * 3 // 8 + 256 * 12 + 256 * 4  # 3 bits per weight, one E4M3 scale per 32, a row scale
    deq = w3.dequantize(packed)
    assert (deq - weight).norm() / weight.norm() < 0.22  # eight levels per group of 32 on Gaussian weights
    # every code is the nearest level for its group's scale, and the extremes are used
    levels = torch.tensor(packed.codebook)
    scale = w3.group_scales(packed).repeat_interleave(32, dim=1)
    codes = w3.codes(packed)
    assert torch.equal(codes, torch.bucketize(weight / scale, (levels[1:] + levels[:-1]) / 2))
    assert codes.min() == 0 and codes.max() == 7
    assert torch.equal(deq, levels[codes] * scale)
    # the PRMT lookup holds the codebook as E4M3 bytes, entry i in byte i
    raw = struct.pack("<q", packed.lut)
    assert packed.lut == w3.lut_of(packed.codebook)
    assert torch.equal(torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn).float(), levels)


def test_fragment_layout_matches_kernel_addressing():
    """The word a lane reads is the one the kernel's rotate/mask sequence expects."""
    torch.manual_seed(1)
    n, k = 128, 256
    weight = torch.randn(n, k)
    packed = w3.pack(weight, "uniform")
    codes = w3.codes(packed)
    planes = packed.planes.to(torch.int64)  # [nb, kc, i, lane, e]
    nb, kc, g, t = 2, 1, 5, 3
    lane = 4 * g + t
    for nt in range(4):
        words = [planes[nb, kc, (3 * nt + b) // 4, lane, (3 * nt + b) % 4].item() for b in range(3)]
        for s in range(8):
            rot = 16 * (s // 4) + (s % 4)
            rotated = [((w >> rot) | (w << (32 - rot))) & 0xFFFFFFFF for w in words]
            q4 = (rotated[0] & 0x1111) | (rotated[1] & 0x2222) | (rotated[2] & 0x4444)
            for j in range(4):
                row, col = 32 * nb + 8 * nt + g, 128 * kc + 16 * s + 4 * t + j
                assert (q4 >> (4 * j)) & 0xF == codes[row, col].item()
    # the scale bytes a lane reads belong to its accumulator columns 8 nt + 2 t + i, group 4 kc + grp
    scale_bytes = packed.scales[nb, kc, t]
    group_scale = w3.group_scales(packed)
    for nt in range(4):
        for grp in range(4):
            for i in range(2):
                byte = scale_bytes[(nt * 4 + grp) * 2 + i].view(torch.float8_e4m3fn).float().item()
                assert abs(byte * packed.row_scale[32 * nb + 8 * nt + 2 * t + i].item() - group_scale[32 * nb + 8 * nt + 2 * t + i, 4 * kc + grp].item()) < 1e-6


def test_pack_rejects_bad_shapes_and_codebooks():
    with pytest.raises(ValueError):
        w3.pack(torch.zeros(128, 160))
    with pytest.raises(ValueError):
        w3.pack(torch.zeros(96, 256))
    with pytest.raises(ValueError):
        w3.pack(torch.zeros(128, 128), (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.3))  # 3.3 is not an E4M3 value


def test_splits_reach_the_block_target_when_k_allows():
    assert w3.splits_for(34816, 5120) == 1
    assert w3.splits_for(5120, 17408, 16) == 8  # 40 blocks: 8 splits of 17 chunks
    assert w3.splits_for(10240, 5120, 16) == 1  # measured: the table overrides the block target
    assert w3.splits_for(10240, 5120, 64) == 1
    assert w3.splits_for(248320, 5120) == 1
    assert w3.splits_for(5120, 17408, 17) == w3.splits_for(5120, 17408, 32)
    assert w3.splits_for(7168, 5120, 16) == 4 and w3.splits_for(7168, 5120, 64) == 1  # an untabulated shape follows the target
