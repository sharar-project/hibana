import torch

from sharar.weights.modelopt import swizzle_block_scales


def test_swizzle_is_a_permutation_with_padding():
    m, k = 256, 8  # two 128-row blocks, two 4-column groups
    scales = torch.arange(m * k, dtype=torch.float32).reshape(m, k)
    sw = swizzle_block_scales(scales)
    assert sw.shape == (256, 8)
    assert sorted(sw.flatten().tolist()) == sorted(scales.flatten().tolist())
    # First element of block (0,0) stays first; row 1 of the source moves to
    # position 32 within the block (the 32x4 inner tile is transposed).
    assert sw[0, 0].item() == scales[0, 0].item()
    assert sw.reshape(2, 4, 32, 2, 4)[0, 0, 0, 0, :].tolist() == scales[0, :4].tolist()


def test_swizzle_pads_to_128_rows():
    scales = torch.ones((130, 4), dtype=torch.float32)
    sw = swizzle_block_scales(scales)
    assert sw.shape == (256, 4)
    assert sw.sum().item() == 130 * 4
