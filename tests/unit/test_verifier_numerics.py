import pytest
import torch

from bench.verifier_numerics import compare_traces, emission, unpack_fp4, unswizzle


def test_nvfp4_nibble_order_signs_and_block_scale():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]], dtype=torch.uint8)
    expected = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]]) * 2
    assert torch.equal(unpack_fp4(packed, torch.tensor([[2.]])), expected)


def test_scale_layout_against_explicit_kernel_addresses_across_tiles():
    rows, groups = 129, 5
    physical = torch.zeros(256 * 8)
    expected = torch.arange(rows * groups).reshape(rows, groups).float()
    for row in range(rows):
        for col in range(groups):
            offset = row // 128 * (128 * 8) + row % 32 * 16 + (row // 32 % 4) * 4 + col // 4 * 512 + col % 4
            physical[offset] = expected[row, col]
    assert torch.equal(unswizzle(physical.reshape(256, 8), rows, groups), expected)


def test_oracle_rejects_mismatched_extents():
    with pytest.raises(ValueError, match='scale extent'):
        unswizzle(torch.zeros(128, 4), 129, 5)
    with pytest.raises(ValueError, match='packed rows'):
        unpack_fp4(torch.zeros(1, 8, dtype=torch.uint8), torch.ones(1, 2))


def test_emission_checks_the_bonus_row_and_stops_at_first_rejection():
    assert emission([10, 20, 30], [10, 99, 30, 40]) == [10, 99]
    assert emission([10, 20, 30], [10, 20, 30, 40]) == [10, 20, 30, 40]
    with pytest.raises(ValueError, match='bonus'):
        emission([10, 20, 30], [10, 20, 30])


def test_trace_comparison_locates_first_difference_and_requires_same_stages():
    original = {'00/input': torch.ones(2), '00/ba': torch.ones(2), '01/output': torch.ones(2)}
    altered = {k: v.clone() for k, v in original.items()}
    altered['00/ba'][1] = 1.125
    altered['01/output'][0] = 2
    result = compare_traces(original, altered)
    assert result['first_difference'] == '00/ba' and result['different_stages'] == 2
    assert result['stages']['00/ba']['max_abs'] == .125
    with pytest.raises(ValueError, match='stages differ'):
        compare_traces(original, {'00/input': torch.ones(2)})
