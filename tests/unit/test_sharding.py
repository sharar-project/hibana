import pytest
import torch

from hibana.distributed import set_tp_info
from hibana.target import QWEN38
from hibana.weights.sharding import REPLICATED, ROW_PARALLEL, ColumnParallel


@pytest.fixture
def rank1_of_2():
    set_tp_info(1, 2)
    yield
    set_tp_info(0, 1)


def test_column_parallel_keeps_its_rows_of_every_fused_block(rank1_of_2):
    # q|k|v as the checkpoint fuses them: 4 q rows, 2 k rows, 2 v rows; rank 1 keeps the second half of each
    t = torch.arange(8).repeat_interleave(3).view(8, 3)
    assert ColumnParallel((2, 1, 1))(t)[:, 0].tolist() == [2, 3, 5, 7]
    # a packed layout whose dim 0 indexes two rows: blocks of 2 + 2 units, rank 1 keeps the second unit of each
    assert ColumnParallel((2, 2)).rows(torch.arange(4), unit=2).tolist() == [1, 3]


def test_row_parallel_keeps_its_input_columns(rank1_of_2):
    t = torch.arange(12).view(2, 6)
    assert ROW_PARALLEL(t).tolist() == [[3, 4, 5], [9, 10, 11]]
    assert ROW_PARALLEL.rows(t) is t and ColumnParallel((1,)).cols(t) is t and REPLICATED(t) is t


def test_target_shard_is_one_ranks_share():
    t = QWEN38.shard(2)
    assert (t.num_attention_heads, t.num_kv_heads, t.linear_num_key_heads, t.linear_num_value_heads, t.intermediate_size) == (12, 2, 8, 24, 8704)
    assert (t.hidden_size, t.vocab_size) == (QWEN38.hidden_size, QWEN38.vocab_size)
    assert t.conv_dim == QWEN38.conv_dim // 2 and t.kv_bytes_per_token == QWEN38.kv_bytes_per_token // 2
    assert QWEN38.shard(1) is QWEN38


def test_flashnext_shard_splits_heads_and_every_expert_width():
    from hibana.target import FLASHNEXT
    t = FLASHNEXT.shard(2)
    assert (t.num_attention_heads, t.num_kv_heads, t.linear_num_key_heads, t.linear_num_value_heads) == (12, 1, 8, 24)
    assert (t.moe_intermediate_size, t.shared_expert_intermediate_size) == (320, 320)
    assert (t.num_experts, t.experts_per_token, t.hidden_size, t.hc_count) == (FLASHNEXT.num_experts, FLASHNEXT.experts_per_token, FLASHNEXT.hidden_size, FLASHNEXT.hc_count)
    assert t.intermediate_size == FLASHNEXT.intermediate_size  # no dense MLP: left unset


def test_trunk_linear_parts_are_the_whole_matrix_in_fp8():
    """Every rank quantizes with the whole matrix's scale, so the ranks' parts are exactly the single-GPU copy's
    rows (column-parallel) or columns (row-parallel)."""
    from hibana.model.flashnext.linear import TrunkLinear
    w = torch.randn(64, 32, dtype=torch.bfloat16)
    whole = TrunkLinear.from_bf16(w, "fp8")
    rows, cols = [], []
    for rank in range(2):
        set_tp_info(rank, 2)
        try:
            rows.append(TrunkLinear.from_bf16(w, "fp8", ColumnParallel((32,))))
            cols.append(TrunkLinear.from_bf16(w, "fp8", ROW_PARALLEL))
        finally:
            set_tp_info(0, 1)
    assert all(torch.equal(p.scale, whole.scale) for p in rows + cols)
    assert torch.equal(torch.cat([p.weight for p in rows]).view(torch.uint8), whole.weight.view(torch.uint8))
    assert torch.equal(torch.cat([p.weight for p in cols], dim=1).view(torch.uint8), whole.weight.view(torch.uint8))
    assert rows[1].shard == ColumnParallel((32,)) and cols[0].shard is ROW_PARALLEL


def test_single_rank_collectives_are_identities():
    from hibana.distributed import all_reduce_min, common_values
    assert all_reduce_min(3, 1) == [3, 1]
    assert common_values([5, 2, 9]) == [5, 2, 9]
