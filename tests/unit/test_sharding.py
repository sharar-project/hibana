import pytest
import torch

from sharar.distributed import set_tp_info
from sharar.target import QWEN38
from sharar.weights.sharding import REPLICATED, ROW_PARALLEL, ColumnParallel


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
