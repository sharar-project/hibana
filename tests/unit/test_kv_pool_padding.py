"""Padded CUDA-graph rows write their K/V to pages of their own, never to an allocatable page or to page 0."""

import pytest
import torch

from sharar.cache.kv_pool import KvPool


def test_padding_pages_are_distinct_and_never_allocated():
    pool = KvPool(8, torch.device("cpu"), layer_ids=[3], padding_rows=5)
    assert pool.free_count == 7  # page 0 stays the dummy; the padding pages are not in the free list
    pad = pool.pad_pages(5).tolist()
    assert len(set(pad)) == 5 and not set(pool._free) & set(pad) and 0 not in pad
    assert pool.k[3].shape[0] == 13
    pool.write(3, pool.pad_pages(5), torch.ones(5, pool.num_kv_heads, pool.head_dim), torch.ones(5, pool.num_kv_heads, pool.head_dim))
    assert pool.k[3][0].float().abs().sum() == 0  # page 0, which padded rows attend to, is untouched


def test_page_aux_covers_padding_pages():
    pool = KvPool(8, torch.device("cpu"), layer_ids=[3], padding_rows=5)
    assert pool.register_page_aux("idx", (4,), torch.bfloat16).shape[0] == 13


def test_too_many_padded_rows_is_an_error():
    with pytest.raises(ValueError, match="padding pages"):
        KvPool(8, torch.device("cpu"), layer_ids=[3], padding_rows=2).pad_pages(3)
