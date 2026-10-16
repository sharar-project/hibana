# SPDX-License-Identifier: Apache-2.0
# The sparse attention kernel is sglang's `_sparse_gqa_chunk_prefill`
# (sglang v0.5.20, srt/layers/attention/qsa/sparse_attn.py, Apache-2.0),
# unchanged except for the wrapper below.
"""Qwen Sparse Attention (Flash-Next): attention of each query row over
its own list of KV pages.

Every row is a query token with up to W selected pages of the FP8 KV pool
(packed first, -1 after) and a count of valid entries. Rows past the indexer
budget select 512 groups of four tokens plus their pending tail; rows inside
it list every page they may see, so the same kernel serves both. K/V are read
through the page ids and cast to the query's dtype (the pool has no scales).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# sglang's launch configurations by query rows (its non-H20 table); BLOCK_N sets the summation order.
_CONFIGS = [(32, (32, 8, 2)), (64, (64, 8, 2)), (128, (64, 4, 2)), (512, (32, 4, 2)), (float("inf"), (16, 1, 2))]


@triton.jit
def _sparse_gqa_chunk_prefill(
    q, k, v, out, indices, cu_q, cu_k, kv_lens, scale, topk,
    sq_m: tl.constexpr, sq_h: tl.constexpr, sq_d: tl.constexpr,
    sk_n: tl.constexpr, sk_h: tl.constexpr, sk_d: tl.constexpr,
    sv_n: tl.constexpr, sv_h: tl.constexpr, sv_d: tl.constexpr,
    so_m: tl.constexpr, so_h: tl.constexpr, so_d: tl.constexpr,
    si_m: tl.constexpr, si_g: tl.constexpr, si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(q + query * sq_m + (group * GROUP_SIZE + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
                       mask=(offs_h < GROUP_SIZE)[:, None], other=0.0)
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d, mask=valid[None, :], other=0.0)
        values = tl.load(v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d, mask=valid[:, None], other=0.0)
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(out + query * so_m + (group * GROUP_SIZE + offs_h[:, None]) * so_h + offs_d[None, :] * so_d, output,
             mask=(offs_h < GROUP_SIZE)[:, None])


def sparse_attention(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, pages: torch.Tensor, counts: torch.Tensor,
                     scale: float, *, optimized_prefill: bool = False) -> torch.Tensor:
    """q [R, Hq, D] (bf16); k_pool/v_pool [num_pages, Hkv, D] (the FP8 pool, page size 1);
    pages [R, W] int32 page ids, valid entries first, -1 after; counts [R] int32 valid
    entries per row. Returns [R, Hq, D] in q's dtype. Every row is its own sequence
    of one query, so its visible length is its count."""
    rows, heads, dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    out = torch.empty_like(q)
    if rows == 0:
        return out
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    cu_k = torch.zeros(rows + 1, dtype=torch.int32, device=q.device)
    block_n, warps, stages = next(cfg for limit, cfg in _CONFIGS if rows <= limit)
    if optimized_prefill and rows > 512 and (heads,kv_heads,dim) in ((24,2,256), (12,1,256)):
        block_n,warps,stages=16,1,1
    _sparse_gqa_chunk_prefill[(1, rows * kv_heads)](
        q, k_pool, v_pool, out, pages, cu_q, cu_k, counts, scale, pages.shape[-1],
        q.stride(0), q.stride(1), q.stride(2), k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), out.stride(0), out.stride(1), out.stride(2),
        pages.stride(0), 0, pages.stride(1),
        NUM_KV_HEADS=kv_heads, GROUP_SIZE=group, BLOCK_M=max(16, triton.next_power_of_2(group)), BLOCK_N=block_n,
        HEAD_DIM=dim, num_warps=warps, num_stages=stages,
    )
    return out
