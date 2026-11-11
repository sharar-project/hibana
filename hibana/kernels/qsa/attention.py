# SPDX-License-Identifier: Apache-2.0
# The sparse attention kernel is sglang's `_sparse_gqa_chunk_prefill`
# (sglang v0.5.20, srt/layers/attention/qsa/sparse_attn.py, Apache-2.0),
# unchanged except for the wrapper below.
"""Qwen Sparse Attention (Flash-Next): attention of each query row over
its own list of KV pages.

Every row is a query token with up to W selected pages of the KV pool
(packed first, -1 after) and a count of valid entries. Rows past the indexer
budget select 512 groups of four tokens plus their pending tail; rows inside
it list every page they may see, so the same kernel serves both. K/V are read
through the page ids: an FP8 pool cast to the query's dtype as stored (no
scales); an NVFP4 pool (`SHARAR_KV_FORMAT=nvfp4`) converted to FP16 by the
hardware (two E2M1 codes an instruction) and multiplied by its E4M3 scale per
16 values, exact in FP16, both products on FP16 operands. NVFP4 keeps a code
pair a byte, so its rows split into even and odd columns (low and high
nibbles), and Q and the output with them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# sglang's launch configurations by query rows (its non-H20 table); BLOCK_N sets the summation order.
_MAX_GRID_Y = 65535  # CUDA's limit on a grid's second dimension, rows x KV heads here
_CONFIGS = [(32, (32, 8, 2)), (64, (64, 8, 2)), (128, (64, 4, 2)), (512, (32, 4, 2)), (float("inf"), (16, 1, 2))]
# NVFP4: the same pages a block (the same summation order), warps and stages measured on GB10 on a 64K prompt's own
# selections (Flash-Next): at or under FP8's time in every bucket, one warp and no pipelining for prefill chunks
_NVFP4_CONFIGS = [(32, (32, 4, 3)), (64, (64, 2, 1)), (128, (64, 2, 1)), (512, (32, 1, 1)), (float("inf"), (16, 1, 1))]


@triton.jit
def _e2m1x2(codes):
    """uint8 codes -> (low nibbles, high nibbles) as FP16, by the hardware conversion (cvt.rn.f16x2.e2m1x2)."""
    return tl.inline_asm_elementwise(
        """{
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 r0, r1, r2, r3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 r0, b0;
        cvt.rn.f16x2.e2m1x2 r1, b1;
        cvt.rn.f16x2.e2m1x2 r2, b2;
        cvt.rn.f16x2.e2m1x2 r3, b3;
        prmt.b32 $0, r0, r1, 0x5410;
        prmt.b32 $1, r2, r3, 0x5410;
        prmt.b32 $2, r0, r1, 0x7632;
        prmt.b32 $3, r2, r3, 0x7632;
        }""",
        "=r,=r,=r,=r,r", [codes], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)


@triton.jit
def _sparse_gqa_nvfp4(
    q, k, v, ks, vs, out, indices, kv_lens, scale, topk,
    sq_m, sq_h, sk_n, sk_h, ss_n, ss_h, so_m, so_h, si_m,
    NUM_KV_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """`_sparse_gqa_chunk_prefill` (below) over an NVFP4 pool, one query a row: k/v [pages, Hkv, D / 2] uint8 codes,
    ks/vs [pages, Hkv, D / 16] E4M3 scales. The same loop and softmax; each product is split over the even and odd
    columns. Neighbouring rows select mostly the same pages, so the reads hit L2 and the conversion is the cost: the
    hardware's (`_e2m1x2`) and one multiply by the scale broadcast over its 16 values keep it at FP8's speed."""
    query = tl.program_id(1) // NUM_KV_HEADS
    group = tl.program_id(1) % NUM_KV_HEADS
    row_topk = tl.minimum(topk, tl.load(kv_lens + query))
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    H: tl.constexpr = HEAD_DIM // 2
    B: tl.constexpr = HEAD_DIM // 16
    query = query.to(tl.int64)
    offs_h = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, H)
    offs_b = tl.arange(0, B)
    head = (group * GROUP_SIZE + offs_h[:, None]) * sq_h
    live = (offs_h < GROUP_SIZE)[:, None]
    q_even = tl.load(q + query * sq_m + head + 2 * offs_c[None, :], mask=live, other=0.0)
    q_odd = tl.load(q + query * sq_m + head + 2 * offs_c[None, :] + 1, mask=live, other=0.0)
    q_even = (q_even.to(tl.float32) * scale * 1.4426950408).to(tl.float16)
    q_odd = (q_odd.to(tl.float32) * scale * 1.4426950408).to(tl.float16)
    k_base = k + group * sk_h
    v_base = v + group * sk_h
    ks_base = ks + group * ss_h
    vs_base = vs + group * ss_h
    idx_row = indices + query * si_m
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, H], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, H], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current, mask=current < topk, other=-1)
        valid = token >= 0
        page = tl.where(valid, token, 0).to(tl.int64)  # page 0, always valid memory, for the empty entries
        codes = tl.load(k_base + page[None, :] * sk_n + offs_c[:, None])  # [H, N]
        sf = tl.load(ks_base + page[None, :] * ss_n + offs_b[:, None]).to(tl.float16)  # [B, N]
        sf = tl.where(valid[None, :], sf, 0.0)[:, None, :]
        k_even, k_odd = _e2m1x2(codes)
        k_even = tl.reshape(tl.reshape(k_even, (B, 8, BLOCK_N)) * sf, (H, BLOCK_N))
        k_odd = tl.reshape(tl.reshape(k_odd, (B, 8, BLOCK_N)) * sf, (H, BLOCK_N))
        scores = tl.dot(q_even, k_even) + tl.dot(q_odd, k_odd)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        codes = tl.load(v_base + page[:, None] * sk_n + offs_c[None, :])  # [N, H]
        sf = tl.load(vs_base + page[:, None] * ss_n + offs_b[None, :]).to(tl.float16)  # [N, B]
        sf = tl.where(valid[:, None], sf, 0.0)[:, :, None]
        v_even, v_odd = _e2m1x2(codes)
        v_even = tl.reshape(tl.reshape(v_even, (BLOCK_N, B, 8)) * sf, (BLOCK_N, H))
        v_odd = tl.reshape(tl.reshape(v_odd, (BLOCK_N, B, 8)) * sf, (BLOCK_N, H))
        p = probabilities.to(q_even.dtype)
        acc_even = tl.dot(p, v_even, acc_even * alpha[:, None])
        acc_odd = tl.dot(p, v_odd, acc_odd * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    dst = out + query * so_m + (group * GROUP_SIZE + offs_h[:, None]) * so_h + 2 * offs_c[None, :]
    tl.store(dst, acc_even / normalizer[:, None], mask=live)
    tl.store(dst + 1, acc_odd / normalizer[:, None], mask=live)


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
                     scale: float, *, optimized_prefill: bool = False, scales: tuple | None = None) -> torch.Tensor:
    """q [R, Hq, D] (bf16); k_pool/v_pool [num_pages, Hkv, D] (the FP8 pool, page size 1), or
    [num_pages, Hkv, D / 2] uint8 NVFP4 codes with `scales` (k, v) [num_pages, Hkv, D / 16] E4M3;
    pages [R, W] int32 page ids, valid entries first, -1 after; counts [R] int32 valid
    entries per row. Returns [R, Hq, D] in q's dtype. Every row is its own sequence
    of one query, so its visible length is its count."""
    rows, heads, dim = q.shape
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    out = torch.empty_like(q)
    if rows == 0:
        return out
    if rows * kv_heads > _MAX_GRID_Y:  # one prompt prefilled whole (the scheduler's chunks stay under it): in parts
        step = _MAX_GRID_Y // kv_heads
        for r in range(0, rows, step):
            out[r:r + step] = sparse_attention(q[r:r + step], k_pool, v_pool, pages[r:r + step], counts[r:r + step], scale,
                                               optimized_prefill=optimized_prefill, scales=scales)
        return out
    if scales is not None:
        ks, vs = scales
        block_n, warps, stages = next(cfg for limit, cfg in _NVFP4_CONFIGS if rows <= limit)
        _sparse_gqa_nvfp4[(1, rows * kv_heads)](
            q, k_pool, v_pool, ks, vs, out, pages, counts, scale, pages.shape[-1],
            q.stride(0), q.stride(1), k_pool.stride(0), k_pool.stride(1), ks.stride(0), ks.stride(1),
            out.stride(0), out.stride(1), pages.stride(0),
            NUM_KV_HEADS=kv_heads, GROUP_SIZE=group, BLOCK_M=max(16, triton.next_power_of_2(group)), BLOCK_N=block_n,
            HEAD_DIM=dim, num_warps=warps, num_stages=stages,
        )
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
