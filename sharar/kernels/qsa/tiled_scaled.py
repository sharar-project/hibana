# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM f26d9f432ced7b1087d02d29bffed2249df1a18e:
# writes causal/empty tiles and fuses FP32 scaling; cache addressing unchanged.
"""Optimized FP32 tiled QSA scorer with fused fill, causal masking and scaling."""
import torch
import triton
import triton.language as tl


@triton.jit
def _score_scaled(
    q_ptr, k_ptr, table_ptr, indptr_ptr, visible_ptr, out_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token,
    stride_table_req, stride_out_row,
    num_rows, query_offset, table_width,
    PAGE_SIZE: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr, BLOCK_N: tl.constexpr, K_TILES: tl.constexpr,
    SCALE: tl.constexpr,
):
    columns_total = table_width * PAGE_SIZE
    HEADS_PAD: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(indptr_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(tl.load(indptr_ptr + request) - query_base, query_offset)
    request_end = tl.minimum(tl.load(indptr_ptr + request + 1) - query_base, query_end)
    row_start = request_start + tl.program_id(1) * TILE_R
    if row_start >= request_end:
        return

    lane_r = tl.arange(0, TILE_R)
    absolute_rows = row_start + lane_r
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(visible_ptr + absolute_rows, mask=valid_rows, other=0)
    max_visible = tl.max(visible, axis=0)
    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * HEADS_PAD)
    q_rows = row_start + m // HEADS_PAD
    heads = m % HEADS_PAD
    query = tl.load(
        q_ptr + q_rows[None, :] * stride_q_row + heads[None, :] * stride_q_head + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS) & (q_rows[None, :] < request_end)
             & (dims[:, None] < HEAD_DIM), other=0.0,
    )
    lane_n = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, K_TILES):
        tile = tl.program_id(2) * K_TILES + local_tile
        col = tile * BLOCK_N + lane_n
        score = tl.full((BLOCK_N, TILE_R), 0.0, tl.float32)
        if tile * BLOCK_N < max_visible:
            logical_page = tl.minimum(col // PAGE_SIZE, table_width - 1)
            physical_page = tl.load(table_ptr + request * stride_table_req + logical_page,
                                    mask=col < max_visible, other=0)
            keys = tl.load(
                k_ptr + physical_page[:, None].to(tl.int64) * stride_k_block
                + (col % PAGE_SIZE)[:, None] * stride_k_token + dims[None, :],
                mask=(col[:, None] < max_visible) & (dims[None, :] < HEAD_DIM),
                other=0.0, eviction_policy="evict_first",
            )
            dot = tl.dot(keys, query, out_dtype=tl.float32)
            dot = tl.reshape(dot, (BLOCK_N, TILE_R, HEADS_PAD))
            score = tl.sum(tl.maximum(dot, 0.0), axis=2)
        value = tl.where(col[:, None] < visible[None, :], score * SCALE, -float("inf"))
        tl.store(out_ptr + rows[None, :] * stride_out_row + col[:, None], value,
                 mask=valid_rows[None, :] & (col[:, None] < columns_total))


def score(q, aux, plan, offset=0, count=None, config=(64, 4, 4)):
    """Compute scaled scores and causal -inf in one launch, writing full rows."""
    count = q.shape[0] - offset if count is None else count
    tile_r, k_tiles, warps = config
    output = torch.empty((count, plan.table.shape[1]), device=q.device, dtype=torch.float32)
    if count == 0 or plan.table.shape[1] == 0:
        return output
    block_n = 64
    grid = (plan.table.shape[0], triton.cdiv(min(count, plan.max_query_len), tile_r),
            triton.cdiv(plan.table.shape[1], block_n * k_tiles))
    _score_scaled[grid](q, aux[:, 1, :], plan.table, plan.indptr, plan.visible, output,
        q.stride(0), q.stride(1), aux.stride(0), 128, plan.table.stride(0), output.stride(0),
        count, offset, plan.table.shape[1], PAGE_SIZE=1, NUM_HEADS=4, HEAD_DIM=128,
        TILE_R=tile_r, BLOCK_N=block_n, K_TILES=k_tiles, SCALE=128**-0.5,
        num_warps=warps)
    return output
