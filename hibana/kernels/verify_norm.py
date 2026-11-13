# SPDX-License-Identifier: Apache-2.0
"""Normalization arithmetic of the pinned SGLang compiled verification path.

Residual variance uses the FP32 sum; the numerator is reloaded from its BF16
store. Keep the two loops and reduction layout: replacing the reload with an
in-register cast or fusing FP8 quantization changes midpoint rounding.
The compiled GDN kernel promotes epsilon and subsequent normalization math
to FP64. Prefill keeps the native CUDA / FP32 implementations.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _input_norm(X, W, Y, M, N: tl.constexpr, EPS: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0) + tl.arange(0, 1)[:, None]
    lane = tl.arange(0, BLOCK)[None, :]
    acc = tl.full((1, BLOCK), 0, tl.float32)
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        mask = (col < N) & (row < M)
        x = tl.load(X + row * N + col, mask, 0, eviction_policy='evict_last').to(tl.float32)
        squared = tl.broadcast_to(x * x, (1, BLOCK))
        acc = tl.where(mask, acc + squared, acc)
    variance_sum = tl.sum(acc, 1)[:, None]
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        mask = (col < N) & (row < M)
        x = tl.load(X + row * N + col, mask, 0, eviction_policy='evict_first').to(tl.float32)
        w = tl.load(W + col, col < N, 0, eviction_policy='evict_last').to(tl.float32)
        variance = variance_sum / tl.full((1, 1), N, tl.float32)
        rstd = libdevice.rsqrt(variance + tl.full((1, 1), EPS, tl.float32))
        y = x * rstd * (w + tl.full((1, 1), 1.0, tl.float32))
        tl.store(Y + row * N + col, y, mask)


def verify_input_norm(x, weight, eps):
    """Initial layer norm with the compiled profile's chunked reduction."""
    assert x.ndim == 2 and x.is_contiguous()
    rows, width = x.shape
    out = torch.empty_like(x)
    _input_norm[(rows,)](x, weight, out, rows, width, eps, 2048,
                         num_warps=8, num_stages=1)
    return out


@triton.jit
def _add_norm(X, R, W, Sum, Y, M, N: tl.constexpr, EPS: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0) + tl.arange(0, 1)[:, None]
    lane = tl.arange(0, BLOCK)[None, :]
    acc = tl.full((1, BLOCK), 0, tl.float32)
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        mask = (col < N) & (row < M)
        x = tl.load(X + row * N + col, mask, 0, eviction_policy='evict_first').to(tl.float32)
        r = tl.load(R + row * N + col, mask, 0, eviction_policy='evict_first').to(tl.float32)
        summed = x + r
        squared = tl.broadcast_to(summed * summed, (1, BLOCK))
        acc = tl.where(mask, acc + squared, acc)
        tl.store(Sum + row * N + col, summed, mask)
    variance_sum = tl.sum(acc, 1)[:, None]
    for start in tl.range(0, N, BLOCK):
        col = start + lane
        mask = (col < N) & (row < M)
        numerator = tl.load(Sum + row * N + col, mask, 0, eviction_policy='evict_first').to(tl.float32)
        w = tl.load(W + col, col < N, 0, eviction_policy='evict_last').to(tl.float32)
        variance = variance_sum / tl.full((1, 1), N, tl.float32)
        rstd = libdevice.rsqrt(variance + tl.full((1, 1), EPS, tl.float32))
        y = numerator * rstd * (w + tl.full((1, 1), 1.0, tl.float32))
        tl.store(Y + row * N + col, y, mask)


def verify_add_norm(x, residual, weight, eps):
    """Return BF16 normalized rows and update the caller-owned residual."""
    assert x.ndim == 2 and x.is_contiguous() and residual.is_contiguous()
    rows, width = x.shape
    summed, out = torch.empty_like(x), torch.empty_like(x)
    _add_norm[(rows,)](x, residual, weight, summed, out, rows, width, eps,
                       triton.next_power_of_2(width), num_warps=16, num_stages=1)
    residual.copy_(summed)
    return out


@triton.jit
def _gated_norm(X, Z, W, Y, M: tl.constexpr, N: tl.constexpr,
                EPS: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, N)
    mask = row[:, None] < M
    x = tl.load(X + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + row[:, None] * N + col[None, :], mask, 0).to(tl.float32)
    weight = tl.load(W + col).to(tl.float32)
    variance = tl.sum(x * x, 1) / N
    rstd = tl.rsqrt(variance.to(tl.float64) + tl.full((), EPS, tl.float64))
    y = x * rstd[:, None] * weight[None, :]
    y *= z * tl.sigmoid(z)
    tl.store(Y + row[:, None] * N + col[None, :], y, mask)


def verify_gated_norm(core, z, weight, eps, head_dim=128):
    """GDN norm with the compiled reference's FP64 scalar promotion."""
    assert core.is_contiguous() and z.is_contiguous()
    rows = core.numel() // head_dim
    out = torch.empty_like(core)
    # 48 SMs on GB10; this is also the reference's per-head launch layout.
    block = min(triton.next_power_of_2(triton.cdiv(rows, 96)), 4)
    _gated_norm[(triton.cdiv(rows, block),)](
        core, z, weight, out, rows, head_dim, eps, block, num_warps=1, num_stages=3,
    )
    return out
