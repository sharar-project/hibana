# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for the Flash-Next trunk's small ops (decode is launch-bound: ~3,600 kernels per step).

Each kernel replaces a chain of torch elementwise/reduction ops with the same FP32 math and BF16 outputs:
grouped Gemma RMS norm (hyper-connection and PLE norms), the mixer's gate-and-mean, the residual
injection, per-tensor dynamic FP8 quantization for a few rows, softmax top-k routing, the MoE combine
with the sigmoid-gated shared expert, SiLU-mul, and the attention's sigmoid output gate. All are
capture-safe (static shapes, no host syncs).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


# ---- grouped Gemma RMS norm: x [T, G*H] -> (x / rms per group) * (1 + w) ----
@triton.jit
def _grouped_norm_kernel(X, W, Y, stride_x, stride_y, H: tl.constexpr, eps, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    s = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = s * H + cols
    x = tl.load(X + t * stride_x + off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, 0) / H
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(var + eps) * (1.0 + w)
    tl.store(Y + t * stride_y + off, y.to(tl.bfloat16), mask=mask)


def grouped_gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int) -> torch.Tensor:
    """[T, G*group] -> bf16 [T, G*group]; RMS per `group` columns, one (1 + w) per column."""
    t, n = x.shape
    g = n // group
    y = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
    _grouped_norm_kernel[(t, g)](x, weight, y, x.stride(0), y.stride(0), group, eps, BLOCK=triton.next_power_of_2(group), num_warps=8)
    return y


# ---- mixer gate and mean: mixed[t, c] = mean_s sigmoid(u[t, s*H + c]) * normed[t, s*H + c] ----
@triton.jit
def _gate_mean_kernel(U, N, Y, stride_u, stride_n, stride_y, H: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(HC):
        u = tl.load(U + t * stride_u + s * H + cols, mask=mask, other=0.0).to(tl.float32)
        n = tl.load(N + t * stride_n + s * H + cols, mask=mask, other=0.0).to(tl.float32)
        acc += _sigmoid(u) * n
    tl.store(Y + t * stride_y + cols, (acc / HC).to(tl.bfloat16), mask=mask)


def gate_mean(up_out: torch.Tensor, normed: torch.Tensor, hc: int, hidden: int) -> torch.Tensor:
    t = up_out.shape[0]
    y = torch.empty((t, hidden), dtype=torch.bfloat16, device=up_out.device)
    block = 1024
    _gate_mean_kernel[(t, triton.cdiv(hidden, block))](up_out, normed, y, up_out.stride(0), normed.stride(0), y.stride(0), hidden, hc, BLOCK=block)
    return y


# ---- residual injection: hyper'[t, s*H + c] = hyper + block[t, c] * 2 sigmoid(inj[t, s]) ----
@triton.jit
def _inject_kernel(R, B, I, Y, stride_r, stride_b, stride_i, stride_y, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    s = tl.program_id(1)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    g = 2.0 * _sigmoid(tl.load(I + t * stride_i + s).to(tl.float32))
    r = tl.load(R + t * stride_r + s * H + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + t * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + s * H + cols, (r + b * g).to(tl.bfloat16), mask=mask)


def inject_combine(hyper: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, hc: int, hidden: int) -> torch.Tensor:
    t = hyper.shape[0]
    y = torch.empty_like(hyper)
    blk = 1024
    _inject_kernel[(t, hc, triton.cdiv(hidden, blk))](hyper, block, inj, y, hyper.stride(0), block.stride(0), inj.stride(0), y.stride(0), hidden, BLOCK=blk)
    return y


# ---- per-tensor dynamic FP8 for a few rows: one program, two passes; also alpha = scale * wscale ----
@triton.jit
def _quant_small_kernel(X, Y, Scale, Alpha, WScale, N: tl.constexpr, BLOCK: tl.constexpr):
    peak = tl.zeros([BLOCK], dtype=tl.float32)
    for start in tl.static_range(0, N, BLOCK):
        i = start + tl.arange(0, BLOCK)
        x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
        peak = tl.maximum(peak, tl.abs(x))
    # PyTorch's division by a Python scalar multiplies by its FP32 reciprocal (matches draft_fp8._pack)
    scale = tl.maximum(tl.max(peak, 0) * (1.0 / 448.0), 1.0e-12)
    tl.store(Scale, scale)
    tl.store(Alpha, scale * tl.load(WScale).to(tl.float32))
    for start in tl.static_range(0, N, BLOCK):
        i = start + tl.arange(0, BLOCK)
        x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
        q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
        tl.store(Y + i, q.to(tl.float8e4nv), mask=i < N)


def quant_fp8_small(x: torch.Tensor, wscale: torch.Tensor):
    """x: contiguous [M, K] with M*K <= 65536 -> (fp8 [M, K], scale [1, 1] fp32, alpha [1] fp32 = scale * wscale)."""
    n = x.numel()
    y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((1, 1), dtype=torch.float32, device=x.device)
    alpha = torch.empty((1,), dtype=torch.float32, device=x.device)
    _quant_small_kernel[(1,)](x, y, scale, alpha, wscale, n, BLOCK=4096, num_warps=8)
    return y, scale, alpha


# ---- softmax top-k routing ----
@triton.jit
def _route_kernel(L, Ids, W, stride_l, E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    logits = tl.load(L + t * stride_l + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(logits, 0)
    p = tl.exp(logits - m)
    p = p / tl.sum(p, 0)
    p = tl.where(mask, p, -1.0)
    work = p
    total = 0.0
    for k in tl.static_range(K):  # pass 1: the top-k mass
        idx = tl.argmax(work, 0)
        total += tl.max(work, 0)
        work = tl.where(cols == idx, -1.0, work)
    work = p
    for k in tl.static_range(K):  # pass 2: store ids and renormalized weights in descending order
        idx = tl.argmax(work, 0)
        val = tl.max(work, 0)
        tl.store(Ids + t * K + k, idx.to(tl.int32))
        tl.store(W + t * K + k, val / total)
        work = tl.where(cols == idx, -1.0, work)


def route_topk(logits: torch.Tensor, k: int):
    """[M, E] -> (int32 ids [M, k], fp32 renormalized softmax weights [M, k]), largest first."""
    m, e = logits.shape
    ids = torch.empty((m, k), dtype=torch.int32, device=logits.device)
    w = torch.empty((m, k), dtype=torch.float32, device=logits.device)
    _route_kernel[(m,)](logits, ids, w, logits.stride(0), e, k, BLOCK=triton.next_power_of_2(e))
    return ids, w


# ---- MoE combine: out = routed + sigmoid(x . g) * shared ----
@triton.jit
def _moe_combine_kernel(R, S, X, G, Y, stride_r, stride_s, stride_x, stride_y, H: tl.constexpr, BLOCK: tl.constexpr, DBLOCK: tl.constexpr):
    t = tl.program_id(0)
    dot = tl.zeros([DBLOCK], dtype=tl.float32)
    for start in tl.static_range(0, H, DBLOCK):
        i = start + tl.arange(0, DBLOCK)
        dot += tl.load(X + t * stride_x + i, mask=i < H, other=0.0).to(tl.float32) * tl.load(G + i, mask=i < H, other=0.0).to(tl.float32)
    g = _sigmoid(tl.sum(dot, 0))
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    r = tl.load(R + t * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S + t * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + cols, (r + g * s).to(tl.bfloat16), mask=mask)


def moe_combine(routed: torch.Tensor, shared: torch.Tensor, x: torch.Tensor, gate_w: torch.Tensor) -> torch.Tensor:
    t, h = routed.shape
    y = torch.empty((t, h), dtype=torch.bfloat16, device=routed.device)
    blk = 1024
    _moe_combine_kernel[(t, triton.cdiv(h, blk))](routed, shared, x, gate_w, y, routed.stride(0), shared.stride(0), x.stride(0), y.stride(0), h, BLOCK=blk, DBLOCK=1024)
    return y


# ---- SiLU-mul: [M, 2I] -> silu(gate) * up ----
@triton.jit
def _silu_mul_kernel(GU, Y, stride_gu, stride_y, I: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(GU + t * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + t * stride_gu + I + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + cols, (g * _sigmoid(g) * u).to(tl.bfloat16), mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    t, n = gu.shape
    i = n // 2
    y = torch.empty((t, i), dtype=torch.bfloat16, device=gu.device)
    blk = 1024
    _silu_mul_kernel[(t, triton.cdiv(i, blk))](gu, y, gu.stride(0), y.stride(0), i, BLOCK=blk)
    return y


# ---- attention output gate: out[t, h*D + d] = attn[t, h, d] * sigmoid(gate[t, h, d]) (gate may be strided) ----
@triton.jit
def _sigmoid_gate_kernel(A, G, Y, stride_at, stride_ah, stride_gt, stride_gh, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    mask = d < D
    a = tl.load(A + t * stride_at + h * stride_ah + d, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + t * stride_gt + h * stride_gh + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + h * D + d, (a * _sigmoid(g)).to(tl.bfloat16), mask=mask)


def sigmoid_gate(attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """attn [T, H, D] (any token/head strides, unit last stride), gate the same -> bf16 [T, H*D]."""
    t, h, d = attn.shape
    y = torch.empty((t, h * d), dtype=torch.bfloat16, device=attn.device)
    _sigmoid_gate_kernel[(t, h)](attn, gate, y, attn.stride(0), attn.stride(1), gate.stride(0), gate.stride(1), y.stride(0), d, BLOCK=triton.next_power_of_2(d))
    return y


# ---- expert GEMV: out[r] = scale[e_r] * (W[e_r] @ x[r]) for FP8 expert stacks [E, N, K] (the MTP head's experts) ----
@triton.jit
def _expert_gemv_kernel(W, S, X, IDX, Y, stride_we, stride_wn, stride_x, stride_y, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    r = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    e = tl.load(IDX + r).to(tl.int64)
    rows = n0 + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        w = tl.load(W + e * stride_we + rows[:, None] * stride_wn + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        x = tl.load(X + r * stride_x + cols, mask=cmask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    s = tl.load(S + e).to(tl.float32)
    tl.store(Y + r * stride_y + rows, (acc * s).to(tl.bfloat16), mask=rmask)


def expert_gemv(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """w: fp8 [E, N, K] with one fp32 scale per expert; x: bf16 [R, K]; idx: int32/int64 [R] expert per row -> bf16 [R, N]."""
    e, n, k = w.shape
    r = x.shape[0]
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    block_n, block_k = 64, 256
    _expert_gemv_kernel[(r, triton.cdiv(n, block_n))](w, scale, x, idx, y, w.stride(0), w.stride(1), x.stride(0), y.stride(0), n, k,
                                                       BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return y


# ---- FP8 GEMV for a few rows and any K (the CUDA GEMV needs K % 512 == 0 and K <= 8192): y = alpha * (W @ x) ----
@triton.jit
def _fp8_gemv_kernel(X, W, Alpha, Y, stride_x, stride_w, stride_y, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    r = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        w = tl.load(W + rows[:, None] * stride_w + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        x = tl.load(X + r * stride_x + cols, mask=cmask, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Y + r * stride_y + rows, (acc * tl.load(Alpha).to(tl.float32)).to(tl.bfloat16), mask=rmask)


def fp8_gemv_any(x_q: torch.Tensor, w: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x_q: fp8 [M, K] (M small), w: fp8 [N, K], alpha: [1] fp32 -> bf16 [M, N]."""
    m, k = x_q.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x_q.device)
    block_n, block_k = 32, 512
    _fp8_gemv_kernel[(m, triton.cdiv(n, block_n))](x_q, w, alpha, y, x_q.stride(0), w.stride(0), y.stride(0), n, k, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return y
