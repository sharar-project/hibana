"""Weight-only IQ2_XXS/Q8_0/NVFP4 projections with BF16 activations.

Native checkpoint bytes persist; tiles decode directly into BF16 MMA operands.
Routed calls select one expert per pair on the device, including padded rows.
There is no activation quantizer, host routing, expanded matrix, or atomic sum.
The IQ2 codebook/layout is GGML's (see weights/ggml.py and THIRD_PARTY_NOTICES).
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sharar.weights.iq2_grid import GRID


@lru_cache(maxsize=None)
def _grid(device):
    return torch.tensor(GRID, device=device, dtype=torch.uint8).contiguous()


@triton.jit
def _half_base(W, offset, mask):
    lo = tl.load(W + offset, mask, 0).to(tl.uint16)
    hi = tl.load(W + offset + 1, mask, 0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _decode(W, S, Gain, Grid, n, k, mask, K: tl.constexpr, Q: tl.constexpr):
    if Q == 0:  # IQ2_XXS: eight 32-value groups in each 66-byte block
        base = (n * (K // 256) + k // 256) * 66
        d = _half_base(W, base, mask)
        sub = base + 2 + ((k % 256) // 32) * 8
        group = (k % 32) // 8
        index = tl.load(W + sub + group, mask, 0).to(tl.int32)
        control = tl.load(W + sub + 4, mask, 0).to(tl.uint32)
        control |= tl.load(W + sub + 5, mask, 0).to(tl.uint32) << 8
        control |= tl.load(W + sub + 6, mask, 0).to(tl.uint32) << 16
        control |= tl.load(W + sub + 7, mask, 0).to(tl.uint32) << 24
        signs = (control >> (7 * group)) & 127
        parity = signs ^ (signs >> 4)
        parity ^= parity >> 2
        parity ^= parity >> 1
        signs |= (parity & 1) << 7
        magnitude = tl.load(Grid + index * 8 + k % 8).to(tl.float32)
        signed = tl.where(((signs >> (k % 8)) & 1) != 0, -magnitude, magnitude)
        return ((d * (0.5 + (control >> 28).to(tl.float32))) * 0.25 * signed).to(tl.bfloat16)
    elif Q == 1:  # Q8_0: FP16 base and 32 signed bytes
        base = (n * (K // 32) + k // 32) * 34
        d = _half_base(W, base, mask)
        value = tl.load(W + base + 2 + k % 32, mask, 0).to(tl.int8).to(tl.float32)
        return (d * value).to(tl.bfloat16)
    else:  # ModelOpt W4A16 NVFP4, low nibble first, unswizzled E4M3 scales
        packed = tl.load(W + n * (K // 2) + k // 2, mask, 0).to(tl.int32)
        code = (packed >> ((k % 2) * 4)) & 15
        a = code & 7
        magnitude = tl.where(a < 4, a.to(tl.float32) * 0.5,
                             tl.where(a == 4, 2., tl.where(a == 5, 3., tl.where(a == 6, 4., 6.))))
        scale = tl.load(S + n * (K // 16) + k // 16, mask, 0.0).to(tl.float32)
        return (tl.where((code & 8) != 0, -magnitude, magnitude) * scale).to(tl.bfloat16)


@triton.jit
def _linear(X, W, S, Gain, Grid, Ids, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            E: tl.constexpr, WS: tl.constexpr, TOPK: tl.constexpr, REPEAT_INPUT: tl.constexpr,
            ROUTED: tl.constexpr, Q: tl.constexpr, RELU2: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pair = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    if ROUTED:
        expert = tl.load(Ids + pair).to(tl.int64)
        row_mask = rows == 0
        out_rows = pair + rows
        x_rows = pair // TOPK + rows if REPEAT_INPUT else pair + rows
        valid = (expert >= 0) & (expert < E)
        W += tl.where(valid, expert, 0) * WS
    else:
        row_mask = rows < M
        out_rows = rows
        x_rows = rows
        valid = True
    acc = tl.full((BM, BN), 0, tl.float32)
    if valid:
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + tl.arange(0, BK)
            a = tl.load(X + x_rows[:, None] * K + kk[None, :], row_mask[:, None] & (kk[None, :] < K), 0)
            b = _decode(W, S, Gain, Grid, cols[None, :], kk[:, None],
                        (cols[None, :] < N) & (kk[:, None] < K), K, Q)
            acc = tl.dot(a, b, acc)
    if Q == 2:
        acc *= tl.load(Gain)
    value = acc.to(tl.bfloat16)
    if RELU2:
        activated = tl.maximum(value.to(tl.float32), 0.)
        value = (activated * activated).to(tl.bfloat16)
    tl.store(Y + out_rows[:, None] * N + cols[None, :], value, row_mask[:, None] & (cols[None, :] < N))


@triton.jit
def _embedding(Ids, W, Y, Grid, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(Ids + t).to(tl.int64)
    value = _decode(W, W, W, Grid, token, k, k < H, H, 1)
    tl.store(Y + t * H + k, value, k < H)


@triton.jit
def _combine(Pairs, Weights, Out, K: tl.constexpr, H: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for choice in range(K):
        value = tl.load(Pairs + (t * K + choice) * H + col, col < H, 0).to(tl.float32)
        weight = tl.load(Weights + t * K + choice) * SCALE
        total += value * weight
    tl.store(Out + t * H + col, total.to(tl.bfloat16), col < H)


def matmul(x, weight, quant, *, scales=None, gain=None, ids=None, repeat_input=False, relu2=False):
    """Dense [M,K] or routed [E,N,K/256,66] IQ2 weights; invalid expert ids write zeros.

    In routed mode ids is [M,topk]. repeat_input reads x[token] for an up
    projection; otherwise x[pair] for a down projection. Output is pair-major.
    All allocations and launches are capture-safe after warming each shape.
    """
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("weight-only GEMM requires contiguous CUDA BF16 inputs")
    if weight.device != x.device or not weight.is_contiguous() or weight.dtype != torch.uint8:
        raise ValueError("packed weights must be contiguous CUDA uint8 on the input device")
    q = {"IQ2_XXS": 0, "Q8_0": 1, "NVFP4": 2}[quant]
    routed = ids is not None
    if routed and (q != 0 or weight.ndim != 4 or ids.ndim != 2 or ids.device != x.device or not ids.is_contiguous()
                   or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("routed GEMM requires IQ2_XXS experts and contiguous device ids")
    n = weight.shape[1] if routed else weight.shape[0]
    k = x.shape[-1]
    expected = (k // 256, 66) if q == 0 else (k // 32, 34) if q == 1 else (k // 2,)
    if tuple(weight.shape[2:] if routed else weight.shape[1:]) != expected:
        raise ValueError("packed projection shape does not match input width")
    if q == 2 and (scales is None or scales.shape != (n, k // 16) or gain is None):
        raise ValueError("NVFP4 requires per-16 scales and a global gain")
    e, topk = (weight.shape[0], ids.shape[1]) if routed else (1, 1)
    rows = ids.numel() if routed else x.shape[0]
    expected_rows = ids.shape[0] if routed and repeat_input else rows
    if x.ndim != 2 or x.shape[0] != expected_rows:
        raise ValueError("input rows do not match routed pairs")
    out = torch.empty(rows, n, device=x.device, dtype=torch.bfloat16)
    if rows:
        _linear[(1 if routed else triton.cdiv(rows, 16), triton.cdiv(n, 64), rows if routed else 1)](
            x, weight, scales if q == 2 else weight, gain if q == 2 else weight, _grid(x.device), ids if routed else weight,
            out, rows, n, k, e, weight.stride(0) if routed else 0, topk, repeat_input, routed, q, relu2,
            16, 64, 64, num_warps=4)
    return out


def embedding(ids, weight):
    out = torch.empty(ids.numel(), weight.shape[1] * 32, device=ids.device, dtype=torch.bfloat16)
    if ids.numel():
        _embedding[(ids.numel(), triton.cdiv(out.shape[1], 256))](ids, weight, out, _grid(ids.device), out.shape[1], 256)
    return out


def combine(pairs, weights, scale=5.0):
    m, k = weights.shape
    out = torch.empty(m, pairs.shape[-1], device=pairs.device, dtype=torch.bfloat16)
    if m:
        _combine[(m, triton.cdiv(out.shape[1], 256))](pairs, weights, out, k, out.shape[1], scale, 256)
    return out
