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
def _grouped_norm_kernel(X, W, Y, Amax, stride_x, stride_y, H: tl.constexpr, eps, BLOCK: tl.constexpr, AMAX: tl.constexpr):
    t = tl.program_id(0)
    s = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = s * H + cols
    x = tl.load(X + t * stride_x + off, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, 0) / H
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(var + eps) * (1.0 + w)
    yb = y.to(tl.bfloat16)
    tl.store(Y + t * stride_y + off, yb, mask=mask)
    if AMAX:  # this block's max |y| as stored, for a parallel per-tensor FP8 quantization of y (`quant_fp8_parts`)
        tl.store(Amax + t * tl.num_programs(1) + s, tl.max(tl.where(mask, tl.abs(yb.to(tl.float32)), 0.0), 0))


def grouped_gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int, amax: bool = False):
    """[T, G*group] -> bf16 [T, G*group]; RMS per `group` columns, one (1 + w) per column. With `amax`, also
    returns each (row, group) block's max |y| [T*G] fp32, for `quant_fp8_parts`."""
    t, n = x.shape
    g = n // group
    y = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
    parts = torch.empty((t * g,), dtype=torch.float32, device=x.device) if amax else y
    _grouped_norm_kernel[(t, g)](x, weight, y, parts, x.stride(0), y.stride(0), group, eps, BLOCK=triton.next_power_of_2(group), AMAX=amax, num_warps=8)
    return (y, parts) if amax else y


def quant_fp8_parts(x: torch.Tensor, parts: torch.Tensor):
    """Per-tensor dynamic FP8 of a contiguous x whose block maxima of |x| its producer already wrote: the bits of
    `quant_fp8_small` / `draft_fp8.quant_dynamic` from one parallel launch. -> (fp8 like x, scale [1, 1])."""
    from sharar.kernels.draft_fp8 import _pack
    n = x.numel()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1), device=x.device, dtype=torch.float32)
    _pack[(triton.cdiv(n, 2048),)](x, parts, out, scale, n, parts.numel(), triton.next_power_of_2(parts.numel()), 2048)
    return out, scale


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


# ---- stream mean: y[t, c] = mean_s hyper[t, s*H + c] (a layer's output as a block drafter reads it) ----
@triton.jit
def _stream_mean_kernel(X, Y, stride_x, stride_y, H: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(HC):
        acc += tl.load(X + t * stride_x + s * H + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + cols, (acc / HC).to(tl.bfloat16), mask=mask)


def stream_mean(hyper: torch.Tensor, hc: int, hidden: int) -> torch.Tensor:
    """[T, hc*H] -> bf16 [T, H], the FP32 mean of the streams: `hyper.unflatten(-1, (hc, H)).float().mean(-2)` in one
    launch, without its FP32 [T, hc, H] temporary."""
    t = hyper.shape[0]
    y = torch.empty((t, hidden), dtype=torch.bfloat16, device=hyper.device)
    block = 1024
    _stream_mean_kernel[(t, triton.cdiv(hidden, block))](hyper, y, hyper.stride(0), y.stride(0), hidden, hc, BLOCK=block)
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


# ---- residual injection, then the next hyper-connection's grouped norm of its result (one pass) ----
@triton.jit
def _inject_norm_kernel(R, B, I, Y, W, N, Amax, stride_r, stride_b, stride_i, stride_y, stride_n, H: tl.constexpr, eps,
                        BLOCK: tl.constexpr):
    t = tl.program_id(0)
    s = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = s * H + cols
    g = 2.0 * _sigmoid(tl.load(I + t * stride_i + s).to(tl.float32))
    r = tl.load(R + t * stride_r + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + t * stride_b + cols, mask=mask, other=0.0).to(tl.float32)
    hb = (r + b * g).to(tl.bfloat16)
    tl.store(Y + t * stride_y + off, hb, mask=mask)
    # `_grouped_norm_kernel` on the stored values: the same block, warps and reduction, so the same bits
    x = hb.to(tl.float32)
    var = tl.sum(x * x, 0) / H
    w = tl.load(W + off, mask=mask, other=0.0).to(tl.float32)
    yb = (x * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16)
    tl.store(N + t * stride_n + off, yb, mask=mask)
    tl.store(Amax + t * tl.num_programs(1) + s, tl.max(tl.where(mask, tl.abs(yb.to(tl.float32)), 0.0), 0))


def inject_combine_norm(hyper: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, hc: int, hidden: int, weight: torch.Tensor, eps: float):
    """`inject_combine`, then `grouped_gemma_rms(..., amax=True)` of its result with `weight`, in one pass (the
    result is not read back): -> (hyper', normed, block maxima), each bitwise the two kernels'."""
    t = hyper.shape[0]
    y = torch.empty_like(hyper)
    normed = torch.empty_like(hyper)
    parts = torch.empty((t * hc,), dtype=torch.float32, device=hyper.device)
    _inject_norm_kernel[(t, hc)](hyper, block, inj, y, weight, normed, parts, hyper.stride(0), block.stride(0), inj.stride(0),
                                 y.stride(0), normed.stride(0), hidden, eps, BLOCK=triton.next_power_of_2(hidden), num_warps=8)
    return y, normed, parts


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
def _route_kernel(L, Slots, Ids, W, stride_l, E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr,
                  HAS_SLOTS: tl.constexpr, REPEAT: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    logits = tl.load(L + t * stride_l + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(logits, 0)
    p = tl.exp(logits - m)
    p = p / tl.sum(p, 0)
    p = tl.where(mask, p, -1.0)
    real = True
    fill = E  # a padded row (a graph's dummy slot 0) reads no expert the call would not read anyway
    if HAS_SLOTS:
        real = tl.load(Slots + t // STEPS) != 0
        if REPEAT:  # row 0's top 1 (row 0 is real: a graph's padding follows its requests)
            fill = tl.argmax(tl.load(L + cols, mask=mask, other=float("-inf")).to(tl.float32), 0).to(tl.int32)
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
        tl.store(Ids + t * K + k, tl.where(real, idx.to(tl.int32), fill))
        tl.store(W + t * K + k, tl.where(real, val / total, 0.0))
        work = tl.where(cols == idx, -1.0, work)


def route_topk(logits: torch.Tensor, k: int, slots: torch.Tensor | None = None, steps: int = 1, repeat: bool = False):
    """[M, E] -> (int32 ids [M, k], fp32 renormalized softmax weights [M, k]), largest first. With `slots`, row r is real
    iff slots[r // steps] != 0; a padded row (a graph's dummy slot 0) takes expert E at weight 0, which the 3-bit experts
    skip, or with `repeat` row 0's top 1 at weight 0 (FlashInfer's NVFP4 MoE needs a real expert: no further read)."""
    m, e = logits.shape
    ids = torch.empty((m, k), dtype=torch.int32, device=logits.device)
    w = torch.empty((m, k), dtype=torch.float32, device=logits.device)
    _route_kernel[(m,)](logits, ids if slots is None else slots, ids, w, logits.stride(0), e, k, BLOCK=triton.next_power_of_2(e),
                        STEPS=steps, HAS_SLOTS=slots is not None, REPEAT=repeat)
    return ids, w


# ---- budgeted expert selection (SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET): a cap on the distinct experts a decode or verify
# forward reads, chosen for all its rows; each row takes its guaranteed top `keep`, then its other candidates in the set ----
ROUTE_CANDIDATES = 20  # a row takes its experts from its router ranks 1..20


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _route_rank_kernel(L, Slots, Cand, P, Bits, Dense, stride_l, E: tl.constexpr, BLOCK: tl.constexpr, KEEP: tl.constexpr,
                       C: tl.constexpr, CP: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr, K: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    logits = tl.load(L + t * stride_l + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(logits, 0)
    s = tl.sum(tl.exp(logits - m), 0)
    # rank key, distinct per column: the bf16 bits as an order-preserving int (-0.0 as +0.0) above BLOCK - 1 - column,
    # so the int32 maximum is the largest logit and, among equal logits, the lowest expert
    hi = logits.to(tl.int32, bitcast=True) >> 16
    hi = tl.where(hi == -32768, 0, hi)
    key = tl.where(mask, (hi ^ ((hi >> 15) & 0x7FFF)) * 65536 + (BLOCK - 1 - cols), -2147483648)
    j = tl.arange(0, CP)
    ids = tl.zeros([CP], dtype=tl.int32)
    vals = tl.full([CP], float("-inf"), tl.float32)
    w = tl.arange(0, BLOCK // 32)
    bits = tl.zeros([BLOCK // 32], dtype=tl.int32)
    dense = tl.zeros([BLOCK], dtype=tl.float32)
    for c in tl.static_range(C):  # rank c: the largest key left
        top = tl.max(key, 0)
        e = BLOCK - 1 - (top & 0xFFFF)
        o = top >> 16
        v = ((o ^ ((o >> 15) & 0x7FFF)) << 16).to(tl.float32, bitcast=True)  # its logit, exactly
        ids = tl.where(j == c, e, ids)
        vals = tl.where(j == c, v, vals)
        if c < K:
            dense = tl.where(cols == e, tl.exp(v - m) / s, dense)
        if c < KEEP:
            bits = tl.where(w == e // 32, bits | (1 << (e % 32)), bits)
        key = tl.where(cols == e, -2147483648, key)
    p = tl.exp(vals - m) / s  # route_topk's p of each candidate (0 past rank C)
    tl.store(Cand + t * CP + j, ids)
    tl.store(P + t * CP + j, p)
    real = True
    if HAS_SLOTS:  # a padded row (dummy slot 0) adds nothing to the core or the scores
        real = tl.load(Slots + t // STEPS) != 0
        bits = tl.where(real, bits, 0)
    tl.store(Bits + t * (BLOCK // 32) + w, bits)
    tl.store(Dense + t * BLOCK + cols, tl.where(real, dense, 0.0))  # its p within its top K by expert, for the scores


@triton.jit
def _route_pick_kernel(Slots, Cand, P, Set, Ids, W, R, K: tl.constexpr, C: tl.constexpr,
                       CP: tl.constexpr, NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr,
                       HAS_SLOTS: tl.constexpr, KP: tl.constexpr, NONE: tl.constexpr, REPEAT: tl.constexpr):
    rr = tl.arange(0, RB)
    core = tl.load(Set + tl.arange(0, NW))  # the call's set, from `_budget_select_kernel`
    rows = tl.program_id(0) * RB + rr
    live = rows < R
    j = tl.arange(0, CP)[None, :]
    cand = tl.load(Cand + rows[:, None] * CP + j, mask=live[:, None], other=0)
    p = tl.load(P + rows[:, None] * CP + j, mask=live[:, None], other=1.0)  # rows past R: no 0 / 0
    word = tl.reshape(tl.gather(core, tl.reshape(cand // 32, [RB * CP]), 0), [RB, CP])
    pull = (j < C) & (((word >> (cand % 32)) & 1) != 0)  # a candidate in the set (the row's guaranteed ones are)
    real = tl.full([RB], 1, tl.int1)
    if HAS_SLOTS:  # a padded row takes no expert
        real = tl.load(Slots + rows // STEPS, mask=live, other=0) != 0
    a = pull.to(tl.int32)
    slot = tl.cumsum(a, 1) - 1  # output slot of each rank: the candidates in the set, in rank order
    chosen = pull & (slot < K) & real[:, None]
    used = tl.where(real, tl.minimum(tl.sum(a, 1), K), 0)
    total = tl.sum(tl.where(chosen, p, 0.0), 1)
    smask = live[:, None] & chosen
    tl.store(Ids + rows[:, None] * K + slot, cand, mask=smask)
    tl.store(W + rows[:, None] * K + slot, p / total[:, None], mask=smask)
    # the slots left over name expert NONE (= E) at weight 0: the w3 kernel reads nothing for them
    q = tl.arange(0, KP)[None, :]
    hole = live[:, None] & (q >= used[:, None]) & (q < K)
    if REPEAT:  # an expert the call already reads, at weight 0: the row's top 1, a padded row row 0's
        first = tl.where(real, tl.load(Cand + rows * CP, mask=live, other=0), tl.load(Cand))
        fill_id = tl.broadcast_to(first[:, None], [RB, KP])
    else:
        fill_id = tl.full([RB, KP], NONE, tl.int32)
    tl.store(Ids + rows[:, None] * K + q, fill_id, mask=hole)
    tl.store(W + rows[:, None] * K + q, tl.zeros([RB, KP], dtype=tl.float32), mask=hole)


@triton.jit
def _budget_score_kernel(Dense, Score, R, BLOCK: tl.constexpr, EB: tl.constexpr, RB: tl.constexpr):
    """Score[e]: the router probability expert e takes within the top K of the call's real rows (`_route_rank_kernel`'s
    dense rows), summed in row order (one program per EB experts; no atomics, so the same bits on every run)."""
    ex = tl.program_id(0) * EB + tl.arange(0, EB)
    rr = tl.arange(0, RB)
    acc = tl.zeros([EB], dtype=tl.float32)
    for r0 in range(0, R, RB):  # RB rows a load, added one row at a time (rows past R add 0)
        tile = tl.load(Dense + (r0 + rr)[:, None] * BLOCK + ex[None, :], mask=(r0 + rr)[:, None] < R, other=0.0)
        for i in tl.static_range(RB):
            acc += tl.sum(tl.where(rr[:, None] == i, tile, 0.0), 0)
    tl.store(Score + ex, acc)


@triton.jit
def _budget_select_kernel(Score, Bits, Slots, Set, R, Budget, PerRequest, MinCap, E: tl.constexpr, BLOCK: tl.constexpr,
                          NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr):
    """Word `program_id` of the selected set: every real row's top KEEP (the OR of Bits), then the highest-scoring other
    experts up to the cap in all (equal scores: the lower expert). The cap is Budget, or with PerRequest,
    min(Budget, max(MinCap, PerRequest x the call's real requests)). One program per 32 experts, each ranking its own
    against all."""
    g = tl.program_id(0)
    if PerRequest > 0:
        reqs = R // STEPS
        if HAS_SLOTS:  # a graph's padded requests (dummy slot 0) do not count
            reqs = 0
            for q0 in range(0, R // STEPS, RB):
                q = q0 + tl.arange(0, RB)
                reqs += tl.sum((tl.load(Slots + q, mask=q < R // STEPS, other=0) != 0).to(tl.int32), 0)
        Budget = tl.minimum(Budget, tl.maximum(MinCap, PerRequest * reqs))
    w = tl.arange(0, NW)
    rr = tl.arange(0, RB)
    core = tl.zeros([NW], dtype=tl.int32)
    for r0 in range(0, R, RB):
        b = tl.load(Bits + (r0 + rr)[:, None] * NW + w[None, :], mask=(r0 + rr)[:, None] < R, other=0)
        core = core | tl.reduce(b, 0, _or)
    cols = tl.arange(0, BLOCK)
    incore = ((tl.gather(core, cols // 32, 0) >> (cols % 32)) & 1) != 0
    s = tl.load(Score + cols, mask=cols < E, other=0.0)
    key = tl.where(incore | (s <= 0.0) | (cols >= E), -1.0, s)
    need = Budget - tl.sum(incore.to(tl.int32), 0)
    f = g * 32 + tl.arange(0, 32)  # this program's experts
    sf = tl.load(Score + f, mask=f < E, other=0.0)
    inf = ((tl.gather(core, f // 32, 0) >> (f % 32)) & 1) != 0
    kf = tl.where(inf | (sf <= 0.0) | (f >= E), -1.0, sf)
    ahead = (key[None, :] > kf[:, None]) | ((key[None, :] == kf[:, None]) & (cols[None, :] < f[:, None]))
    rank = tl.sum(ahead.to(tl.int32), 1)  # other experts ahead of each: higher score, or equal and lower id
    sel = inf | ((kf > 0.0) & (rank < need))
    tl.store(Set + g, tl.sum(tl.where(sel, 1 << (f % 32), 0), 0))  # distinct bits: the sum is the OR


def route_budget(logits: torch.Tensor, k: int, keep: int, budget: int, slots: torch.Tensor | None = None, steps: int = 1,
                 repeat: bool = False, per_request: int = 0, min_cap: int = 0):
    """Budgeted top-k: logits bf16 [R, E] contiguous -> (int32 ids [R, k], fp32 weights [R, k]). The experts the call
    reads are chosen for the whole call, under a cap:
      - the set: every real row's top `keep` (guaranteed), then the experts with the most router probability within the
        real rows' top k, summed over the rows, until it holds the cap (more only when the guaranteed experts alone
        exceed it). The cap is `budget`, or with `per_request`, min(budget, max(min_cap, per_request x the real
        requests));
      - each real row takes its candidates (its ROUTE_CANDIDATES largest logits, largest first, an exact logit tie to
        the lower expert) that are in the set, in rank order, up to k: its top `keep` first.
    Slots left over (all of a padded row's) hold expert id E at weight 0, which the 3-bit experts skip; with `repeat`
    they name the row's top 1 (a padded row: row 0's) at weight 0, for kernels that need a real expert in every slot
    (FlashInfer's NVFP4 MoE: the same bits, no further expert read). Row r is real iff slots[r // steps] != 0 (slot 0
    is the dummy padded graph rows use); without `slots` every row is. Weights: the fp32 softmax over E at the chosen
    experts, renormalized over them. Four launches, no atomics (the same bits on every run), no host sync:
    graph-capturable."""
    r, e = logits.shape
    block, cp = triton.next_power_of_2(e), triton.next_power_of_2(ROUTE_CANDIDATES)
    nw = block // 32
    dev = logits.device
    cand = torch.empty((r, cp), dtype=torch.int32, device=dev)
    p = torch.empty((r, cp), dtype=torch.float32, device=dev)
    bits = torch.empty((r, nw), dtype=torch.int32, device=dev)
    dense = torch.empty((r, block), dtype=torch.float32, device=dev)
    score = torch.empty((block,), dtype=torch.float32, device=dev)
    chosen = torch.empty((nw,), dtype=torch.int32, device=dev)
    ids = torch.empty((r, k), dtype=torch.int32, device=dev)
    w = torch.empty((r, k), dtype=torch.float32, device=dev)
    sl = cand if slots is None else slots
    common = dict(C=ROUTE_CANDIDATES, CP=cp, STEPS=steps, HAS_SLOTS=slots is not None)
    # one warp per row: each of the ROUTE_CANDIDATES rank passes reduces with warp shuffles alone
    _route_rank_kernel[(r,)](logits, sl, cand, p, bits, dense, logits.stride(0), e, BLOCK=block, KEEP=keep, K=k,
                             num_warps=1, **common)
    _budget_score_kernel[(block // 32,)](dense, score, r, BLOCK=block, EB=32, RB=16, num_warps=1)  # 16 rows a load
    rb = min(64, max(16, triton.next_power_of_2(r)))
    _budget_select_kernel[(nw,)](score, bits, sl, chosen, r, budget, per_request, min_cap, E=e, BLOCK=block, NW=nw, RB=rb, STEPS=steps,
                                 HAS_SLOTS=slots is not None, num_warps=4)
    # the rows take their candidates in the set in rank order: a row's top `keep` are in it, so they come first
    _route_pick_kernel[(triton.cdiv(r, rb),)](sl, cand, p, chosen, ids, w, r, k, NW=nw, RB=rb, num_warps=4,
                                              KP=triton.next_power_of_2(k), NONE=e, REPEAT=repeat, **common)
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


# ---- the 3-bit experts' routed sum + MoE combine: `w3_moe`'s combine, then `moe_combine`, in one pass ----
@triton.jit
def _moe_combine_pairs_kernel(Yp, Ids, Wt, S, X, G, Y, e_lo, e_n, stride_yp, stride_s, stride_x, stride_y, H: tl.constexpr,
                              TOPK: tl.constexpr, BLOCK: tl.constexpr, DBLOCK: tl.constexpr):
    t = tl.program_id(0)
    dot = tl.zeros([DBLOCK], dtype=tl.float32)
    for start in tl.static_range(0, H, DBLOCK):
        i = start + tl.arange(0, DBLOCK)
        dot += tl.load(X + t * stride_x + i, mask=i < H, other=0.0).to(tl.float32) * tl.load(G + i, mask=i < H, other=0.0).to(tl.float32)
    g = _sigmoid(tl.sum(dot, 0))
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    # csrc/w3_gemm.cu's combine_kernel: v += w_k * y_k over this rank's pairs in slot order, one FMA each (it is
    # compiled with contraction), rounded to BF16 once
    v = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(Ids + t * TOPK + k) - e_lo
        if (e >= 0) & (e < e_n):
            yk = tl.load(Yp + (t * TOPK + k).to(tl.int64) * stride_yp + cols, mask=mask, other=0.0).to(tl.float32)  # FP32 or BF16 pairs
            v = tl.fma(tl.full([BLOCK], 0.0, tl.float32) + tl.load(Wt + t * TOPK + k), yk, v)
    r = v.to(tl.bfloat16).to(tl.float32)
    s = tl.load(S + t * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + cols, (r + g * s).to(tl.bfloat16), mask=mask)


def moe_combine_pairs(y: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, shared: torch.Tensor, x: torch.Tensor, gate_w: torch.Tensor,
                      e_lo: int = 0, e_n: int = 512) -> torch.Tensor:
    """`moe_combine(w3_moe(...), shared, x, gate_w)` from `w3_moe(..., pairs=True)`'s FP32 pair outputs y [m*topk, H], bit
    for bit, without writing the routed sum: ids int32 / weights fp32 [m, topk]; pairs whose expert is outside
    e_lo .. e_lo + e_n - 1 add nothing."""
    m, topk = ids.shape
    h = y.shape[1]
    out = torch.empty((m, h), dtype=torch.bfloat16, device=y.device)
    blk = 1024
    _moe_combine_pairs_kernel[(m, triton.cdiv(h, blk))](y, ids, weights.contiguous(), shared, x, gate_w, out, e_lo, e_n, y.stride(0), shared.stride(0),
                                                        x.stride(0), out.stride(0), h, topk, BLOCK=blk, DBLOCK=1024)
    return out


# ---- routed sum + MoE combine: out = bf16(sum_k w_k * y_k) + sigmoid(x . g) * shared (the MTP head's experts) ----
@triton.jit
def _moe_combine_routed_kernel(Yr, Wt, S, X, G, Y, stride_yr, stride_s, stride_x, stride_y, H: tl.constexpr, TOPK: tl.constexpr,
                               BLOCK: tl.constexpr, DBLOCK: tl.constexpr):
    t = tl.program_id(0)
    dot = tl.zeros([DBLOCK], dtype=tl.float32)
    for start in tl.static_range(0, H, DBLOCK):
        i = start + tl.arange(0, DBLOCK)
        dot += tl.load(X + t * stride_x + i, mask=i < H, other=0.0).to(tl.float32) * tl.load(G + i, mask=i < H, other=0.0).to(tl.float32)
    g = _sigmoid(tl.sum(dot, 0))
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    # (y.float() * w).view(m, topk, H).sum(1) as PyTorch reduces it: four interleaved accumulators, then
    # ((a0 + a1) + a2) + a3 (tests/gpu/test_flashnext_fused.py checks the bits). Each product is rounded on its own
    # (`mul.rn`, which the compiler cannot contract into an FMA with the accumulation), as PyTorch's separate kernels do.
    a0 = tl.zeros([BLOCK], dtype=tl.float32)
    a1 = tl.zeros([BLOCK], dtype=tl.float32)
    a2 = tl.zeros([BLOCK], dtype=tl.float32)
    a3 = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        yk = tl.load(Yr + (t * TOPK + k) * stride_yr + cols, mask=mask, other=0.0).to(tl.float32)
        wk = tl.full([BLOCK], 0.0, tl.float32) + tl.load(Wt + t * TOPK + k)
        p = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [yk, wk], dtype=tl.float32, is_pure=True, pack=1)
        if k % 4 == 0:
            a0 += p
        elif k % 4 == 1:
            a1 += p
        elif k % 4 == 2:
            a2 += p
        else:
            a3 += p
    r = (((a0 + a1) + a2) + a3).to(tl.bfloat16).to(tl.float32)
    s = tl.load(S + t * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + t * stride_y + cols, (r + g * s).to(tl.bfloat16), mask=mask)


def moe_combine_routed(y: torch.Tensor, weights: torch.Tensor, shared: torch.Tensor, x: torch.Tensor, gate_w: torch.Tensor) -> torch.Tensor:
    """`moe_combine((y.float() * weights.reshape(-1)[:, None]).view(m, topk, -1).sum(1).to(bf16), shared, x, gate_w)` in
    one launch, bit for bit: y bf16 [m*topk, H] routed expert outputs, weights fp32 [m, topk]."""
    m, topk = weights.shape
    h = y.shape[1]
    out = torch.empty((m, h), dtype=torch.bfloat16, device=y.device)
    blk = 1024
    _moe_combine_routed_kernel[(m, triton.cdiv(h, blk))](y, weights.contiguous(), shared, x, gate_w, out, y.stride(0), shared.stride(0), x.stride(0), out.stride(0),
                                                         h, topk, BLOCK=blk, DBLOCK=1024)
    return out


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


# ---- draft-step fusions: the arithmetic of the unfused chains above, fewer launches ----
# Each kernel below reproduces, bit for bit, a chain the MTP draft step (and the trunk's small-row calls) ran as
# separate launches; tests/gpu/test_flashnext_fused.py compares them bitwise with the unfused chains.

ACT_NONE, ACT_SILU, ACT_SILU_MUL = 0, 1, 2


@triton.jit
def _silu_torch(x):
    # torch.nn.functional.silu on CUDA: x / (1 + exp(-x)) in FP32 with the precise expf and an IEEE division
    return tl.div_rn(x, 1.0 + tl.extra.cuda.libdevice.exp(-x))


@triton.jit
def _act_load(X, base, cols, cmask, HALF: tl.constexpr, ACT: tl.constexpr):
    """One chunk of an activation row as the BF16 tensor the unfused chain materialized, in FP32:
    ACT_NONE the row itself, ACT_SILU F.silu of it, ACT_SILU_MUL `silu_mul` of a [gate | up] row (HALF = I)."""
    if ACT == 2:
        g = tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)
        u = tl.load(X + base + HALF + cols, mask=cmask, other=0.0).to(tl.float32)
        x = (g * _sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
    elif ACT == 1:
        x = _silu_torch(tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(X + base + cols, mask=cmask, other=0.0).to(tl.float32)
    return x


@triton.jit
def _fp8_gemv_quant_kernel(X, W, WScale, Y, ScaleOut, Parts, M, NPARTS, stride_x, stride_w, stride_y, N: tl.constexpr, K: tl.constexpr,
                           ACT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PBLOCK: tl.constexpr,
                           STORE_SCALE: tl.constexpr, HAS_PARTS: tl.constexpr):
    # quant_fp8_small's per-tensor scale over all M rows (a max: order-free; from the producer's block maxima when
    # it wrote them), then `_fp8_gemv_kernel` on the quantized values, recomputed per program from the (small,
    # cached) activation.
    if HAS_PARTS:
        j = tl.arange(0, PBLOCK)
        peak = tl.load(Parts + j, mask=j < NPARTS, other=0.0)
    else:
        peak = tl.zeros([PBLOCK], dtype=tl.float32)
        for m in range(M):
            for p0 in tl.static_range(0, K, PBLOCK):
                pc = p0 + tl.arange(0, PBLOCK)
                peak = tl.maximum(peak, tl.abs(_act_load(X, m * stride_x, pc, pc < K, K, ACT)))
    scale = tl.maximum(tl.max(peak, 0) * (1.0 / 448.0), 1.0e-12)
    alpha = scale * tl.load(WScale).to(tl.float32)
    r = tl.program_id(0)
    if STORE_SCALE:
        if r == 0 and tl.program_id(1) == 0:
            tl.store(ScaleOut, scale)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        w = tl.load(W + rows[:, None] * stride_w + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        xa = _act_load(X, r * stride_x, cols, cmask, K, ACT)
        x = tl.minimum(tl.maximum(tl.div_rn(xa, scale), -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Y + r * stride_y + rows, (acc * alpha).to(tl.bfloat16), mask=rmask)


def fp8_gemv_quant_any(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, act: int = ACT_NONE, scale_out: torch.Tensor | None = None,
                       amax_parts: torch.Tensor | None = None) -> torch.Tensor:
    """`fp8_gemv_any` of `quant_fp8_small(act(x), w_scale)` in one launch, bit for bit: x bf16 [M, K]
    ([M, 2K] = [gate | up] for ACT_SILU_MUL), w fp8 [N, K] with its fp32 scale -> bf16 [M, N]. `amax_parts`: the
    producer's block maxima of |x| (ACT_NONE), which spare every program the pass over x for the scale."""
    m = x.shape[0]
    k = x.shape[1] // 2 if act == ACT_SILU_MUL else x.shape[1]
    n = w.shape[0]
    assert w.shape[1] == k and x.stride(1) == 1 and (amax_parts is None or act == ACT_NONE)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    block_n, block_k = 32, 512
    nparts = amax_parts.numel() if amax_parts is not None else 0
    pblock = triton.next_power_of_2(nparts) if amax_parts is not None else min(4096, triton.next_power_of_2(k))
    _fp8_gemv_quant_kernel[(m, triton.cdiv(n, block_n))](x, w, w_scale, y, scale_out if scale_out is not None else y,
                                                         amax_parts if amax_parts is not None else y, m, nparts, x.stride(0), w.stride(0),
                                                         y.stride(0), n, k, ACT=act, BLOCK_N=block_n, BLOCK_K=block_k, PBLOCK=pblock,
                                                         STORE_SCALE=scale_out is not None, HAS_PARTS=amax_parts is not None, num_warps=4)
    return y


# ---- SiLU-mul into the 3-bit GEMM's per-(row, 32-group) E4M3 input (`w3.quant_groups`, csrc/w3_gemm.cu) ----
@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_ftz(a, b):
    return tl.inline_asm_elementwise("mul.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu_mul_quant_groups_kernel(GU, Q, S, stride_gu, I: tl.constexpr, GB: tl.constexpr, PAIRS: tl.constexpr,
                                  TORCH_CHAIN: tl.constexpr):
    # quant_groups_kernel's arithmetic as nvcc --use_fast_math compiles it: s = max(amax * (1/448), 1e-12),
    # q = e4m3(x * rcp.approx.ftz(s)) with mul.ftz; tests/gpu/test_flashnext_fused.py checks the bits
    t = tl.program_id(0)
    grp = tl.program_id(1) * GB + tl.arange(0, GB)
    cols = grp[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = cols < I
    if PAIRS:  # gate and up interleaved in blocks of PAIRS columns: output column c is gate column c's block's
        gcol = (cols // PAIRS) * (2 * PAIRS) + cols % PAIRS
        ucol = gcol + PAIRS
    else:
        gcol = cols
        ucol = cols + I
    g = tl.load(GU + t * stride_gu + gcol, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + t * stride_gu + ucol, mask=mask, other=0.0).to(tl.float32)
    if TORCH_CHAIN:  # ops.silu_and_mul: F.silu rounded to BF16, then the BF16 product
        x = (_silu_torch(g).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
    else:
        x = (g * _sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    s = tl.maximum(_mul_ftz(amax, tl.full(amax.shape, 1.0 / 448.0, tl.float32)), 1.0e-12)
    inv = _rcp_approx_ftz(s)
    tl.store(S + t * (I // 32) + grp, s, mask=grp < I // 32)
    tl.store(Q + t * I + cols, _mul_ftz(x, inv[:, None]).to(tl.float8e4nv), mask=mask)


def silu_mul_quant_groups(gu: torch.Tensor, pairs: int = 0, torch_chain: bool = False):
    """`w3.quant_groups(silu_mul(gu))` in one launch, bit for bit: gu bf16 [M, 2I] -> (e4m3 [M, I], fp32 scales [M, I/32]).
    `pairs`: gate and up interleaved in blocks of that many columns (`ops.silu_and_mul(gu, pairs)`); `torch_chain`: the
    product as `ops.silu_and_mul` rounds it (F.silu to BF16, then the product), the 27B MLP's 3-bit decode input."""
    m, n = gu.shape
    i = n // 2
    assert i % 32 == 0 and gu.stride(1) == 1 and (not pairs or (pairs % 32 == 0 and i % pairs == 0))
    q = torch.empty((m, i), dtype=torch.float8_e4m3fn, device=gu.device)
    s = torch.empty((m, i // 32), dtype=torch.float32, device=gu.device)
    gb = 8
    _silu_mul_quant_groups_kernel[(m, triton.cdiv(i // 32, gb))](gu, q, s, gu.stride(0), i, GB=gb, PAIRS=pairs,
                                                                   TORCH_CHAIN=torch_chain, num_warps=2)
    return q, s


# ---- expert GEMV with the routed rows' input by index (no repeat_interleave) and the down projection's SiLU-mul ----
@triton.jit
def _expert_gemv_fused_kernel(W, S, X, IDX, Y, stride_we, stride_wn, stride_x, stride_y, N: tl.constexpr, K: tl.constexpr,
                              X_DIV: tl.constexpr, ACT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    r = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    e = tl.load(IDX + r).to(tl.int64)
    xr = r // X_DIV
    rows = n0 + tl.arange(0, BLOCK_N)
    rmask = rows < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        cmask = cols < K
        w = tl.load(W + e * stride_we + rows[:, None] * stride_wn + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        x = _act_load(X, xr * stride_x, cols, cmask, K, ACT)
        acc += tl.sum(w * x[None, :], 1)
    s = tl.load(S + e).to(tl.float32)
    tl.store(Y + r * stride_y + rows, (acc * s).to(tl.bfloat16), mask=rmask)


def expert_gemv_fused(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, idx: torch.Tensor, x_div: int = 1, act: int = ACT_NONE) -> torch.Tensor:
    """`expert_gemv(w, scale, act(x).repeat_interleave(x_div, 0), idx)` bit for bit: routed row r reads input row
    r // x_div; ACT_SILU_MUL applies `silu_mul` to a [gate | up] input row on the fly. (Rows that chose the same
    expert already share its weights through L2: a program per expert serving all of them was slower.)"""
    e, n, k = w.shape
    r = idx.numel()
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    block_n, block_k = 64, 256
    _expert_gemv_fused_kernel[(r, triton.cdiv(n, block_n))](w, scale, x, idx, y, w.stride(0), w.stride(1), x.stride(0), y.stride(0), n, k,
                                                             X_DIV=x_div, ACT=act, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return y


# ---- attention q/k norm + RoPE, v, and the KV write in one launch (the draft layer's rows) ----
@triton.jit
def _qkv_epilogue_kernel(QKV, CosSin, Pos, QW, KW, Q, KC, VC, Pages, stride_t, stride_qt, stride_qh, stride_kc, stride_kh, eps,
                         HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, HALF: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, D)
    page = tl.load(Pages + t).to(tl.int64)
    if h >= HQ + HKV:  # v: a copy into the cache
        hv = h - HQ - HKV
        v = tl.load(QKV + t * stride_t + HQ * 2 * D + HKV * D + hv * D + cols)
        tl.store(VC + page * stride_kc + hv * stride_kh + cols, v)
    else:
        # `qk_norm_rope._norm_rope_kernel`, its arithmetic unchanged; q to its own tensor, k into the cache
        if h < HQ:
            base = QKV + t * stride_t + h * 2 * D
            w_ptr = QW
            out = Q + t * stride_qt + h * stride_qh
        else:
            base = QKV + t * stride_t + HQ * 2 * D + (h - HQ) * D
            w_ptr = KW
            out = KC + page * stride_kc + (h - HQ) * stride_kh
        x = tl.load(base + cols).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        y = (x * rstd * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
        tl.store(out + cols, y.to(tl.bfloat16), mask=cols >= R)
        lane = tl.arange(0, HALF)
        pos = tl.load(Pos + t)
        cos = tl.load(CosSin + pos * R + lane).to(tl.float32)
        sin = tl.load(CosSin + pos * R + HALF + lane).to(tl.float32)
        x1 = tl.load(base + lane).to(tl.float32)
        x2 = tl.load(base + HALF + lane).to(tl.float32)
        w1 = tl.load(w_ptr + lane).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + lane).to(tl.float32)
        y1 = (x1 * rstd * (1.0 + w1)).to(tl.bfloat16).to(tl.float32)
        y2 = (x2 * rstd * (1.0 + w2)).to(tl.bfloat16).to(tl.float32)
        o1 = y1 * cos - y2 * sin
        o2 = y2 * cos + y1 * sin
        tl.store(out + lane, o1.to(tl.bfloat16))
        tl.store(out + HALF + lane, o2.to(tl.bfloat16))


def qkv_epilogue(qkv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                 k_cache: torch.Tensor, v_cache: torch.Tensor, pages: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int,
                 rotary_dim: int, eps: float) -> torch.Tensor:
    """`qk_norm_rope` plus `KvPool.write` (page size 1) of k and v in one launch: returns q [T, Hq, D]; k and v land
    in the caches at [pages, 0]. qkv rows are [head0: q | gate, ..., k heads, v heads] with unit column stride."""
    t = qkv.shape[0]
    assert qkv.stride(1) == 1 and k_cache.stride(-1) == 1 and k_cache.stride() == v_cache.stride() and k_cache.dtype == torch.bfloat16
    q = torch.empty((t, num_heads, head_dim), dtype=torch.bfloat16, device=qkv.device)
    _qkv_epilogue_kernel[(t, num_heads + 2 * num_kv_heads)](
        qkv, cos_sin, positions, q_weight, k_weight, q, k_cache, v_cache, pages, qkv.stride(0), q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(2), eps, HQ=num_heads, HKV=num_kv_heads, D=head_dim, R=rotary_dim, HALF=rotary_dim // 2,
        num_warps=4)
    return q
