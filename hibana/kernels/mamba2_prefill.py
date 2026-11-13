# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Derived from mamba2.py's chunk scan, adapted from vLLM's Mamba ops and mamba_ssm v2.2.4.
"""GB10 grouped Mamba prefill scan with gate and RMSNorm in the epilogue.

A CTA owns the 16 heads sharing B/C and 32 output tokens. Tiling the entering
state avoids staging the group's complete state; shared C/CB operands and the
on-chip gate/norm remove repeated reads and the global scan-output round trip.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scan_norm_kernel(CB, X, Cm, sx, Dt, DA, Entering, Dskip, CuChunks,
                      Z, sz, W, Out, so, eps, nchunks,
                      H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
                      G: tl.constexpr, CS: tl.constexpr, BM: tl.constexpr,
                      BK: tl.constexpr, STAGES: tl.constexpr, NK: tl.constexpr):
    HH: tl.constexpr = H // G
    tl.static_assert(HH == 16 and P == 64 and N == 128 and CS == 128)
    mi, c, hg = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    h = hg * HH + tl.arange(0, HH)
    g = hg
    t0 = tl.load(CuChunks + c).to(tl.int64)
    n = tl.load(CuChunks + c + 1) - t0
    if mi * BM >= n:
        return
    om = mi * BM + tl.arange(0, BM)
    on = tl.arange(0, P)
    od = tl.arange(0, NK)
    hc = h.to(tl.int64) * nchunks * CS + c * CS
    dm = tl.load(DA + hc[:, None] + om[None, :])
    acc = tl.full((HH, BM, P), 0., tl.float32)
    for nk in range(N // NK):
        cm = tl.load(Cm + (t0 + om)[:, None] * sx + g * N + nk * NK + od[None, :],
                     (om < n)[:, None], 0)
        prev = tl.load(Entering + ((c * H + h[:, None, None]) * P
                                  + on[None, None, :]) * N + nk * NK + od[None, :, None])
        acc = tl.dot(tl.broadcast_to(cm[None, :, :], (HH, BM, NK)), prev, acc)
    acc *= tl.exp(dm)[:, :, None]
    ok = tl.arange(0, BK)
    kmax = tl.minimum((mi + 1) * BM, n)
    for k in tl.range(0, kmax, BK, num_stages=STAGES):
        raw = tl.load(CB + ((c * G + g) * CS + om[:, None]) * CS + k + ok[None, :])
        dk = tl.load(DA + hc[:, None] + k + ok[None, :])
        dt = tl.load(Dt + hc[:, None] + k + ok[None, :])
        cb = raw[None, :, :] * (tl.exp(tl.minimum(dm[:, :, None] - dk[:, None, :], 0.)) * dt[:, None, :])
        cb = tl.where(om[None, :, None] >= k + ok[None, None, :], cb, 0.)
        x = tl.load(X + (t0 + k + ok[None, :, None]) * sx
                    + h[:, None, None] * P + on[None, None, :],
                    (k + ok[None, :, None] < n), 0)
        acc += tl.dot(cb.to(tl.bfloat16), x)
    xm = tl.load(X + (t0 + om[None, :, None]) * sx
                 + h[:, None, None] * P + on[None, None, :],
                 (om[None, :, None] < n), 0).to(tl.float32)
    acc += xm * tl.load(Dskip + h)[:, None, None]
    y = acc.to(tl.bfloat16)
    # Head/value axes are adjacent for the original 1024-wide norm.
    yy = tl.reshape(tl.trans(y, (1, 0, 2)), (BM, HH * P)).to(tl.float32)
    cols = g * HH * P + tl.arange(0, HH * P)
    z = tl.load(Z + (t0 + om)[:, None] * sz + cols[None, :],
                (om < n)[:, None], 0).to(tl.float32)
    gate = yy * (z / (1. + tl.exp(-z)))
    norm = gate * tl.rsqrt(tl.sum(gate * gate, 1) / (HH * P) + eps)[:, None]
    w = tl.load(W + cols)
    val = (w[None, :].to(tl.float32) * norm.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(Out + (t0 + om)[:, None] * so + cols[None, :], val, (om < n)[:, None])


def scan_norm(cb, x, cm, dt, da, entering, z, params, plan, chunk):
    """Normalized BF16 output; caller guards GB10, geometry and state/output formats."""
    out = torch.empty((plan.tokens, params.d_inner), dtype=torch.bfloat16, device=x.device)
    _scan_norm_kernel[(triton.cdiv(chunk, 32), plan.num_chunks, params.groups)](
        cb, x, cm, x.stride(0), dt, da, entering, params.D, plan.cu_chunks,
        z, z.stride(0), params.norm_weight, out, out.stride(0), params.eps, plan.num_chunks,
        params.heads, params.head_dim, params.state, params.groups, chunk,
        32, 16, 2, 32, num_warps=8, num_stages=2)
    return out
