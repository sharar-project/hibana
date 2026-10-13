# SPDX-License-Identifier: Apache-2.0
"""Per-window quantized recurrent state for the Gated DeltaNet layers (LeapQuant, arXiv 2609.38166).

`sharar/cache/ssm_format.py` defines the storage format and its byte layout (INT8 or INT4 residual). Four kernels
act on it:

- `window_verify`: the verify (and plain decode) recurrence. It reads the committed state as the paper's decode
  kernel does (its eq. 10), without expanding it: the integer residual enters the tensor cores as exact FP16
  integers against the smoothed queries and keys (`(x C) Z^T`, then the row scale B and the window decay), and
  the compensators and the window's committed records form one low-rank product. The verified tokens then run
  in the chunked delta-rule form: one 16-row triangular solve and a few small products instead of a serial loop
  over the tokens. It writes the outputs and one record per token (normalized key, correction u, log decay) into
  the round's ring and never writes the slot. One program owns a value head of a sequence and streams the
  residual through shared memory in K chunks. Optionally it ends with the layer's gated RMS norm (the 27B's swish
  gate or Flash-Next's sigmoid one) and the out projection's input quantization (the 27B's static-scale FP8, or the
  3-bit GEMM's E4M3 per 32-column group), so the core output never round-trips through memory.
- `window_commit`: appends a row's first `count` ring records to its slot's window or, when the window would fill,
  folds the boundary and every record into the new state, refits the compensators by power iteration, smooths
  the key columns and quantizes once. One program owns one value head of one layer and slot.
- `materialize` / `quantize`: the FP32 state of a slot for the chunked prefill kernel, and back.

Products with exact FP16 operands (integers, records, compensators) run as FP16 with FP32 accumulation, exact up
to the accumulation; `_split_dot` splits an FP32 operand into two FP16 halves; the in-block solve uses TF32x3.
The GPU tests compare against the PyTorch formulas in `reference_*`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sharar.cache.ssm_format import WindowLayout
from sharar.kernels.flashnext_fused import _mul_ftz, _rcp_approx_ftz

PREC = "tf32x3"
POWER_ITERS = 1  # subspace iterations per fold, warm-started from the previous compensators
PREFILL_ITERS = 3  # from a prefill's final state (also cold slots)
POLAR_ITERS = 4  # Newton-Schulz steps orthonormalizing a fold's iterate (0: Gram-Schmidt)
VERIFY_BKC = 64  # K chunk of the residual products, streamed through shared memory
VERIFY_STAGES = 2
VERIFY_WARPS = 4
FOLD_WARPS = 4
FIELDS = ("z", "scale", "smooth", "comp_k", "comp_u", "log_k", "log_u", "log_g", "log_len")  # kernel argument order
OUT_BF16, OUT_FP8, OUT_GROUPS = 0, 1, 2  # the fused epilogue's output: BF16, FP8 with a static scale, E4M3 per 32-group


@triton.jit
def _load_z(zb, o_r, c0, BKC: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr):
    """Key columns c0 .. c0 + BKC of one head's residual rows o_r as exact FP16 values [len(o_r), BKC]."""
    if BITS == 8:
        z = tl.load(zb + o_r[:, None] * K + (c0 + tl.arange(0, BKC))[None, :]).to(tl.float16)
    else:
        byte = tl.load(zb + o_r[:, None] * (K // 2) + (c0 // 2 + tl.arange(0, BKC // 2))[None, :]).to(tl.int32)
        z = tl.reshape(tl.join((byte & 15) - 8, (byte >> 4) - 8), [o_r.shape[0], BKC]).to(tl.float16)
    return z


@triton.jit
def _store_z(zb, q, o_r, K: tl.constexpr, BITS: tl.constexpr):
    """Store integer-valued FP32 rows q [len(o_r), K] (|q| <= the format's maximum) into the residual."""
    if BITS == 8:
        tl.store(zb + o_r[:, None] * K + tl.arange(0, K)[None, :], q.to(tl.int8))
    else:
        lo, hi = tl.split(tl.reshape(q.to(tl.int32) + 8, [o_r.shape[0], K // 2, 2]))
        tl.store(zb + o_r[:, None] * (K // 2) + tl.arange(0, K // 2)[None, :], (lo | (hi << 4)).to(tl.uint8))


@triton.jit
def _split_dot(x, y16):
    """x @ y16 for FP32 x and exact FP16 y16: x as two FP16 halves, to within FP32 rounding."""
    hi = x.to(tl.float16)
    return tl.dot(hi, y16) + tl.dot((x - hi.to(tl.float32)).to(tl.float16), y16)


@triton.jit
def _window_state(zb, scb, smb, ckb, cub, lkb, lub, lgb, rkb, rub, rgb, n, c, o_v,
                  K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, BJ: tl.constexpr,
                  BITS: tl.constexpr):
    """One head's state rows o_v as FP32 [len(o_v), K] after n logged and then c ring records:
    e^G (C . dequant(Z) . B + K~ U~^T) + sum_j e^(G - G_j) u_j k_j^T. Pointers are per head (or key head)."""
    o_k = tl.arange(0, K)
    st = _load_z(zb, o_v, 0, K, K, BITS).to(tl.float32) * tl.load(scb + o_v)[:, None] * tl.load(smb + o_k)[None, :]
    if R > 0:
        o_r = tl.arange(0, RP)
        ck = tl.load(ckb + o_r[:, None] * K + o_k[None, :], mask=o_r[:, None] < R, other=0.)
        cu = tl.load(cub + o_r[None, :] * V + o_v[:, None], mask=o_r[None, :] < R, other=0.)
        st += tl.dot(cu, ck)  # FP16 factors: exact products
    o_j = tl.arange(0, BJ)
    from_log = o_j < n
    r_j = o_j - n
    from_ring = (o_j >= n) & (r_j < c)
    g = tl.load(lgb + o_j, mask=from_log, other=0.) + tl.load(rgb + r_j, mask=from_ring, other=0.)
    gt = tl.sum(g, 0)
    w = tl.where(from_log | from_ring, tl.exp(gt - tl.cumsum(g, 0)), 0.)
    kk = (tl.load(lkb + o_j[:, None] * K + o_k[None, :], mask=from_log[:, None], other=0.)
          + tl.load(rkb + r_j[:, None] * K + o_k[None, :], mask=from_ring[:, None], other=0.))
    uw = (tl.load(lub + o_j[None, :] * V + o_v[:, None], mask=from_log[None, :], other=0.).to(tl.float32)
          + tl.load(rub + r_j[None, :] * V + o_v[:, None], mask=from_ring[None, :], other=0.).to(tl.float32))
    return st * tl.exp(gt) + _split_dot(uw * w[None, :], kk)


@triton.jit
def _orthonormal(x, R: tl.constexpr, RP: tl.constexpr):
    """Gram-Schmidt over the first R columns of x [K, RP] (the rest zero); a vanishing column stays zero."""
    o_r = tl.arange(0, RP)
    out = tl.zeros_like(x)
    for i in tl.static_range(R):
        col = tl.sum(tl.where(o_r[None, :] == i, x, 0.), 1)
        for j in tl.static_range(i):
            prev = tl.sum(tl.where(o_r[None, :] == j, out, 0.), 1)
            col -= tl.sum(col * prev, 0) * prev
        col = col / tl.sqrt(tl.sum(col * col, 0) + 1e-30)
        out = tl.where(o_r[None, :] == i, col[:, None], out)
    return out


@triton.jit
def _polar(x, R: tl.constexpr, RP: tl.constexpr, STEPS: tl.constexpr):
    """An orthonormal basis of span(x) for x [K, RP] with unit columns: the polar factor by Newton-Schulz,
    x <- x (3 I - x^T x) / 2, two small tensor-core products per step and no serial reductions (STEPS = 0: Gram-Schmidt).

    Newton-Schulz converges for singular values in (0, sqrt(3)); unit columns have them at most sqrt(R), so the
    function first scales x by sqrt(2 / R). Orthogonal columns then converge in 3-4 steps; a direction the columns
    barely span grows only 1.5x per step, which leaves the fit partial but never wrong: the caller forms the
    residual exactly from whatever factors come out, and the next window's fit starts from them. Zero columns
    stay zero."""
    if STEPS == 0:
        x = _orthonormal(x, R, RP)
    else:
        o_r = tl.arange(0, RP)
        eye = tl.where(o_r[:, None] == o_r[None, :], 1.5, 0.)
        x = x * tl.sqrt(2. / R)
        for _ in tl.static_range(STEPS):
            gram = tl.dot(tl.trans(x), x, input_precision="tf32")
            x = tl.dot(x, eye - 0.5 * gram, input_precision="tf32")
    return x


@triton.jit
def _fit_quantize(st, zb, scb, smb, ckb, cub, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
                  RP: tl.constexpr, ITERS: tl.constexpr, BITS: tl.constexpr, QMAX: tl.constexpr,
                  POLAR: tl.constexpr):
    """Fit R compensators to one head's state st [V, K], smooth the key columns of the residual and quantize it per
    value row (pointers per head).

    Subspace iteration with BF16 products finds the state's dominant key subspace for the compensators, starting
    from the previous compensators (fixed pseudo-random directions on a fresh slot). Whatever they miss stays in
    the residual, which this function forms exactly from their stored FP16 factors."""
    o_v = tl.arange(0, V)
    o_k = tl.arange(0, K)
    if R > 0:
        o_r = tl.arange(0, RP)
        x = tl.load(ckb + o_r[None, :] * K + o_k[:, None], mask=o_r[None, :] < R, other=0.).to(tl.float32)
        seed = tl.where(o_r[None, :] < R, tl.sin((o_k[:, None] + 1.) * (o_r[None, :] * 0.618 + 1.3) * 2.399), 0.)
        x = x / tl.sqrt(tl.sum(x * x, 0) + 1e-30)[None, :] + 0.01 * seed
        x = _polar(x / tl.sqrt(tl.sum(x * x, 0) + 1e-30)[None, :], R, RP, POLAR)
        sb = st.to(tl.bfloat16)
        for _ in tl.static_range(ITERS):
            y = tl.dot(sb, x.to(tl.bfloat16))  # [V, RP]
            x = tl.trans(tl.dot(tl.trans(y.to(tl.bfloat16)), sb))
            x = _polar(x / tl.sqrt(tl.sum(x * x, 0) + 1e-30)[None, :], R, RP, POLAR)
        y = tl.dot(sb, x.to(tl.bfloat16))
        # sigma_r y_hat x^T as two FP16 factors of norm sqrt(sigma_r) each, so neither overflows.
        root = tl.sqrt(tl.sqrt(tl.sum(y * y, 0)))
        kh = (x * root[None, :]).to(tl.float16)
        uh = (y / tl.maximum(root, 1e-30)[None, :]).to(tl.float16)
        st -= tl.dot(uh, tl.trans(kh))  # exact products of the stored factors
        tl.store(ckb + o_r[None, :] * K + o_k[:, None], kh, mask=o_r[None, :] < R)
        tl.store(cub + o_r[None, :] * V + o_v[:, None], uh, mask=o_r[None, :] < R)
    smooth = tl.sqrt(tl.maximum(tl.sum(tl.abs(st), 0) / V, 1e-24))
    st = st / smooth[None, :]
    amax = tl.max(tl.abs(st), 1)
    q = libdevice.rint(st * tl.where(amax > 0, QMAX / amax, 0.)[:, None])
    _store_z(zb, tl.minimum(tl.maximum(q, -QMAX), QMAX), o_v, K, BITS)
    tl.store(scb + o_v, amax / QMAX)
    tl.store(smb + o_k, smooth)


@triton.jit(do_not_specialize=["T"])
def _verify_kernel(q, k, v, a, b, A_log, dt_bias, o, o_scale, gate, norm_w, out_scale,
                   z, scale, smooth, comp_k, comp_u, log_k, log_u, log_g, log_len, slots,
                   ring_k, ring_u, ring_g,
                   s_z, s_sc, s_sm, s_ck, s_cu, s_lk, s_lu, s_lg, s_ll,
                   stride_q, stride_k, stride_v, stride_a, stride_b, stride_gate, stride_o,
                   qk_scale, eps, T,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BKC: tl.constexpr,
                   BT: tl.constexpr, P: tl.constexpr, R: tl.constexpr, BJ: tl.constexpr, RS: tl.constexpr,
                   BITS: tl.constexpr, PREC: tl.constexpr, LOG_BT: tl.constexpr, FUSED: tl.constexpr,
                   SIGMOID_GATE: tl.constexpr, OUT: tl.constexpr, STAGES: tl.constexpr):
    """One value head of one sequence, all V columns. The integer residual streams through shared memory in K chunks
    (software-pipelined, so its loads overlap the products) into Q S and K S accumulators that hold whole rows; the
    kernel forms the rest of the block after the stream.

    FUSED: the epilogue is the layer's gated RMS norm (gate z * sigmoid(z), or sigmoid(z) with SIGMOID_GATE) and o is
    the out projection's input [tokens, HV * V]: BF16, FP8 with the projection's static input scale (OUT_FP8), or E4M3
    per (token, 32-column group) with the group scales in o_scale [tokens, HV * V / 32] (OUT_GROUPS)."""
    hv, i_n = tl.program_id(0), tl.program_id(1)
    i_h = hv // (HV // H)
    slot = tl.load(slots + i_n).to(tl.int64)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    # The committed state, read as in the paper's eq. 10 without expanding it:
    #   x^T S = e^(G_n) [((x C) Z^T) B + (x K~^T) U~^T] + sum_j e^(G_n - G_j) (x k_j) u_j^T.
    # Records: rows j < P are the window's committed tokens (j < n), rows P..P+R-1 the compensators.
    n = tl.minimum(tl.load(log_len + slot * s_ll + hv), P)
    o_j = tl.arange(0, BJ)
    is_l = o_j < n
    is_c = (o_j >= P) & (o_j < P + R)
    rec = is_l | is_c
    p_k = tl.where(is_c, comp_k + slot * s_ck + (hv * R + o_j - P) * K, log_k + slot * s_lk + (i_h * P + o_j) * K)
    p_u = tl.where(is_c, comp_u + slot * s_cu + (hv * R + o_j - P) * V, log_u + slot * s_lu + (hv * P + o_j) * V)
    lk = tl.load(p_k[:, None] + o_k[None, :], mask=rec[:, None], other=0.)  # [BJ, K]
    lg = tl.load(log_g + slot * s_lg + hv * P + o_j, mask=is_l, other=0.)

    # The verified tokens: gates, L2-normalized queries and keys. Round the keys to FP16 here, as the ring stores
    # them, so each record's correction uses exactly the key the commit applies.
    bos = i_n * T
    o_t = tl.arange(0, BT)
    mt = o_t < T
    tok = bos + o_t
    x = tl.load(a + tok * stride_a + hv, mask=mt, other=0.).to(tl.float32) + tl.load(dt_bias + hv).to(tl.float32)
    g = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * tl.where(x <= 20., tl.log(1. + tl.exp(x)), x)
    g = tl.where(mt, g, 0.)
    beta = tl.where(mt, 1. / (1. + tl.exp(-tl.load(b + tok * stride_b + hv, mask=mt, other=0.).to(tl.float32))), 0.)
    bq = tl.load(q + tok[:, None] * stride_q + i_h * K + o_k[None, :], mask=mt[:, None], other=0.).to(tl.float32)
    bk = tl.load(k + tok[:, None] * stride_k + i_h * K + o_k[None, :], mask=mt[:, None], other=0.).to(tl.float32)
    rq = qk_scale / tl.sqrt(tl.sum(bq * bq, 1) + 1e-6)
    rk = 1. / tl.sqrt(tl.sum(bk * bk, 1) + 1e-6)
    qh = (bq * rq[:, None]).to(tl.float16)
    kh = (bk * rk[:, None]).to(tl.float16)
    # This round's records, for the commit (the corrections u below).
    tl.store(ring_g + (i_n * HV + hv) * RS + o_t, g, mask=mt)
    if hv % (HV // H) == 0:  # the normalized keys, once per key head
        tl.store(ring_k + (i_n * H + i_h) * RS * K + o_t[:, None] * K + o_k[None, :], kh, mask=mt[:, None])

    # ((x C) Z^T) of the integer residual for x = Q and K: exact FP16 integers against the smoothed rows, chunk by
    # chunk of K (reloading the chunks of q and k from cache instead of slicing them).
    zb = z + slot * s_z + hv * V * (K * BITS // 8)
    acc_q = tl.zeros([BT, V], tl.float32)
    acc_k = tl.zeros([BT, V], tl.float32)
    for c in tl.range(0, K // BKC, num_stages=STAGES):
        o_c = c * BKC + tl.arange(0, BKC)
        zc = _load_z(zb, o_v, c * BKC, BKC, K, BITS)  # [V, BKC]
        smc = tl.load(smooth + slot * s_sm + hv * K + o_c)
        qc = tl.load(q + tok[:, None] * stride_q + i_h * K + o_c[None, :], mask=mt[:, None], other=0.).to(tl.float32)
        kc = tl.load(k + tok[:, None] * stride_k + i_h * K + o_c[None, :], mask=mt[:, None], other=0.).to(tl.float32)
        qc = (qc * rq[:, None] * smc[None, :]).to(tl.float16)
        kc = ((kc * rk[:, None]).to(tl.float16).to(tl.float32) * smc[None, :]).to(tl.float16)
        zt = tl.trans(zc)
        acc_q = tl.dot(qc, zt, acc_q)
        acc_k = tl.dot(kc, zt, acc_k)

    # The verified block, chunked: U = (I + A)^-1 diag(beta) (V - diag(e^G) K S),
    # O = diag(e^G) Q S + (M . Q K^T) U, with A_tj = beta_t e^(G_t - G_j) k_t.k_j (j < t) and M_tj = e^(G_t - G_j) (j <= t).
    # A is nilpotent (A^BT = 0): (I + A)^-1 = (I - A)(I + A^2)(I + A^4)..., the factors commuting.
    kk = tl.dot(kh, tl.trans(kh))
    qk = tl.dot(qh, tl.trans(kh))
    gc = tl.cumsum(g, 0)
    eg = tl.exp(gc)
    lower = o_t[:, None] > o_t[None, :]
    decay = tl.exp(tl.where(o_t[:, None] >= o_t[None, :], gc[:, None] - gc[None, :], 0.))
    am = tl.where(lower, beta[:, None] * decay * kk, 0.)
    eye = tl.where(o_t[:, None] == o_t[None, :], 1., 0.)
    inv = eye - am
    pw = am
    for _ in tl.static_range(LOG_BT - 1):
        pw = tl.dot(pw, pw, input_precision=PREC)
        inv = tl.dot(inv, eye + pw, input_precision=PREC)

    gn = tl.sum(lg, 0)
    en = tl.exp(gn)
    w = tl.where(is_l, tl.exp(gn - tl.cumsum(lg, 0)), tl.where(is_c, en, 0.))
    lu = tl.load(p_u[:, None] + o_v[None, :], mask=rec[:, None], other=0.)  # [BJ, V]
    lkt = tl.trans(lk)
    sc = tl.load(scale + slot * s_sc + hv * V + o_v) * en
    qs = _split_dot(tl.dot(qh, lkt) * w[None, :], lu) + acc_q * sc[None, :]
    ks = _split_dot(tl.dot(kh, lkt) * w[None, :], lu) + acc_k * sc[None, :]
    bv = tl.load(v + tok[:, None] * stride_v + hv * V + o_v[None, :], mask=mt[:, None], other=0.).to(tl.float32)
    u = tl.dot(inv, beta[:, None] * (bv - eg[:, None] * ks), input_precision=PREC)
    out = eg[:, None] * qs + tl.dot(tl.where(lower | (o_t[:, None] == o_t[None, :]), decay * qk, 0.), u, input_precision=PREC)
    tl.store(ring_u + (i_n * HV + hv) * RS * V + o_t[:, None] * V + o_v[None, :], u.to(tl.float16), mask=mt[:, None])
    if FUSED:
        # The gated RMS norm of the BF16 core (verify_gated_norm, or rms_norm_gated's sigmoid gate, with FP32 scalars),
        # then the quantization of its BF16 result: quant_fp8 (OUT_FP8) or w3.quant_groups (OUT_GROUPS).
        xo = out.to(tl.bfloat16).to(tl.float32)
        zg = tl.load(gate + tok[:, None] * stride_gate + hv * V + o_v[None, :], mask=mt[:, None], other=0.).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(xo * xo, 1) / V + eps)
        y = xo * rstd[:, None] * tl.load(norm_w + o_v).to(tl.float32)[None, :]
        if SIGMOID_GATE:
            y *= tl.sigmoid(zg)
        else:
            y *= zg * tl.sigmoid(zg)
        if OUT == 1:
            yq = tl.minimum(tl.maximum(y.to(tl.bfloat16).to(tl.float32) / tl.load(out_scale), -448.), 448.)
            tl.store(o + tok[:, None] * stride_o + hv * V + o_v[None, :], yq.to(tl.float8e4nv), mask=mt[:, None])
        elif OUT == 2:
            # quant_groups_kernel's arithmetic as nvcc --use_fast_math compiles it (silu_mul_quant_groups'):
            # s = max(amax * (1/448), 1e-12), q = e4m3(x * rcp.approx.ftz(s)) with mul.ftz.
            yg = tl.reshape(y.to(tl.bfloat16).to(tl.float32), [BT, V // 32, 32])
            amax = tl.max(tl.abs(yg), 2)
            sg = tl.maximum(_mul_ftz(amax, tl.full(amax.shape, 1.0 / 448.0, tl.float32)), 1.0e-12)
            yq = tl.reshape(_mul_ftz(yg, _rcp_approx_ftz(sg)[:, :, None]), [BT, V])
            tl.store(o + tok[:, None] * stride_o + hv * V + o_v[None, :], yq.to(tl.float8e4nv), mask=mt[:, None])
            o_g = tl.arange(0, V // 32)
            tl.store(o_scale + tok[:, None] * (stride_o // 32) + hv * (V // 32) + o_g[None, :], sg, mask=mt[:, None])
        else:
            tl.store(o + tok[:, None] * stride_o + hv * V + o_v[None, :], y.to(tl.bfloat16), mask=mt[:, None])
    else:
        tl.store(o + tok[:, None] * stride_o + hv * V + o_v[None, :], out.to(o.dtype.element_ty), mask=mt[:, None])


@triton.jit
def _commit_kernel(z, scale, smooth, comp_k, comp_u, log_k, log_u, log_g, log_len, slots, counts,
                   ring_k, ring_u, ring_g,
                   l_z, s_z, l_sc, s_sc, l_sm, s_sm, l_ck, s_ck, l_cu, s_cu, l_lk, s_lk, l_lu, s_lu, l_lg, s_lg, l_ll, s_ll,
                   r_k, r_u, r_g,
                   H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                   P: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, RS: tl.constexpr, BS: tl.constexpr,
                   BJ: tl.constexpr, ITERS: tl.constexpr, BITS: tl.constexpr, QMAX: tl.constexpr,
                   POLAR: tl.constexpr):
    lh, row = tl.program_id(0), tl.program_id(1)
    layer = (lh // HV).to(tl.int64)
    hv = lh % HV
    i_h = hv // (HV // H)
    slot = tl.load(slots + row).to(tl.int64)
    c = tl.load(counts + row)
    # Slot 0 is the padded rows' dummy and a count of 0 commits nothing (device acceptance's padded rows).
    if (slot > 0) & (c > 0):
        llb = log_len + layer * l_ll + slot * s_ll
        lkb = log_k + layer * l_lk + slot * s_lk + i_h * P * K
        lub = log_u + layer * l_lu + slot * s_lu + hv * P * V
        lgb = log_g + layer * l_lg + slot * s_lg + hv * P
        rkb = ring_k + layer * r_k + (row * H + i_h) * RS * K
        rub = ring_u + layer * r_u + (row * HV + hv) * RS * V
        rgb = ring_g + layer * r_g + (row * HV + hv) * RS
        o_v = tl.arange(0, V)
        o_k = tl.arange(0, K)
        n = tl.minimum(tl.load(llb + hv), P - 1)
        # Every warp reads the record count before any warp rewrites it. The append path has no other synchronization,
        # and a warp that read n + c after another warp stored it would take the fold path alone: its reductions would
        # run without partners and write NaN into its share of the boundary.
        tl.debug_barrier()
        if n + c < P:
            o_s = tl.arange(0, BS)
            ms = o_s < c
            tl.store(lub + (n + o_s)[:, None] * V + o_v[None, :], tl.load(rub + o_s[:, None] * V + o_v[None, :], mask=ms[:, None]), mask=ms[:, None])
            tl.store(lgb + n + o_s, tl.load(rgb + o_s, mask=ms), mask=ms)
            if hv % (HV // H) == 0:  # one writer per key head; its value heads only read it while folding
                tl.store(lkb + (n + o_s)[:, None] * K + o_k[None, :], tl.load(rkb + o_s[:, None] * K + o_k[None, :], mask=ms[:, None]), mask=ms[:, None])
            tl.store(llb + hv, n + c)
        else:
            zb = z + layer * l_z + slot * s_z + hv * V * (K * BITS // 8)
            scb = scale + layer * l_sc + slot * s_sc + hv * V
            smb = smooth + layer * l_sm + slot * s_sm + hv * K
            ckb = comp_k + layer * l_ck + slot * s_ck + hv * R * K
            cub = comp_u + layer * l_cu + slot * s_cu + hv * R * V
            st = _window_state(zb, scb, smb, ckb, cub, lkb, lub, lgb, rkb, rub, rgb, n, c, o_v, K, V, R, RP, BJ, BITS)
            _fit_quantize(st, zb, scb, smb, ckb, cub, K, V, R, RP, ITERS, BITS, QMAX, POLAR)
            tl.store(llb + hv, 0)


@triton.jit
def _materialize_kernel(z, scale, smooth, comp_k, comp_u, log_k, log_u, log_g, log_len, slots, out,
                        s_z, s_sc, s_sm, s_ck, s_cu, s_lk, s_lu, s_lg, s_ll,
                        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                        P: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, BJ: tl.constexpr, BITS: tl.constexpr):
    i_v, i_n, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = hv // (HV // H)
    slot = tl.load(slots + i_n).to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    n = tl.minimum(tl.load(log_len + slot * s_ll + hv), P)
    lkb = log_k + slot * s_lk + i_h * P * K
    lub = log_u + slot * s_lu + hv * P * V
    lgb = log_g + slot * s_lg + hv * P
    st = _window_state(z + slot * s_z + hv * V * (K * BITS // 8), scale + slot * s_sc + hv * V, smooth + slot * s_sm + hv * K,
                       comp_k + slot * s_ck + hv * R * K, comp_u + slot * s_cu + hv * R * V, lkb, lub, lgb, lkb, lub, lgb,
                       n, 0, o_v, K, V, R, RP, BJ, BITS)
    tl.store(out + (i_n.to(tl.int64) * HV + hv) * V * K + o_v[:, None] * K + o_k[None, :], st)


@triton.jit
def _quantize_kernel(src, z, scale, smooth, comp_k, comp_u, log_k, log_u, log_g, log_len, slots,
                     s_src, s_z, s_sc, s_sm, s_ck, s_cu, s_lk, s_lu, s_lg, s_ll,
                     HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                     ITERS: tl.constexpr, BITS: tl.constexpr, QMAX: tl.constexpr):
    i_n, hv = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + i_n).to(tl.int64)
    o_v = tl.arange(0, V)
    o_k = tl.arange(0, K)
    st = tl.load(src + i_n.to(tl.int64) * s_src + hv * V * K + o_v[:, None] * K + o_k[None, :]).to(tl.float32)
    _fit_quantize(st, z + slot * s_z + hv * V * (K * BITS // 8), scale + slot * s_sc + hv * V, smooth + slot * s_sm + hv * K,
                  comp_k + slot * s_ck + hv * R * K, comp_u + slot * s_cu + hv * R * V, K, V, R, RP, ITERS, BITS, QMAX, 0)
    tl.store(log_len + slot * s_ll + hv, 0)


def _parts(views: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return [views[name] for name in FIELDS]


def _strides(views: dict[str, torch.Tensor], dim: int) -> list[int]:
    return [t.stride(dim) for t in _parts(views)]


def _tile(n: int) -> int:
    """Rows of a tile that enters an MMA: a power of two, at least 16."""
    return max(16, triton.next_power_of_2(n))


def _qmax(layout: WindowLayout) -> float:
    return float(2 ** (layout.bits - 1) - 1)


def window_verify(layout: WindowLayout, views: dict[str, torch.Tensor], q, k, v, a, b, A_log, dt_bias, slots,
                  steps: int, ring_k, ring_u, ring_g, norm=None) -> torch.Tensor:
    """One layer's verify (or decode, steps=1) over N sequences of `steps` tokens each, laid out token-major.

    views: this layer's region views ([slots, ...] each). q/k [1, N*steps, H, K] and v [1, N*steps, HV, V] may be
    token-strided views of the convolution output; a/b [N*steps, HV] gate inputs. ring_* [>=N, heads, >=steps, ...]
    receive the records. Returns the core output [1, N*steps, HV, V] in q's dtype, or with norm = (gate [N*steps,
    HV*V], weight [V], eps, quantization, gate activation "swish" or "sigmoid") the out projection's input
    [N*steps, HV*V]: the gated RMS norm of `verify_gated_norm` (swish) or `rms_norm_gated` (sigmoid) applied to that
    output, then with an FP8 input scale `quant_fp8` (FP8), with "groups" `w3.quant_groups` (E4M3 and its FP32 scales
    [N*steps, HV*V/32], a pair, the 3-bit GEMM's input), or with None nothing (BF16)."""
    tokens, h, kd = q.shape[1], q.shape[2], q.shape[3]
    hv, vd = v.shape[2], v.shape[3]
    n = slots.numel()
    if tokens != n * steps or (h, hv, kd, vd) != (layout.key_heads, layout.value_heads, layout.k, layout.v):
        raise ValueError("window verify: inputs do not match the slots, steps or layout")
    if steps > ring_u.shape[2] or n > ring_u.shape[0]:
        raise ValueError("window verify: the ring is smaller than the verified block")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1 or q.stride(-2) != kd or k.stride(-2) != kd or v.stride(-2) != vd:
        raise ValueError("window verify: q/k/v must be head-contiguous views")
    bt = _tile(steps)
    fused = norm is not None
    if fused:
        gate, norm_w, eps, quant, activation = norm
        if gate.shape != (tokens, hv * vd) or gate.stride(-1) != 1:
            raise ValueError("window verify: the norm gate must be [tokens, HV * V] with unit column stride")
        if activation not in ("swish", "sigmoid"):
            raise ValueError(f"window verify: no fused {activation!r} gate (swish or sigmoid)")
        if isinstance(quant, str) and quant != "groups":
            raise ValueError(f"window verify: unknown quantization {quant!r} (an FP8 scale, \"groups\" or None)")
        out_kind = OUT_BF16 if quant is None else OUT_GROUPS if isinstance(quant, str) else OUT_FP8
        o = torch.empty((tokens, hv * vd), dtype=torch.bfloat16 if out_kind == OUT_BF16 else torch.float8_e4m3fn, device=q.device)
        o_scale = torch.empty((tokens, hv * vd // 32), dtype=torch.float32, device=q.device) if out_kind == OUT_GROUPS else o
        out_scale = quant if out_kind == OUT_FP8 else gate
    else:
        gate, norm_w, eps, out_scale, activation, out_kind = q, q, 0., q, "swish", OUT_BF16
        o = q.new_empty(1, tokens, hv, vd)
        o_scale = o
    _verify_kernel[(hv, n)](
        q, k, v, a, b, A_log, dt_bias, o, o_scale, gate, norm_w, out_scale, *_parts(views), slots, ring_k, ring_u, ring_g,
        *_strides(views, 0), q.stride(1), k.stride(1), v.stride(1), a.stride(-2), b.stride(-2),
        gate.stride(0) if fused else 0, hv * vd, kd ** -0.5, eps, steps,
        H=h, HV=hv, K=kd, V=vd, BKC=VERIFY_BKC, BT=bt, P=layout.window, LOG_BT=bt.bit_length() - 1, R=layout.rank,
        BJ=_tile(layout.window + layout.rank), RS=ring_u.shape[2], BITS=layout.bits, PREC=PREC, FUSED=fused,
        SIGMOID_GATE=activation == "sigmoid", OUT=out_kind,
        STAGES=VERIFY_STAGES, num_warps=VERIFY_WARPS, num_stages=VERIFY_STAGES,
    )
    return (o, o_scale) if out_kind == OUT_GROUPS else o


def window_commit(layout: WindowLayout, views: dict[str, torch.Tensor], ring_k, ring_u, ring_g, slots, counts) -> None:
    """Commit `counts[i]` ring records of row i into `slots[i]` for every layer in `views` ([layers, slots, ...])
    and ring_* ([layers, rows, heads, steps, ...]). Counts are device values (0 commits nothing)."""
    layers = views["z"].shape[0]
    b, steps = slots.numel(), ring_u.shape[3]
    _commit_kernel[(layers * layout.value_heads, b)](
        *_parts(views), slots, counts, ring_k, ring_u, ring_g,
        *[s for pair in zip(_strides(views, 0), _strides(views, 1)) for s in pair],
        ring_k.stride(0), ring_u.stride(0), ring_g.stride(0),
        H=layout.key_heads, HV=layout.value_heads, K=layout.k, V=layout.v, P=layout.window, R=layout.rank,
        RP=_tile(layout.rank), RS=steps, BS=_tile(steps), BJ=_tile(layout.window - 1 + steps),
        ITERS=POWER_ITERS, BITS=layout.bits, QMAX=_qmax(layout), POLAR=POLAR_ITERS,
        num_warps=FOLD_WARPS, num_stages=1,
    )


def materialize(layout: WindowLayout, views: dict[str, torch.Tensor], slots: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """One layer's FP32 states [N, HV, V, K] of `slots` (the chunked prefill's initial states)."""
    n = slots.numel()
    hv, kd, vd = layout.value_heads, layout.k, layout.v
    if out is None:
        out = torch.empty((n, hv, vd, kd), dtype=torch.float32, device=slots.device)
    _materialize_kernel[(vd // 32, n, hv)](
        *_parts(views), slots, out, *_strides(views, 0),
        H=layout.key_heads, HV=hv, K=kd, V=vd, BV=32, P=layout.window, R=layout.rank, RP=_tile(layout.rank),
        BJ=_tile(layout.window), BITS=layout.bits, num_warps=4, num_stages=1,
    )
    return out


def quantize(layout: WindowLayout, views: dict[str, torch.Tensor], states: torch.Tensor, slots: torch.Tensor) -> None:
    """Store FP32/BF16 states [N, HV, V, K] into one layer's `slots` as new boundary states (empty windows)."""
    n = slots.numel()
    if states.shape[1:] != (layout.value_heads, layout.v, layout.k) or states.shape[0] != n or states.stride(-1) != 1:
        raise ValueError("window quantize: states do not match the layout or slots")
    _quantize_kernel[(n, layout.value_heads)](
        states, *_parts(views), slots, states.stride(0), *_strides(views, 0),
        HV=layout.value_heads, K=layout.k, V=layout.v, R=layout.rank, RP=_tile(layout.rank), ITERS=PREFILL_ITERS, BITS=layout.bits,
        QMAX=_qmax(layout), num_warps=FOLD_WARPS, num_stages=1,
    )


class WindowState:
    """The verify ring of a quantized pool: one record per verified token and layer; `run` commits accepted prefixes.
    It replaces `AcceptedState` for quantized pools, with the same `layer_inputs`, `run` and `bytes`."""

    def __init__(self, pool, batch, steps, key_heads):
        layout = pool.window.layout
        layers = len(pool.layer_ids)
        device = pool.ssm_all.device
        self.batch, self.steps = batch, steps
        hv, kd, vd = layout.value_heads, layout.k, layout.v
        self.ring_k = torch.zeros((layers, batch, key_heads, steps, kd), dtype=torch.float16, device=device)
        self.ring_u = torch.zeros((layers, batch, hv, steps, vd), dtype=torch.float16, device=device)
        self.ring_g = torch.zeros((layers, batch, hv, steps), dtype=torch.float32, device=device)
        self.layer_inputs = {lid: dict(ring_k=self.ring_k[j], ring_u=self.ring_u[j], ring_g=self.ring_g[j])
                             for j, lid in enumerate(pool.layer_ids)}

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.ring_k, self.ring_u, self.ring_g))

    def run(self, pool, slots, counts):
        """Commit counts[i] records of row i into slots[i], on the caller's stream."""
        window_commit(pool.window.layout, pool.window.all, self.ring_k, self.ring_u, self.ring_g, slots, counts)


# ---- PyTorch references (tests) ----

def reference_dequant(layout: WindowLayout, z: torch.Tensor) -> torch.Tensor:
    """The integer residual values of a `z` region (any leading dimensions) as FP64 [..., V, K]."""
    if layout.bits == 8:
        return z.double()
    byte = z.to(torch.int16)
    return torch.stack(((byte & 15) - 8, (byte >> 4) - 8), -1).flatten(-2).double()


def reference_state(layout: WindowLayout, views: dict[str, torch.Tensor], slot: int) -> torch.Tensor:
    """One layer's committed state of `slot` as FP64 [HV, V, K] (views: [slots, ...])."""
    z = reference_dequant(layout, views["z"][slot])
    st = z * views["scale"][slot].double()[..., None] * views["smooth"][slot].double()[:, None, :]
    st = st + torch.einsum("hrv,hrk->hvk", views["comp_u"][slot].double(), views["comp_k"][slot].double())
    group = layout.value_heads // layout.key_heads
    for h in range(layout.value_heads):
        n = int(views["log_len"][slot][h])
        for j in range(n):
            g = float(views["log_g"][slot][h, j])
            st[h] = st[h] * torch.exp(torch.tensor(g, dtype=torch.float64)) + torch.outer(
                views["log_u"][slot][h, j].double(), views["log_k"][slot][h // group, j].double())
    return st


def reference_block(state, q, k, v, a, b, A_log, dt_bias):
    """The recurrent delta rule over one sequence's T tokens from FP64 state [HV, V, K]: (outputs [T, HV, V], records
    (k_hat [T, H, K], u [T, HV, V], g [T, HV]), final state). q/k [T, H, K], v [T, HV, V], a/b [T, HV]."""
    st = state.clone().double()
    t_len, h, kd = q.shape
    hv = v.shape[1]
    group = hv // h
    scale = kd ** -0.5
    x = a.double() + dt_bias.double()
    g = -torch.exp(A_log.double()) * torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
    beta = torch.sigmoid(b.double())
    qn = q.double() / torch.sqrt((q.double() ** 2).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k.double() / torch.sqrt((k.double() ** 2).sum(-1, keepdim=True) + 1e-6)
    out = torch.empty((t_len, hv, v.shape[2]), dtype=torch.float64, device=state.device)
    us = torch.empty_like(out)
    for t in range(t_len):
        kh = kn[t].repeat_interleave(group, 0)  # [HV, K]
        qh = qn[t].repeat_interleave(group, 0)
        st = st * torch.exp(g[t])[:, None, None]
        u = beta[t][:, None] * (v[t].double() - torch.einsum("hvk,hk->hv", st, kh))
        st = st + u[:, :, None] * kh[:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", st, qh)
        us[t] = u
    return out, (kn, us, g), st
