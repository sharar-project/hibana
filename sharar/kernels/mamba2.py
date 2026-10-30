# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao, Albert Gu.
# The chunked-scan kernels are adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops
# (ssd_chunk_state.py and ssd_state_passing.py, fused here; ssd_bmm.py; ssd_chunk_scan.py), themselves adapted from
# https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/
"""Mamba-2 mixer kernels for Nemotron-3.5-Super (40 Mamba layers per forward).

The mixer between the FP8 projections, as `NemotronHMamba2Mixer` defines it (transformers' nemotron_h, the same
semantics as vLLM's MambaMixer2): in_proj -> z | xBC | dt (8192 | 10240 | 128); xBC through a width-4 depthwise
causal convolution with bias and SiLU, split into x (128 heads x 64), B and C (8 groups x 128); dt = softplus(dt +
dt_bias), A = -exp(A_log); h' = exp(dt A) h + dt x B^T per head, y = h' C + D x; then y * silu(z), RMS-normalized
in groups of 1024 (norm_before_gate=False) and scaled by the norm weight; out_proj. The HF port clamps dt from below
at time_step_min (1e-3) in its chunked path only (not in its recurrent path); vLLM and the checkpoint's
time_step_limit (0, inf) do not clamp, and neither does this module. Layer 0's dt does go below 1e-3.

The recurrent state is [slot, head, headdim, d_state], which is the state pool's [slot, HV, V, K], FP32 or FP16
(`SsmFormat` fp16_sr); the convolution state is the last three pre-convolution inputs, [slot, conv_dim, 3]
BF16 (the in_proj's own BF16 outputs, so it stores them exactly in every format). Every path addresses the pool by
slot index and reads no host value, so decode, verify and replay are CUDA-graph safe with static shapes.

An FP16 state is computed in FP32 and rounded on every store (`_round_state`): decode's in-place update, each verify
step (the running state continues from the rounded value, so verify == decode bitwise; snapshots hold it), each
replayed step, and each prefill chunk end (checkpoints and the final state; chunks sit on the 128-token grid, so a
split prefill still reproduces the unsplit one). The readout y uses the FP32 state before rounding. Stochastic
rounding (fp16_sr, NVIDIA's recipe for this model on DGX Spark) draws its bits from Philox (5 rounds) with the counter
(element // 4, token position, layer index) and a fixed key: no host RNG, nothing per launch, so a run reproduces
itself bit for bit whatever its slot or batch, and every path that stores the state after a given token uses the
same bits. GB10 has no cvt.rs, so the rounding is integer arithmetic on the FP32 bits (`_sr_fp16`).

- Prefill (`prefill`) runs the chunked SSD algorithm over packed varlen sequences: the convolution, a cumulative
  sum of dt A per chunk, the chunk-local C B^T, one sequential pass per sequence over its chunks that carries the
  FP32 state (`_chunk_states_kernel`: each chunk's contribution formed in registers with a 3xTF32 dot and folded in,
  so no per-chunk FP32 state buffer round-trips through memory; it records the state entering each chunk in BF16,
  the operand the scan's dot consumes), and the scan producing y. Physical chunks never cross a sequence and are
  aligned to absolute positions (multiples of 128 from the sequence's start including its prefix), so a prefill
  split at a multiple of 128 reproduces the unsplit run bitwise. A requested checkpoint boundary also ends a chunk;
  the sequential pass writes the state after it to the checkpoint's slot and the final state to the sequence's own,
  having read the initial state from there first (zero without a prefix). FlashInfer's SSDCombined is tcgen05-only
  (SM100/103/110) with BF16 states, so this is Triton.
- Decode (`decode`), verify (`verify`) and replay (`AcceptedState.run`) share one recurrence step (`_advance`) and
  tile, so verify's outputs are bitwise decode's, and the state a replay commits for `count` accepted tokens is
  bitwise the state `count` decode steps would have produced (FlashInfer's CUDA selective_state_update runs on SM121
  at the same bandwidth, but its arithmetic would differ from a Triton replay's). Verify reads the committed slot and
  never writes it; it records either every step's state (the snapshots strategy) or, for reconstruction, the compact
  recurrence inputs: the convolution output (x, B, C) and dt, exp(dt A). The replay advances each committed slot by
  exactly its accepted count in one launch over all layers; a count of 0 leaves the slot untouched.
- The convolutions (`decode`'s update, `verify`'s candidate windows, the prefill's varlen pass) share one
  arithmetic (`_conv_value`: FP32 FMAs from the bias in tap order, SiLU in FP32, rounded to BF16 once), so a token
  convolves to the same bits on every path.
- `gated_rmsnorm` fuses the gate, the grouped RMS norm, the weight and the out projection's FP8 input quantization.
  The norm spans 16 heads, so it is not fused into the per-head recurrence.

A quantized pool (`SsmFormat` int8 / int4, every path's `window` argument) stores LeapQuant's per-window state
(`cache/ssm_format.py`): a quantized boundary state plus the window's committed records. Mamba-2's update is a
rank-one record that does not depend on the state (key B per group, value dt x per head, log decay dt A), so the
format's commit, fold, materialize and quantize kernels (`kernels/ssm_window.py`) serve it unchanged; only the
recurrence is this module's (`window_verify`). Verify reads the committed state through the expansion and writes one
record per step into the round's ring (`ssm_window.WindowState`, the reconstruct strategy's replay), decode is a
verify of one step committed at once, and the prefill runs the chunked kernels on FP32 copies of its slots' states and
quantizes the final states and checkpoints back as new boundary states.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sharar.cache.ssm_format import WindowLayout
from sharar.kernels.ssm_window import (_load_z, _parts, _split_dot, _strides, _tile, materialize, quantize,
                                       window_commit)

if TYPE_CHECKING:
    from sharar.engine.context import ForwardContext

# Nemotron-3.5-Super's mixer geometry (config.json llm_config); the kernels take it as arguments.
HEADS, HEAD_DIM, GROUPS, STATE = 128, 64, 8, 128
CONV_KERNEL, CHUNK = 4, 128
D_INNER = HEADS * HEAD_DIM  # 8192: z, x and y
CONV_DIM = D_INNER + 2 * GROUPS * STATE  # 10240: x | B | C
PROJ_DIM = D_INNER + CONV_DIM + HEADS  # 18560: z | xBC | dt
FP8_MAX = tl.constexpr(448.0)

# The recurrence tile: BLOCK_P rows of a head's [headdim, d_state] state per program. Decode and verify must use
# the same tile and warp count for their outputs to agree bitwise (the readout's reduction order follows the layout;
# the state update is elementwise). 32 x 8 warps: the verify reads the state at 225 GB/s against 16 x 4's 190 (B = 16,
# S = 5), decode at the same speed.
_SSU_BLOCK_P, _SSU_WARPS = 32, 8
# The prefill's inter-chunk recurrence tile [headdim rows, d_state columns].
_STATES_TILE = (64, 64)
# The prefill state's error against the FP32 recurrence, at 300 random tokens: relative 1.6e-3 with BF16 operands
# (mamba_ssm's choice), 4e-4 with TF32, 5e-6 with 3xTF32 for 8% more time in that kernel (1.69 -> 1.83 ms at 8192
# tokens). The state persists through every later decode step, so it gets the 3xTF32 dot.
_STATES_PRECISION = "tf32x3"
# FP16 state storage (SsmFormat fp16_sr): the Philox key and rounds of the stochastic rounding (NVIDIA's DGX
# Spark recipe for this model runs vLLM with 5 rounds).
_SEED = tl.constexpr(0x5348415241520001)
_PHILOX_ROUNDS = 5


@dataclass(frozen=True)
class Mamba2Params:
    """One layer's mixer parameters, in the dtypes the kernels read."""

    conv_weight: torch.Tensor  # [conv_dim, 4] bf16
    conv_bias: torch.Tensor  # [conv_dim] bf16
    A: torch.Tensor  # [H] fp32, -exp(A_log)
    D: torch.Tensor  # [H] fp32
    dt_bias: torch.Tensor  # [H] fp32
    norm_weight: torch.Tensor  # [d_inner] bf16
    eps: float
    groups: int = GROUPS
    head_dim: int = HEAD_DIM
    state: int = STATE

    @property
    def heads(self) -> int:
        return self.A.numel()

    @property
    def d_inner(self) -> int:
        return self.heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.groups * self.state


def prepare_params(conv_weight, conv_bias, A_log, D, dt_bias, norm_weight, eps: float, *, groups: int = GROUPS,
                   state: int = STATE) -> Mamba2Params:
    """Checkpoint tensors (any float dtype; conv1d.weight [C, 1, 4] or [C, 4]) -> `Mamba2Params`."""
    heads = A_log.numel()
    head_dim = norm_weight.numel() // heads
    return Mamba2Params(
        conv_weight=conv_weight.reshape(conv_weight.shape[0], -1).to(torch.bfloat16).contiguous(),
        conv_bias=conv_bias.to(torch.bfloat16).contiguous(),
        A=(-torch.exp(A_log.float())).contiguous(),
        D=D.float().contiguous(), dt_bias=dt_bias.float().contiguous(),
        norm_weight=norm_weight.to(torch.bfloat16).contiguous(), eps=float(eps),
        groups=groups, head_dim=head_dim, state=state,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Shared arithmetic


@triton.jit
def _dt(raw, bias):
    """softplus(raw + bias) with torch's threshold of 20, unclamped (time_step_limit (0, inf))."""
    v = raw + bias
    return tl.where(v <= 20.0, libdevice.log1p(tl.exp(v)), v)


@triton.jit
def _conv_value(bias, w0, w1, w2, w3, x0, x1, x2, x3):
    """SiLU(bias + w . x) in FP32, oldest tap first; explicit FMAs keep every kernel on the same rounding."""
    acc = tl.fma(w0, x0, bias)
    acc = tl.fma(w1, x1, acc)
    acc = tl.fma(w2, x2, acc)
    acc = tl.fma(w3, x3, acc)
    return acc / (1.0 + tl.exp(-acc))


@triton.jit
def _advance(state, x, b, dt, dA):
    """h' = exp(dt A) h + (dt x) B^T on a [BLOCK_P, N] tile."""
    return tl.fma(state, dA, (dt * x)[:, None] * b[None, :])


@triton.jit
def _readout(state, c, x, d):
    return tl.sum(state * c[None, :], axis=1) + d * x


@triton.jit
def _sr_fp16(v, r):
    """FP32 -> FP16 rounded up in magnitude with probability (distance to the value below) / (FP16 spacing), from
    the random word r: adds r's low bits below the FP16 grid and truncates (exact, unbiased), every binade including
    the FP16 subnormals; |v| < 2^-24 becomes 0 or 2^-24 with probability |v| 2^24. Saturates at +-65504. GB10 has
    no cvt.rs (SM100/103 only), so this is the integer form of that instruction."""
    bits = v.to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF).to(tl.int32)
    shift = tl.minimum(tl.maximum(126 - e, 13), 23).to(tl.uint32)  # FP32 mantissa bits below the FP16 grid
    mask = (tl.full(v.shape, 1, tl.uint32) << shift) - 1
    up = ((bits + (r & mask)) & ~mask).to(tl.float32, bitcast=True)
    tiny = tl.where((r >> 8).to(tl.float32) * 5.9604644775390625e-08 < tl.abs(v) * 16777216.0, 5.9604644775390625e-08,
                    0.0)
    out = tl.where(e < 103, tl.where(v < 0.0, -tiny, tiny), up)
    return tl.minimum(tl.maximum(out, -65504.0), 65504.0).to(tl.float16)


@triton.jit
def _round_state(state, rows, layer, pos, SR: tl.constexpr, ROUNDS: tl.constexpr):
    """The stored form of an FP32 state tile: itself (an FP32 pool) or, with SR (an FP16 pool), stochastically rounded
    FP16. rows [BM] is each tile row's offset in the layer's [H, P, N] state (its first column's element). Philox
    draws four words per counter (element // 4, position, layer index), one per element of four consecutive columns,
    so every path that stores the state after the token at `pos` draws the same bits, and a run reproduces itself
    whatever the slot, batch or launch."""
    if not SR:
        return state
    else:
        BM: tl.constexpr = state.shape[0]
        BN: tl.constexpr = state.shape[1]
        c0 = (rows[:, None] // 4 + tl.arange(0, BN // 4)[None, :]).to(tl.uint32)
        zero = c0 * 0
        r0, r1, r2, r3 = tl.philox(_SEED, c0, zero + pos.to(tl.uint32), zero + layer.to(tl.uint32), zero, ROUNDS)
        r = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), (BM, BN))
        return _sr_fp16(state, r)


# ---------------------------------------------------------------------------------------------------------------------
# Convolution


@triton.jit
def _conv_decode_kernel(X, sx, W, Bias, State, s_slot, s_ch, s_tap, Slots, Out, so,
                        C: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = ch < C
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    base = State + slot * s_slot + ch * s_ch
    h0 = tl.load(base, m, 0.0)
    h1 = tl.load(base + s_tap, m, 0.0)
    h2 = tl.load(base + 2 * s_tap, m, 0.0)
    x = tl.load(X + row * sx + ch, m, 0.0)
    out = _conv_value(tl.load(Bias + ch, m, 0.0).to(tl.float32),
                      tl.load(W + ch * 4, m, 0.0).to(tl.float32), tl.load(W + ch * 4 + 1, m, 0.0).to(tl.float32),
                      tl.load(W + ch * 4 + 2, m, 0.0).to(tl.float32), tl.load(W + ch * 4 + 3, m, 0.0).to(tl.float32),
                      h0.to(tl.float32), h1.to(tl.float32), h2.to(tl.float32), x.to(tl.float32))
    tl.store(Out + row * so + ch, out.to(Out.dtype.element_ty), m)
    tl.store(base, h1, m)
    tl.store(base + s_tap, h2, m)
    tl.store(base + 2 * s_tap, x, m)


@triton.jit
def _conv_verify_kernel(X, sx, W, Bias, State, s_slot, s_ch, s_tap, Slots, Out, so_row, so_step,
                        Win, sw_row, sw_step, sw_ch, sw_tap,
                        C: tl.constexpr, S: tl.constexpr, WINDOWS: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = ch < C
    slot = tl.load(Slots + row).to(tl.int64)
    base = State + slot * s_slot + ch * s_ch
    h0 = tl.load(base, m, 0.0)
    h1 = tl.load(base + s_tap, m, 0.0)
    h2 = tl.load(base + 2 * s_tap, m, 0.0)
    bias = tl.load(Bias + ch, m, 0.0).to(tl.float32)
    w0 = tl.load(W + ch * 4, m, 0.0).to(tl.float32)
    w1 = tl.load(W + ch * 4 + 1, m, 0.0).to(tl.float32)
    w2 = tl.load(W + ch * 4 + 2, m, 0.0).to(tl.float32)
    w3 = tl.load(W + ch * 4 + 3, m, 0.0).to(tl.float32)
    for t in tl.static_range(S):
        x = tl.load(X + (row * S + t) * sx + ch, m, 0.0)
        out = _conv_value(bias, w0, w1, w2, w3, h0.to(tl.float32), h1.to(tl.float32), h2.to(tl.float32),
                          x.to(tl.float32))
        tl.store(Out + row * so_row + t * so_step + ch, out.to(Out.dtype.element_ty), m)
        h0, h1, h2 = h1, h2, x
        if WINDOWS:
            wb = Win + row * sw_row + t * sw_step + ch * sw_ch
            tl.store(wb, h0, m)
            tl.store(wb + sw_tap, h1, m)
            tl.store(wb + 2 * sw_tap, h2, m)


@triton.jit
def _conv_prefill_tap(X, sx, State, s_tap, state_base, s0, t, n, ch, cm, init, J: tl.constexpr):
    """[BLOCK_T, BLOCK_C] inputs at local token t - 3 + J: the sequence's own, or its slot's history before it."""
    tt = t - 3 + J
    own = tl.load(X + (s0 + tt).to(tl.int64)[:, None] * sx + ch[None, :],
                  ((tt >= 0) & (tt < n))[:, None] & cm[None, :], 0.0)
    old = tl.load(state_base[None, :] + (tt + 3)[:, None] * s_tap, ((tt < 0) & init)[:, None] & cm[None, :], 0.0)
    return tl.where((tt < 0)[:, None], old, own).to(tl.float32)


@triton.jit
def _conv_prefill_kernel(X, sx, W, Bias, State, s_slot, s_ch, s_tap, Cu, Slots, HasInit, Out, so,
                         C: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    seq, tb = tl.program_id(0), tl.program_id(1)
    s0 = tl.load(Cu + seq)
    n = tl.load(Cu + seq + 1) - s0
    t0 = tb * BLOCK_T
    if t0 >= n:
        return
    ch = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = ch < C
    slot = tl.load(Slots + seq).to(tl.int64)
    init = tl.load(HasInit + seq) != 0
    state_base = State + slot * s_slot + ch * s_ch
    t = t0 + tl.arange(0, BLOCK_T)
    x0 = _conv_prefill_tap(X, sx, State, s_tap, state_base, s0, t, n, ch, cm, init, 0)
    x1 = _conv_prefill_tap(X, sx, State, s_tap, state_base, s0, t, n, ch, cm, init, 1)
    x2 = _conv_prefill_tap(X, sx, State, s_tap, state_base, s0, t, n, ch, cm, init, 2)
    x3 = _conv_prefill_tap(X, sx, State, s_tap, state_base, s0, t, n, ch, cm, init, 3)
    out = _conv_value(tl.load(Bias + ch, cm, 0.0).to(tl.float32)[None, :],
                      tl.load(W + ch * 4, cm, 0.0).to(tl.float32)[None, :],
                      tl.load(W + ch * 4 + 1, cm, 0.0).to(tl.float32)[None, :],
                      tl.load(W + ch * 4 + 2, cm, 0.0).to(tl.float32)[None, :],
                      tl.load(W + ch * 4 + 3, cm, 0.0).to(tl.float32)[None, :], x0, x1, x2, x3)
    tl.store(Out + (s0 + t).to(tl.int64)[:, None] * so + ch[None, :], out.to(Out.dtype.element_ty),
             (t < n)[:, None] & cm[None, :])
    if tb == 0:
        # The new conv state is the last three inputs of [history | sequence]; only this program reads the
        # history, so it alone rewrites the slot, after its reads.
        k = tl.arange(0, 4)
        idx = n - 3 + k
        km = k < 3
        own = tl.load(X + (s0 + idx).to(tl.int64)[:, None] * sx + ch[None, :],
                      ((idx >= 0) & km)[:, None] & cm[None, :], 0.0)
        old = tl.load(state_base[None, :] + (n + k)[:, None] * s_tap, ((idx < 0) & km & init)[:, None] & cm[None, :],
                      0.0)
        new = tl.where((idx >= 0)[:, None], own, old)
        tl.debug_barrier()
        tl.store(state_base[None, :] + k[:, None] * s_tap, new, km[:, None] & cm[None, :])


@triton.jit
def _conv_checkpoint_kernel(X, sx, State, s_slot, s_ch, s_tap, Ends, Slots, C: tl.constexpr, BLOCK_C: tl.constexpr):
    i = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = ch < C
    end = tl.load(Ends + i).to(tl.int64)
    base = State + tl.load(Slots + i).to(tl.int64) * s_slot + ch * s_ch
    for k in tl.static_range(3):
        tl.store(base + k * s_tap, tl.load(X + (end - 3 + k) * sx + ch, m), m)


def conv_decode(xbc: torch.Tensor, params: Mamba2Params, conv_state: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """One token per row: [B, conv_dim] (row stride free) -> [B, conv_dim] bf16; conv_state[slots] advanced."""
    b, c = xbc.shape
    out = torch.empty((b, c), dtype=torch.bfloat16, device=xbc.device)
    block = 512
    _conv_decode_kernel[(b, triton.cdiv(c, block))](
        xbc, xbc.stride(0), params.conv_weight, params.conv_bias, conv_state, *conv_state.stride(), slots, out,
        out.stride(0), c, block, num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Recurrence: decode, verify, replay


@triton.jit(do_not_specialize=["layer"])
def _ssu_decode_kernel(State, s_slot, Slots, XBC, sx, DtRaw, sdt, DtBias, A, Dskip, Y, sy, Pos, layer,
                       H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
                       BLOCK_P: tl.constexpr, SR: tl.constexpr, ROUNDS: tl.constexpr):
    pp, h, row = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    g = h // (H // G)
    op = pp * BLOCK_P + tl.arange(0, BLOCK_P)
    on = tl.arange(0, N)
    xrow = XBC + row * sx
    x = tl.load(xrow + h * P + op).to(tl.float32)
    b = tl.load(xrow + H * P + g * N + on).to(tl.float32)
    c = tl.load(xrow + H * P + G * N + g * N + on).to(tl.float32)
    dt = _dt(tl.load(DtRaw + row * sdt + h).to(tl.float32), tl.load(DtBias + h))
    dA = tl.exp(dt * tl.load(A + h))
    tile = h * P * N + op[:, None] * N + on[None, :]
    ptrs = State + slot * s_slot + tile
    state = _advance(tl.load(ptrs).to(tl.float32), x, b, dt, dA)
    pos = tl.load(Pos + row) if SR else 0
    tl.store(ptrs, _round_state(state, h * P * N + op * N, layer, pos, SR, ROUNDS))
    tl.store(Y + row * sy + h * P + op, _readout(state, c, x, tl.load(Dskip + h)).to(Y.dtype.element_ty))


@triton.jit(do_not_specialize=["layer"])
def _ssu_verify_kernel(State, s_slot, Slots, XBC, sx_row, sx_step, DtRaw, sdt, DtBias, A, Dskip, Y, sy,
                       Snap, s_snap_row, s_snap_step, SnapIdx, DtOut, DAOut, s_rec_row, s_rec_step,
                       Pos, PosOut, s_pos_row, layer,
                       H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, G: tl.constexpr, S: tl.constexpr,
                       SNAPSHOTS: tl.constexpr, RECORD: tl.constexpr,
                       BLOCK_P: tl.constexpr, SR: tl.constexpr, ROUNDS: tl.constexpr):
    pp, h, row = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Slots + row).to(tl.int64)
    g = h // (H // G)
    op = pp * BLOCK_P + tl.arange(0, BLOCK_P)
    on = tl.arange(0, N)
    tile = h * P * N + op[:, None] * N + on[None, :]
    state = tl.load(State + slot * s_slot + tile).to(tl.float32)  # the committed state is read, never written
    bias, a, d = tl.load(DtBias + h), tl.load(A + h), tl.load(Dskip + h)
    if SNAPSHOTS:
        snap = Snap + tl.load(SnapIdx + row).to(tl.int64) * s_snap_row + tile
    for t in tl.static_range(S):
        xrow = XBC + row * sx_row + t * sx_step
        x = tl.load(xrow + h * P + op).to(tl.float32)
        b = tl.load(xrow + H * P + g * N + on).to(tl.float32)
        c = tl.load(xrow + H * P + G * N + g * N + on).to(tl.float32)
        dt = _dt(tl.load(DtRaw + (row * S + t) * sdt + h).to(tl.float32), bias)
        dA = tl.exp(dt * a)
        state = _advance(state, x, b, dt, dA)
        tl.store(Y + (row * S + t) * sy + h * P + op, _readout(state, c, x, d).to(Y.dtype.element_ty))
        # The next step continues from the state as decode would have stored it.
        pos = tl.load(Pos + row * S + t) if SR else 0
        state = _round_state(state, h * P * N + op * N, layer, pos, SR, ROUNDS).to(tl.float32)
        if SNAPSHOTS:
            tl.store(snap + t * s_snap_step, state.to(Snap.dtype.element_ty))
        if RECORD:
            if pp == 0:
                tl.store(DtOut + row * s_rec_row + t * s_rec_step + h, dt)
                tl.store(DAOut + row * s_rec_row + t * s_rec_step + h, dA)
                if SR:
                    if h == 0:
                        tl.store(PosOut + row * s_pos_row + t, pos.to(tl.int32))


@triton.jit
def _ssu_replay_kernel(State, s_layer, s_slot, Slots, Counts, XBC, sx_layer, sx_row, sx_step,
                       Dt, DA, s_rec_layer, s_rec_row, s_rec_step, ROWS, Pos, s_pos_row,
                       H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, G: tl.constexpr, BLOCK_P: tl.constexpr,
                       SR: tl.constexpr, ROUNDS: tl.constexpr):
    pp, h, lr = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, row = (lr // ROWS).to(tl.int64), lr % ROWS
    count = tl.load(Counts + row)
    if count <= 0:
        return
    slot = tl.load(Slots + row).to(tl.int64)
    g = h // (H // G)
    op = pp * BLOCK_P + tl.arange(0, BLOCK_P)
    on = tl.arange(0, N)
    tile = h * P * N + op[:, None] * N + on[None, :]
    ptrs = State + layer * s_layer + slot * s_slot + tile
    state = tl.load(ptrs).to(tl.float32)
    xbase = XBC + layer * sx_layer + row * sx_row
    rbase = layer * s_rec_layer + row * s_rec_row + h
    for t in range(count):
        x = tl.load(xbase + t * sx_step + h * P + op).to(tl.float32)
        b = tl.load(xbase + t * sx_step + H * P + g * N + on).to(tl.float32)
        state = _advance(state, x, b, tl.load(Dt + rbase + t * s_rec_step), tl.load(DA + rbase + t * s_rec_step))
        if SR:  # each step rounds as decode's store does
            pos = tl.load(Pos + row * s_pos_row + t)
            state = _round_state(state, h * P * N + op * N, layer, pos, SR, ROUNDS).to(tl.float32)
    tl.store(ptrs, state.to(State.dtype.element_ty))


def _check_state(ssm_state: torch.Tensor, params: Mamba2Params, positions: torch.Tensor | None = None) -> bool:
    """Whether the state is stored stochastically rounded (FP16; FP32 is stored as computed), after checking its layout
    and the positions the rounding keys on."""
    if (ssm_state.dtype not in (torch.float32, torch.float16)
            or ssm_state.shape[1:] != (params.heads, params.head_dim, params.state) or not ssm_state[0].is_contiguous()):
        raise ValueError("the SSM state must be FP32 or FP16 [slots, heads, headdim, d_state] with contiguous slots")
    sr = ssm_state.dtype == torch.float16
    if sr and positions is None:
        raise ValueError("a stochastically rounded FP16 state needs the token positions")
    return sr


def _split(zxbcdt: torch.Tensor, params: Mamba2Params):
    """z, xBC and dt as strided views of the in_proj output."""
    if zxbcdt.ndim != 2 or zxbcdt.stride(1) != 1:
        raise ValueError("the in_proj output must be a row-major [tokens, z | xBC | dt] matrix")
    di, c = params.d_inner, params.conv_dim
    return zxbcdt[:, :di], zxbcdt[:, di:di + c], zxbcdt[:, di + c:]


def decode(zxbcdt: torch.Tensor, params: Mamba2Params, conv_state: torch.Tensor, ssm_state: torch.Tensor,
           slots: torch.Tensor, positions: torch.Tensor | None = None, *, layer: int = 0,
           window: tuple[WindowLayout, dict] | None = None) -> torch.Tensor:
    """One token per row: conv state and SSM state of slots[i] advanced in place; returns y [B, d_inner] bf16
    (before the gated norm). Rows whose slot is negative are skipped.

    An FP16 `ssm_state` is stored stochastically rounded, keyed on positions[i] (the row's token position) and
    `layer` (the layer's index among the Mamba layers); y reads the FP32 state.
    With window = (layout, this layer's region views) the pool is quantized: the token runs as a verify of one step
    and its record is committed at once (appended to the window, or folding a full one)."""
    _, xbc, dt = _split(zxbcdt, params)
    b = zxbcdt.shape[0]
    if window is not None:
        layout, views = window
        conv = conv_decode(xbc, params, conv_state, slots)
        dev = zxbcdt.device
        ring_k = torch.empty((1, b, params.groups, 1, params.state), dtype=torch.float16, device=dev)
        ring_u = torch.empty((1, b, params.heads, 1, params.head_dim), dtype=torch.float16, device=dev)
        ring_g = torch.empty((1, b, params.heads, 1), dtype=torch.float32, device=dev)
        y = window_verify(layout, views, conv.unsqueeze(1), dt, params, slots, 1, ring_k[0], ring_u[0], ring_g[0])
        window_commit(layout, {name: t.unsqueeze(0) for name, t in views.items()}, ring_k, ring_u, ring_g, slots,
                      torch.ones(b, dtype=torch.int32, device=dev))
        return y
    sr = _check_state(ssm_state, params, positions)
    conv = conv_decode(xbc, params, conv_state, slots)
    y = torch.empty((b, params.d_inner), dtype=torch.bfloat16, device=zxbcdt.device)
    _ssu_decode_kernel[(params.head_dim // _SSU_BLOCK_P, params.heads, b)](
        ssm_state, ssm_state.stride(0), slots, conv, conv.stride(0), dt, dt.stride(0), params.dt_bias, params.A,
        params.D, y, y.stride(0), positions if sr else slots, layer,
        params.heads, params.head_dim, params.state, params.groups,
        BLOCK_P=_SSU_BLOCK_P, SR=sr, ROUNDS=_PHILOX_ROUNDS, num_warps=_SSU_WARPS)
    return y


def verify(zxbcdt: torch.Tensor, params: Mamba2Params, conv_state: torch.Tensor, ssm_state: torch.Tensor,
           slots: torch.Tensor, steps: int, conv_windows: torch.Tensor | None = None, *,
           snapshots: torch.Tensor | None = None, snapshot_indices: torch.Tensor | None = None,
           replay: dict | None = None, positions: torch.Tensor | None = None, layer: int = 0,
           window: tuple[WindowLayout, dict] | None = None) -> torch.Tensor:
    """Speculative verify: B rows of `steps` tokens each ([B * steps, proj] in row-major order), each row starting
    from the committed state of slots[i], which is left untouched (conv and SSM). Returns y [B * steps, d_inner].

    conv_windows [capacity, >= steps, conv_dim, 3] receives each row's convolution window after every step.
    snapshots [scratch, >= steps, H, P, N] (with snapshot_indices [B]) receives the SSM state after every step;
    replay (an `AcceptedState.layer_inputs` entry) instead receives the compact inputs `AcceptedState.run` needs.
    An FP16 state is rounded after every step as `decode` stores it (positions [B * steps], `layer` as there), so
    every output, snapshot and replayed state is bitwise decode's.

    With window = (layout, this layer's region views) the pool is quantized (`window_verify`): replay is then a
    `ssm_window.WindowState` layer entry, whose ring receives each step's record, and there are no snapshots.
    """
    _, xbc, dt = _split(zxbcdt, params)
    rows_total, c = xbc.shape
    if steps < 1 or rows_total % steps:
        raise ValueError("verify needs B rows of `steps` tokens")
    b = rows_total // steps
    if window is not None and (replay is None or "ring_u" not in replay or snapshots is not None):
        raise ValueError("a quantized window verify records into a WindowState ring and takes no snapshots")
    sr = window is None and _check_state(ssm_state, params, positions)
    if replay is not None and window is None:
        conv = replay["xbc"]
        if conv.shape[0] < b or conv.shape[1] < steps or conv.shape[2] != c:
            raise ValueError("replay buffers are narrower than this verify")
    else:
        conv = torch.empty((b, steps, c), dtype=torch.bfloat16, device=xbc.device)
    windows = conv_windows is not None
    if windows and (conv_windows.shape[0] < b or conv_windows.shape[1] < steps or conv_windows.shape[2:] != (c, 3)):
        raise ValueError("invalid verify conv window buffer")
    win = conv_windows if windows else conv
    block = 512
    _conv_verify_kernel[(b, triton.cdiv(c, block))](
        xbc, xbc.stride(0), params.conv_weight, params.conv_bias, conv_state, *conv_state.stride(), slots,
        conv, conv.stride(0), conv.stride(1), win, *(win.stride() if windows else (0, 0, 0, 0)),
        c, steps, windows, block, num_warps=4)
    if window is not None:
        return window_verify(*window, conv, dt, params, slots, steps, replay["ring_k"], replay["ring_u"],
                             replay["ring_g"])
    y = torch.empty((rows_total, params.d_inner), dtype=torch.bfloat16, device=xbc.device)
    snap = snapshots is not None
    if snap and (snapshot_indices is None or snapshots.shape[1] < steps or snapshots.shape[2:] != ssm_state.shape[1:]):
        raise ValueError("invalid verify snapshot buffer")
    rec = replay is not None
    dt_out, da_out = (replay["dt"], replay["dA"]) if rec else (y, y)
    pos_out = replay["pos"] if rec and sr else slots
    if rec and sr and (pos_out.shape[0] < b or pos_out.shape[1] < steps):
        raise ValueError("replay buffers are narrower than this verify")
    _ssu_verify_kernel[(params.head_dim // _SSU_BLOCK_P, params.heads, b)](
        ssm_state, ssm_state.stride(0), slots, conv, conv.stride(0), conv.stride(1), dt, dt.stride(0),
        params.dt_bias, params.A, params.D, y, y.stride(0),
        snapshots if snap else y, snapshots.stride(0) if snap else 0, snapshots.stride(1) if snap else 0,
        snapshot_indices if snap else slots, dt_out, da_out,
        dt_out.stride(0) if rec else 0, dt_out.stride(1) if rec else 0,
        positions if sr else slots, pos_out, pos_out.stride(0) if pos_out.ndim == 2 else 0, layer,
        params.heads, params.head_dim, params.state, params.groups, steps, snap, rec,
        BLOCK_P=_SSU_BLOCK_P, SR=sr, ROUNDS=_PHILOX_ROUNDS, num_warps=_SSU_WARPS)
    return y


def replay(ssm_all: torch.Tensor, xbc: torch.Tensor, dt: torch.Tensor, dA: torch.Tensor, slots: torch.Tensor,
           counts: torch.Tensor, groups: int = GROUPS, positions: torch.Tensor | None = None) -> None:
    """Advance ssm_all[layer, slots[i]] by counts[i] recorded verify steps, for every layer in one launch.

    ssm_all [L, slots, H, P, N]; xbc [L, capacity, steps, conv_dim]; dt, dA [L, capacity, steps, H]. Row i of this
    batch is capacity row i. Slots must be distinct (each program reads its tile before writing it); counts outside
    [1, steps] other than 0 are the caller's error, and 0 leaves the slot untouched. An FP16 pool rounds every step
    as decode does: it reads the verified positions [capacity, steps] (int32) and keys on the layer's index in
    ssm_all."""
    layers, _, heads, head_dim, state = ssm_all.shape
    rows = slots.numel()
    if counts.numel() != rows or xbc.shape[0] != layers or dt.shape != dA.shape or dt.shape[-1] != heads:
        raise ValueError("invalid replay geometry")
    sr = ssm_all.dtype == torch.float16
    if sr and (positions is None or positions.dtype != torch.int32 or positions.ndim != 2):
        raise ValueError("a stochastically rounded FP16 replay needs the int32 [capacity, steps] verify positions")
    _ssu_replay_kernel[(head_dim // _SSU_BLOCK_P, heads, layers * rows)](
        ssm_all, ssm_all.stride(0), ssm_all.stride(1), slots, counts, xbc, xbc.stride(0), xbc.stride(1),
        xbc.stride(2), dt, dA, dt.stride(0), dt.stride(1), dt.stride(2), rows,
        positions if sr else slots, positions.stride(0) if sr else 0,
        heads, head_dim, state, groups, BLOCK_P=_SSU_BLOCK_P, SR=sr, ROUNDS=_PHILOX_ROUNDS,
        num_warps=_SSU_WARPS)


@triton.jit
def _conv_commit_kernel(Conv, sc_layer, sc_slot, Win, sw_layer, sw_row, sw_step, Slots, Counts, ROWS,
                        SIZE: tl.constexpr, BLOCK: tl.constexpr):
    lr = tl.program_id(0)
    layer, row = (lr // ROWS).to(tl.int64), lr % ROWS
    count = tl.load(Counts + row)
    if count <= 0:
        return
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < SIZE
    src = Win + layer * sw_layer + row * sw_row + (count - 1).to(tl.int64) * sw_step + i
    tl.store(Conv + layer * sc_layer + tl.load(Slots + row).to(tl.int64) * sc_slot + i, tl.load(src, m), m)


def commit_conv(conv_all: torch.Tensor, windows_all: torch.Tensor, slots: torch.Tensor, counts: torch.Tensor) -> None:
    """conv_all[layer, slots[i]] = windows_all[layer, i, counts[i] - 1] for every layer in one launch; 0 skips.

    conv_all [L, slots, conv_dim, 3] and windows_all [L, capacity, steps, conv_dim, 3] with contiguous windows: the
    graph-safe form of the accept step's per-row copies."""
    layers, _, c, k = conv_all.shape
    if not conv_all[0, 0].is_contiguous() or not windows_all[0, 0, 0].is_contiguous():
        raise ValueError("conv windows must be contiguous per step")
    rows, size, block = slots.numel(), c * k, 4096
    _conv_commit_kernel[(layers * rows, triton.cdiv(size, block))](
        conv_all, conv_all.stride(0), conv_all.stride(1), windows_all, windows_all.stride(0), windows_all.stride(1),
        windows_all.stride(2), slots, counts, rows, size, block, num_warps=4)


class AcceptedState:
    """Compact verify inputs and the accept-step replay; a drop-in for `kernels.fla.accepted_state.AcceptedState`.

    `layer_inputs[layer_id]` is the `replay` argument of that layer's `verify` (the engine passes it as
    ctx.verify_replay[layer_id]): the convolution output (x | B | C) and dt, exp(dt A) of every verified step, and
    (shared by every layer) the verified positions a stochastically rounded FP16 state keys on.
    `run` advances each committed slot by its accepted count from them, every layer in one launch, on the caller's
    stream and with no host synchronization (it is captured inside the accept graphs)."""

    def __init__(self, pool, batch: int, steps: int, key_heads: int = GROUPS, *, cuda: bool = False):
        if cuda:
            raise ValueError('Mamba accepted-state reconstruction uses its Triton verify arithmetic')
        layers, _, heads, _, _ = pool.ssm_all.shape
        conv_dim = pool.conv_all.shape[2]
        device = pool.ssm_all.device
        self.batch, self.steps, self.groups = batch, steps, key_heads
        self.xbc = torch.empty((layers, batch, steps, conv_dim), dtype=torch.bfloat16, device=device)
        self.dt = torch.empty((layers, batch, steps, heads), dtype=torch.float32, device=device)
        self.dA = torch.empty_like(self.dt)
        self.pos = torch.zeros((batch, steps), dtype=torch.int32, device=device)
        self.layer_inputs = {lid: {"xbc": self.xbc[j], "dt": self.dt[j], "dA": self.dA[j], "pos": self.pos}
                             for j, lid in enumerate(pool.layer_ids)}

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.xbc, self.dt, self.dA, self.pos))

    def run(self, pool, slots: torch.Tensor, counts: torch.Tensor) -> None:
        replay(pool.ssm_all, self.xbc, self.dt, self.dA, slots, counts, self.groups, self.pos)


# ---------------------------------------------------------------------------------------------------------------------
# Quantized window state (SsmFormat int8 / int4)


@triton.jit
def _window_verify_kernel(XBC, sx_row, sx_step, DtRaw, sdt, DtBias, A, Dskip, Y, sy,
                          z, scale, smooth, comp_k, comp_u, log_k, log_u, log_g, log_len, Slots,
                          ring_k, ring_u, ring_g,
                          s_z, s_sc, s_sm, s_ck, s_cu, s_lk, s_lu, s_lg, s_ll,
                          H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, G: tl.constexpr, S: tl.constexpr,
                          BT: tl.constexpr, WIN: tl.constexpr, R: tl.constexpr, BJ: tl.constexpr, RS: tl.constexpr,
                          BKC: tl.constexpr, BITS: tl.constexpr):
    """One head of one row, all S steps at once. The committed state is read as `ssm_window` reads the GDN's, without
    expanding it: S_0 c = e^(G_n) [((c C) Z^T) B + (c K~^T) U~^T] + sum_j e^(G_n - G_j) (c k_j) u_j. The verified
    steps then add their own records in the chunked form: y_t = e^(g_t) S_0 c_t + sum_(j <= t) e^(g_t - g_j)
    (c_t . B_j) u_j + D x_t, g_t the cumulative dt A of steps 1..t and u_j = dt_j x_j, with the same FP16 records the
    ring receives (B is BF16, exact in FP16), so the state the commit builds is the one these outputs read."""
    h, row = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot < 0:
        return
    g = h // (H // G)
    o_p = tl.arange(0, P)
    o_n = tl.arange(0, N)
    o_t = tl.arange(0, BT)
    mt = o_t < S
    xrow = XBC + row * sx_row + o_t * sx_step
    x = tl.load(xrow[:, None] + h * P + o_p[None, :], mt[:, None], 0.0).to(tl.float32)  # [BT, P]
    bh = tl.load(xrow[:, None] + H * P + g * N + o_n[None, :], mt[:, None], 0.0).to(tl.float16)  # [BT, N]
    ch = tl.load(xrow[:, None] + H * P + G * N + g * N + o_n[None, :], mt[:, None], 0.0).to(tl.float16)
    dt = _dt(tl.load(DtRaw + (row * S + o_t) * sdt + h, mt, 0.0).to(tl.float32), tl.load(DtBias + h))
    dt = tl.where(mt, dt, 0.0)
    lg_t = dt * tl.load(A + h)
    uh = (dt[:, None] * x).to(tl.float16)
    # This round's records, for the commit.
    tl.store(ring_g + (row * H + h) * RS + o_t, lg_t, mt)
    tl.store(ring_u + (row * H + h) * RS * P + o_t[:, None] * P + o_p[None, :], uh, mt[:, None])
    if h % (H // G) == 0:  # B once per group
        tl.store(ring_k + (row * G + g) * RS * N + o_t[:, None] * N + o_n[None, :], bh, mt[:, None])

    # ((c C) Z^T) of the integer residual, K chunk by chunk; c C as two FP16 halves against the exact integers.
    zb = z + slot * s_z + h * P * (N * BITS // 8)
    acc = tl.zeros([BT, P], tl.float32)
    for kc in tl.static_range(N // BKC):
        o_c = kc * BKC + tl.arange(0, BKC)
        zt = tl.trans(_load_z(zb, o_p, kc * BKC, BKC, N, BITS))  # [BKC, P]
        cc = tl.load(xrow[:, None] + H * P + G * N + g * N + o_c[None, :], mt[:, None], 0.0).to(tl.float32)
        acc += _split_dot(cc * tl.load(smooth + slot * s_sm + h * N + o_c)[None, :], zt)
    # Records: rows j < WIN are the window's committed tokens (j < n), rows WIN..WIN+R-1 the compensators.
    n = tl.minimum(tl.load(log_len + slot * s_ll + h), WIN)
    o_j = tl.arange(0, BJ)
    is_l = o_j < n
    is_c = (o_j >= WIN) & (o_j < WIN + R)
    rec = is_l | is_c
    p_k = tl.where(is_c, comp_k + slot * s_ck + (h * R + o_j - WIN) * N, log_k + slot * s_lk + (g * WIN + o_j) * N)
    p_u = tl.where(is_c, comp_u + slot * s_cu + (h * R + o_j - WIN) * P, log_u + slot * s_lu + (h * WIN + o_j) * P)
    lk = tl.load(p_k[:, None] + o_n[None, :], rec[:, None], 0.0)  # [BJ, N]
    lu = tl.load(p_u[:, None] + o_p[None, :], rec[:, None], 0.0)  # [BJ, P]
    lg = tl.load(log_g + slot * s_lg + h * WIN + o_j, is_l, 0.0)
    gn = tl.sum(lg, 0)
    en = tl.exp(gn)
    w = tl.where(is_l, tl.exp(gn - tl.cumsum(lg, 0)), tl.where(is_c, en, 0.0))
    sc = tl.load(scale + slot * s_sc + h * P + o_p) * en
    s0c = _split_dot(tl.dot(ch, tl.trans(lk)) * w[None, :], lu) + acc * sc[None, :]  # [BT, P]: S_0 c_t

    gc = tl.cumsum(lg_t, 0)
    causal = o_t[:, None] >= o_t[None, :]
    decay = tl.exp(tl.where(causal, gc[:, None] - gc[None, :], 0.0))
    cb = tl.where(causal, decay * tl.dot(ch, tl.trans(bh)), 0.0)
    y = tl.exp(gc)[:, None] * s0c + _split_dot(cb, uh) + tl.load(Dskip + h) * x
    tl.store(Y + (row * S + o_t)[:, None] * sy + h * P + o_p[None, :], y.to(Y.dtype.element_ty), mt[:, None])


def window_verify(layout: WindowLayout, views: dict[str, torch.Tensor], conv: torch.Tensor, dt: torch.Tensor,
                  params: Mamba2Params, slots: torch.Tensor, steps: int, ring_k, ring_u, ring_g) -> torch.Tensor:
    """The recurrence on a quantized window pool: B rows of `steps` convolved tokens conv [B, >= steps, conv_dim]
    (any row and step strides) with raw dt [B * steps, H] (row stride free), each row reading the committed state of
    slots[i] (this layer's region `views`) and never writing it. ring_* [>= B, heads, >= steps, ...] (a
    `ssm_window.WindowState` layer entry) receive one record per step for the commit. Returns y [B * steps, d_inner]
    bf16 before the gated norm; rows whose slot is negative are skipped."""
    b = slots.numel()
    if (layout.key_heads, layout.value_heads, layout.k, layout.v) != (params.groups, params.heads, params.state,
                                                                      params.head_dim):
        raise ValueError("window verify: the layout is not this mixer's geometry")
    if steps > ring_u.shape[2] or b > ring_u.shape[0] or conv.shape[0] < b or conv.shape[1] < steps:
        raise ValueError("window verify: the ring or the convolution output is smaller than the verified block")
    y = torch.empty((b * steps, params.d_inner), dtype=torch.bfloat16, device=conv.device)
    _window_verify_kernel[(params.heads, b)](
        conv, conv.stride(0), conv.stride(1), dt, dt.stride(0), params.dt_bias, params.A, params.D, y, y.stride(0),
        *_parts(views), slots, ring_k, ring_u, ring_g, *_strides(views, 0),
        params.heads, params.head_dim, params.state, params.groups, steps, _tile(steps), layout.window,
        layout.rank, _tile(layout.window + layout.rank), ring_u.shape[2], 64, layout.bits, num_warps=4)
    return y


# ---------------------------------------------------------------------------------------------------------------------
# Gated RMS norm (+ FP8 quantization of the out projection's input)


@triton.jit
def _gated_norm_kernel(Y, sy, Z, sz, W, Out, so, Scale, eps, GROUP: tl.constexpr, QUANT: tl.constexpr):
    row, g = tl.program_id(0).to(tl.int64), tl.program_id(1)
    cols = g * GROUP + tl.arange(0, GROUP)
    z = tl.load(Z + row * sz + cols).to(tl.float32)
    h = tl.load(Y + row * sy + cols).to(tl.float32) * (z / (1.0 + tl.exp(-z)))
    h = h * tl.rsqrt(tl.sum(h * h, 0) / GROUP + eps)
    # transformers: weight * hidden.to(bf16), a BF16 product
    v = (tl.load(W + cols).to(tl.float32) * h.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if QUANT:
        q = v.to(tl.float32) / tl.load(Scale)
        tl.store(Out + row * so + cols, tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX).to(tl.float8e4nv))
    else:
        tl.store(Out + row * so + cols, v)


def gated_rmsnorm(y: torch.Tensor, z: torch.Tensor, params: Mamba2Params,
                  out_scale: torch.Tensor | None = None) -> torch.Tensor:
    """norm(y * silu(z)) * weight over groups of d_inner / groups; FP8 (x / out_scale) when out_scale is given."""
    t, n = y.shape
    group = n // params.groups
    out = torch.empty((t, n), dtype=torch.float8_e4m3fn if out_scale is not None else torch.bfloat16, device=y.device)
    if t:
        _gated_norm_kernel[(t, params.groups)](
            y, y.stride(0), z, z.stride(0), params.norm_weight, out, out.stride(0),
            out_scale if out_scale is not None else params.A, params.eps, group, out_scale is not None, num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Prefill: chunked SSD over packed varlen sequences


@dataclass(frozen=True)
class Mamba2PrefillPlan:
    """Host-planned metadata shared by every Mamba layer of one prefill forward (one upload)."""

    tokens: int
    seq_lens: tuple[int, ...]
    num_chunks: int
    num_checkpoints: int
    cu_seqlens: torch.Tensor  # int32 [S + 1]
    slots: torch.Tensor  # int32 [S] (the caller's device tensor when given one)
    has_initial_state: torch.Tensor  # int32 [S]
    cu_chunks: torch.Tensor  # int32 [C + 1], token offsets of the physical chunks
    seq_chunks: torch.Tensor  # int32 [S + 1], chunk offsets of the sequences
    chunk_dest: torch.Tensor  # int32 [C], checkpoint slot receiving the state after the chunk, or -1
    chunk_last: torch.Tensor  # int32 [C], absolute position of the chunk's last token (an FP16 state's rounding key)
    checkpoint_ends: torch.Tensor  # int32 [K], packed token index each checkpoint ends at
    checkpoint_slots: torch.Tensor  # int32 [K]
    # A quantized pool's prefill runs on FP32 scratch states: the sequences' in rows 0..S-1, then one per distinct
    # checkpoint boundary.
    scratch_rows: int
    seq_rows: torch.Tensor  # int32 [S], 0..S-1
    chunk_scratch: torch.Tensor  # int32 [C], scratch row receiving the state after the chunk, or -1
    checkpoint_rows: torch.Tensor  # int32 [K], each checkpoint's scratch row


def plan_prefill(seq_lens, prefix_lens, slots, device, checkpoints=(), chunk: int = CHUNK) -> Mamba2PrefillPlan:
    """seq_lens / prefix_lens: per sequence (host ints); slots: host ints or an int32 device tensor (read by the
    kernels only); checkpoints: (sequence index, local end, slot) with 3 <= local end < its length, the state after
    that many of the sequence's new tokens going to the slot. Nothing here reads a device value."""
    seq_lens, prefix_lens = tuple(seq_lens), tuple(prefix_lens)
    device_slots = slots if isinstance(slots, torch.Tensor) else None
    if device_slots is not None and (device_slots.dtype != torch.int32 or device_slots.shape != (len(seq_lens),)):
        raise ValueError("Mamba prefill slots must be int32 [sequences]")
    host_slots = () if device_slots is not None else tuple(slots)
    if (not seq_lens or any(n <= 0 for n in seq_lens) or len(prefix_lens) != len(seq_lens)
            or (device_slots is None and len(host_slots) != len(seq_lens))):
        raise ValueError("Mamba prefill needs positive CPU sequence lengths, one prefix and slot each")
    starts = (0, *accumulate(seq_lens))
    cuts = [set() for _ in seq_lens]
    ends, ck_slots, ck_rows, dest_at, row_at = [], [], [], {}, {}
    for i, local, slot in checkpoints:
        if not 3 <= local < seq_lens[i]:
            raise ValueError("Mamba checkpoint must end at an interior token >= 3")
        cuts[i].add(local)
        ends.append(starts[i] + local)
        ck_slots.append(slot)
        dest_at[(i, local)] = slot
        ck_rows.append(row_at.setdefault((i, local), len(seq_lens) + len(row_at)))
    cu_chunks, seq_chunks, dest, scratch, last = [0], [0], [], [], []
    for i, (n, p) in enumerate(zip(seq_lens, prefix_lens)):
        first = (chunk - p % chunk) % chunk or chunk
        bounds = sorted(cuts[i] | set(range(first, n, chunk)) | {n})
        for b in bounds:
            cu_chunks.append(starts[i] + b)
            dest.append(dest_at.get((i, b), -1))
            scratch.append(row_at.get((i, b), -1))
            last.append(p + b - 1)
        seq_chunks.append(len(dest))
    packed = [*starts, *host_slots, *(int(p > 0) for p in prefix_lens), *cu_chunks, *seq_chunks, *dest, *last, *ends,
              *ck_slots, *range(len(seq_lens)), *scratch, *ck_rows]
    from sharar.engine.upload import upload
    flat = upload(packed, torch.int32, device)
    views, offset = [], 0
    for size in (len(starts), len(host_slots), len(seq_lens), len(cu_chunks), len(seq_chunks), len(dest), len(last),
                 len(ends), len(ends), len(seq_lens), len(scratch), len(ck_rows)):
        views.append(flat[offset:offset + size])
        offset += size
    if device_slots is not None:
        views[1] = device_slots
    return Mamba2PrefillPlan(starts[-1], seq_lens, len(dest), len(ends), *views[:9], len(seq_lens) + len(row_at),
                             *views[9:])



@dataclass
class SideInputs:
    """A prefill forward's side input (`ctx.extras`): the plan its first Mamba layer makes for the rest."""
    plan: Mamba2PrefillPlan | None = None


def prepare_prefill(ctx: ForwardContext | None, tokens: int, device) -> Mamba2PrefillPlan:
    """The immutable prefill plan in the model's side inputs, shared by every recurrent layer."""
    if ctx is not None:
        if ctx.extras is None:  # stand-alone parity forwards use the same side input as the executor
            ctx.extras = SideInputs()
        if ctx.extras.plan is not None:
            return ctx.extras.plan
    if ctx is not None and ctx.batched:
        lengths, prefixes = list(ctx.seq_lens), list(ctx.seq_prefix_lens)
        slots = ctx.seq_slots.to(torch.int32)
        seqs = list(ctx.checkpoint_seqs)
    else:
        lengths = [tokens]
        prefixes = [ctx.prefix_len if ctx is not None else 0]
        slots = [ctx.slot if ctx is not None and ctx.state_pool is not None else 0]
        seqs = [0] * (len(ctx.checkpoint_positions) if ctx is not None else 0)
    checkpoints = []
    if ctx is not None:
        for pos, slot, i in zip(ctx.checkpoint_positions, ctx.checkpoint_slots, seqs):
            checkpoints.append((i, pos - prefixes[i], slot))
    plan = plan_prefill(lengths, prefixes, slots, device, checkpoints)
    if ctx is not None:
        ctx.extras.plan = plan
    return plan


@triton.jit
def _chunk_cumsum_kernel(DtRaw, sdt, DtBias, A, DtOut, DACs, CuChunks, nchunks,
                         H: tl.constexpr, CS: tl.constexpr, BLOCK_H: tl.constexpr):
    c = tl.program_id(0).to(tl.int64)
    oh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    ot = tl.arange(0, CS)
    t0 = tl.load(CuChunks + c)
    n = tl.load(CuChunks + c + 1) - t0
    hm = oh < H
    valid = hm[:, None] & (ot[None, :] < n)
    raw = tl.load(DtRaw + (t0 + ot)[None, :].to(tl.int64) * sdt + oh[:, None], valid, 0.0).to(tl.float32)
    dt = _dt(raw, tl.load(DtBias + oh, hm, 0.0)[:, None])
    dt = tl.where(valid, dt, 0.0)
    out = oh[:, None].to(tl.int64) * nchunks * CS + c * CS + ot[None, :]
    tl.store(DtOut + out, dt, hm[:, None])
    tl.store(DACs + out, tl.cumsum(dt * tl.load(A + oh, hm, 0.0)[:, None], axis=1), hm[:, None])


@triton.jit(do_not_specialize=["layer"])
def _chunk_states_kernel(X, B, sx, Dt, DACs, Entering, CuChunks, SeqChunks, Slots, HasInit, ChunkDest, ChunkLast,
                         Pool, s_slot, nchunks, layer, H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
                         G: tl.constexpr, CS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, PRECISION: tl.constexpr, SR: tl.constexpr, ROUNDS: tl.constexpr):
    """The inter-chunk recurrence of one sequence for one [BLOCK_M, BLOCK_N] tile of a head's state, walking its
    chunks in order: each chunk's own contribution sum_t exp(dA_last - dA_t) dt_t x_t B_t^T is formed in registers
    and folded into the running FP32 state, so no per-chunk state buffer round-trips through memory. PRECISION is
    the contribution's dot's input precision (`_STATES_PRECISION`).
    Entering[c] (BF16, what the scan's dot consumes) receives the state entering chunk c, ChunkDest[c] the state
    after it (a checkpoint), and the sequence's slot the final state; the initial state is read from that slot
    first, by the only program that writes this tile of it. An FP16 pool (SR) rounds the running state at
    every chunk end as decode would store the state after that chunk's last token; chunks sit on the absolute
    128-token grid, so a prefill split on it still reproduces the unsplit run bitwise."""
    pid, seq, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pid_m, pid_n = pid // tl.cdiv(N, BLOCK_N), pid % tl.cdiv(N, BLOCK_N)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    tile = h * P * N + om[:, None] * N + on[None, :]
    c0, c1 = tl.load(SeqChunks + seq), tl.load(SeqChunks + seq + 1)
    init = tl.load(HasInit + seq) != 0
    slot = tl.load(Slots + seq).to(tl.int64)
    state = tl.load(Pool + slot * s_slot + tile, init & (om[:, None] < P), 0.0).to(tl.float32)
    x_col = X + h * P + om[:, None]
    b_col = B + (h // (H // G)) * N + on[None, :]
    for c in range(c0, c1):
        tl.store(Entering + c.to(tl.int64) * H * P * N + tile, state.to(Entering.dtype.element_ty))
        t0 = tl.load(CuChunks + c).to(tl.int64)
        n = tl.load(CuChunks + c + 1) - t0
        hc = (h.to(tl.int64) * nchunks + c) * CS
        da_last = tl.load(DACs + hc + CS - 1)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, n, BLOCK_K):
            km = ok < n - k
            rows = (t0 + k + ok).to(tl.int64) * sx
            x = tl.load(x_col + rows[None, :], km[None, :], 0.0)
            b = tl.load(b_col + rows[:, None], km[:, None], 0.0).to(tl.float32)
            scale = tl.exp(tl.minimum(da_last - tl.load(DACs + hc + k + ok, km, 0.0), 0.0)) * tl.load(
                Dt + hc + k + ok, km, 0.0)
            acc += tl.dot(x.to(tl.float32), b * scale[:, None], input_precision=PRECISION)
        state = tl.exp(da_last) * state + acc
        if SR:
            pos = tl.load(ChunkLast + c)
            state = _round_state(state, h * P * N + om * N + pid_n * BLOCK_N, layer, pos, SR, ROUNDS).to(tl.float32)
        dest = tl.load(ChunkDest + c).to(tl.int64)
        if dest >= 0:
            tl.store(Pool + dest * s_slot + tile, state.to(Pool.dtype.element_ty))
    tl.store(Pool + slot * s_slot + tile, state.to(Pool.dtype.element_ty))


@triton.jit
def _bmm_chunk_kernel(Cm, Bm, sx, CB, CuChunks, N: tl.constexpr, G: tl.constexpr, CS: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid, cg = tl.program_id(0), tl.program_id(1).to(tl.int64)
    c, g = cg // G, cg % G
    pid_m, pid_n = pid // tl.cdiv(CS, BLOCK_N), pid % tl.cdiv(CS, BLOCK_N)
    if pid_n * BLOCK_N >= (pid_m + 1) * BLOCK_M:
        return  # strictly above the diagonal: the scan never reads it
    t0 = tl.load(CuChunks + c)
    n = tl.load(CuChunks + c + 1) - t0
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    a_ptrs = Cm + (t0 + om).to(tl.int64)[:, None] * sx + g * N + ok[None, :]
    b_ptrs = Bm + (t0 + on).to(tl.int64)[None, :] * sx + g * N + ok[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_K):
        a = tl.load(a_ptrs, (om < n)[:, None], 0.0)
        b = tl.load(b_ptrs, (on < n)[None, :], 0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    tl.store(CB + ((c * G + g) * CS + om[:, None]) * CS + on[None, :], acc)


@triton.jit
def _chunk_scan_kernel(CB, X, Cm, sx, Dt, DACs, Entering, Dskip, Y, sy, CuChunks, nchunks,
                       H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, G: tl.constexpr, CS: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid, c, h = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    pid_m, pid_n = pid // tl.cdiv(P, BLOCK_N), pid % tl.cdiv(P, BLOCK_N)
    g = h // (H // G)
    t0 = tl.load(CuChunks + c).to(tl.int64)
    n = tl.load(CuChunks + c + 1) - t0
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if pid_m * BLOCK_M >= n:
        return
    hc = h.to(tl.int64) * nchunks * CS + c * CS
    da_m = tl.load(DACs + hc + om)
    # Inter-chunk term: C_t . (state entering the chunk) * exp(dA_cs[t])
    od = tl.arange(0, N)
    cmat = tl.load(Cm + (t0 + om)[:, None] * sx + g * N + od[None, :], (om < n)[:, None], 0.0)
    prev = tl.load(Entering + ((c * H + h) * P + on[None, :]) * N + od[:, None])
    acc = tl.dot(cmat, prev) * tl.exp(da_m)[:, None]
    # Intra-chunk term: sum_{s <= t} CB[t, s] exp(dA_cs[t] - dA_cs[s]) dt_s x_s
    ok = tl.arange(0, BLOCK_K)
    cb_ptrs = CB + ((c * G + g) * CS + om[:, None]) * CS + ok[None, :]
    x_ptrs = X + (t0 + ok)[:, None] * sx + h * P + on[None, :]
    k_max = tl.minimum((pid_m + 1) * BLOCK_M, n)
    for k in range(0, k_max, BLOCK_K):
        cb = tl.load(cb_ptrs)
        da_k = tl.load(DACs + hc + k + ok)
        cb *= tl.exp(tl.minimum(da_m[:, None] - da_k[None, :], 0.0)) * tl.load(Dt + hc + k + ok)[None, :]
        cb = tl.where(om[:, None] >= (k + ok)[None, :], cb, 0.0)
        x = tl.load(x_ptrs, (ok < n - k)[:, None], 0.0)
        acc += tl.dot(cb.to(X.dtype.element_ty), x)
        cb_ptrs += BLOCK_K
        x_ptrs += BLOCK_K * sx
    x_m = tl.load(X + (t0 + om)[:, None] * sx + h * P + on[None, :], (om < n)[:, None], 0.0).to(tl.float32)
    acc += x_m * tl.load(Dskip + h)
    tl.store(Y + (t0 + om)[:, None] * sy + h * P + on[None, :], acc.to(Y.dtype.element_ty), (om < n)[:, None])


def prefill(zxbcdt: torch.Tensor, params: Mamba2Params, conv_state: torch.Tensor, ssm_state: torch.Tensor,
            plan: Mamba2PrefillPlan, *, layer: int = 0,
            window: tuple[WindowLayout, dict] | None = None) -> torch.Tensor:
    """Packed varlen prefill: returns y [T, d_inner] bf16 (before the gated norm).

    Each sequence continues from its slot's conv and SSM state when it has a prefix (zero otherwise); its final
    states replace them, and each checkpoint slot receives the conv window and SSM state at its boundary. An FP16
    state rounds at every chunk end (`layer` as in `decode`; the plan carries the positions).
    With window = (layout, this layer's region views) the SSM states are quantized: the recurrence runs on FP32
    copies of the sequences' states (`plan.seq_rows`, checkpoints in `plan.chunk_scratch`), and the final and
    checkpoint states are quantized back as new boundary states."""
    sr = window is None and _check_state(ssm_state, params, plan.chunk_last)
    _, xbc, dt = _split(zxbcdt, params)
    t, c = xbc.shape
    if t != plan.tokens:
        raise ValueError("the prefill plan covers a different token count")
    dev = xbc.device
    H, P, N, G = params.heads, params.head_dim, params.state, params.groups
    conv = torch.empty((t, c), dtype=torch.bfloat16, device=dev)
    nseq = len(plan.seq_lens)
    if plan.num_checkpoints:  # reads the inputs only; checkpoint slots are not any sequence's own
        _conv_checkpoint_kernel[(plan.num_checkpoints, triton.cdiv(c, 1024))](
            xbc, xbc.stride(0), conv_state, *conv_state.stride(), plan.checkpoint_ends, plan.checkpoint_slots, c, 1024)
    block_t, block_c = 8, 256  # larger token tiles spill (44 GB/s at 32 x 256 against 210 here)
    _conv_prefill_kernel[(nseq, triton.cdiv(max(plan.seq_lens), block_t), triton.cdiv(c, block_c))](
        xbc, xbc.stride(0), params.conv_weight, params.conv_bias, conv_state, *conv_state.stride(), plan.cu_seqlens,
        plan.slots, plan.has_initial_state, conv, conv.stride(0), c, block_t, block_c, num_warps=4)
    nc = plan.num_chunks
    dt_out = torch.empty((H, nc, CHUNK), dtype=torch.float32, device=dev)
    da_cs = torch.empty_like(dt_out)
    _chunk_cumsum_kernel[(nc, triton.cdiv(H, 16))](
        dt, dt.stride(0), params.dt_bias, params.A, dt_out, da_cs, plan.cu_chunks, nc, H, CHUNK, 16,
        num_warps=4)
    x, bm, cm = conv, conv[:, H * P:], conv[:, H * P + G * N:]
    entering = torch.empty((nc, H, P, N), dtype=torch.bfloat16, device=dev)
    if window is not None:
        layout, views = window
        states = torch.empty((plan.scratch_rows, H, P, N), dtype=torch.float32, device=dev)
        materialize(layout, views, plan.slots, states[:nseq])
        state_slots, chunk_dest = plan.seq_rows, plan.chunk_scratch
    else:
        states, state_slots, chunk_dest = ssm_state, plan.slots, plan.chunk_dest
    bm_, bn_ = _STATES_TILE
    _chunk_states_kernel[(triton.cdiv(P, bm_) * triton.cdiv(N, bn_), nseq, H)](
        x, bm, conv.stride(0), dt_out, da_cs, entering, plan.cu_chunks, plan.seq_chunks, state_slots,
        plan.has_initial_state, chunk_dest, plan.chunk_last, states, states.stride(0), nc, layer,
        H, P, N, G, CHUNK, bm_, bn_, 32, _STATES_PRECISION, sr, _PHILOX_ROUNDS, num_warps=4, num_stages=2)
    if window is not None:
        quantize(layout, views, states[:nseq], plan.slots)
        if plan.num_checkpoints:
            quantize(layout, views, states[plan.checkpoint_rows.long()], plan.checkpoint_slots)
    cb = torch.empty((nc, G, CHUNK, CHUNK), dtype=torch.float32, device=dev)
    _bmm_chunk_kernel[(triton.cdiv(CHUNK, 64) ** 2, nc * G)](
        cm, bm, conv.stride(0), cb, plan.cu_chunks, N, G, CHUNK, 64, 64, 64, num_warps=4, num_stages=2)
    y = torch.empty((t, params.d_inner), dtype=torch.bfloat16, device=dev)
    _chunk_scan_kernel[(triton.cdiv(CHUNK, 32) * triton.cdiv(P, 64), nc, H)](
        cb, x, cm, conv.stride(0), dt_out, da_cs, entering, params.D, y, y.stride(0), plan.cu_chunks, nc,
        H, P, N, G, CHUNK, 32, 64, 32, num_warps=4, num_stages=2)
    return y
