# SPDX-License-Identifier: Apache-2.0
"""Nemotron-3.5-Super's latent MoE block: 512 routed NVFP4 experts (top 22, sigmoid routing) that work in a 1,024-wide
latent space, plus an NVFP4 shared expert on the full 4,096 hidden. The checkpoint's definition (transformers'
NemotronHMoE, which the snapshot's omni model wraps) for post-norm rows x:

    s = sigmoid(x W_gate^T)            FP32 router (the logits are taken in FP32)
    ids = top22(s + e_score_correction_bias)  the bias steers the choice only (n_group = topk_group = 1)
    w = 5 * s[ids] / (sum s[ids] + 1e-20)    weights from the plain sigmoid, renormalized, routed_scaling_factor 5
    out = fc2_latent(sum_k w_k down_k(relu(up_k(fc1_latent(x)))^2)) + shared_down(relu(shared_up(x))^2)

so the weighting happens in latent space before fc2, and the shared expert sees x, not the latent.

Formats and fusions:
- Router and fc1_latent both read x: one BF16 GEMM over their concatenated rows [512 + 1024, 4096] with an FP32 output.
  FP32 logits matter: a block's 22nd and 23rd choice scores sit a median 7e-4 to 1.6e-3 apart, and BF16 logits change
  the top-22 set of 9-22% of rows. The routing kernel reads the logits and also casts the latent half to BF16 for the
  experts, so the cast costs no launch.
- Routed experts (NVFP4), by call size: Sharar's kernels (below) up to SMALL_ROWS rows, beyond them FlashInfer's b12x
  SM12x MoE (`kernels/b12x_moe.py`, the non-gated ReLU² activation) on the same tensors: fc1 is up_proj alone
  ([E, 2688, 512] packed), block scales swizzled per expert (Flash-Next's SM120 interleave). Every expert of a block
  carries the same input scales, so the one per-layer activation scale (Flash-Next's convention, the largest
  expert's) is exactly the checkpoint's. routed_scaling_factor rides in the fc2 dequantization scale, so the routing
  weights are plain renormalized ones (route_budget's pick kernel writes those). b12x is not deterministic: calls past
  16 rows (prefill chunks, wide verify forwards) differ run to run (a prompt's teacher-forced logprobs by up to 1.5).
  The 3-bit experts serve every size deterministically.
- Shared expert: the checkpoint's NVFP4 through Sharar's NVFP4 linear (`ops.nvfp4_linear`), ReLU² in place between.
- fc2_latent: BF16, as shipped, as an addmm onto the shared expert's output (the block's sum in the GEMM epilogue).
- 3-bit routed experts (`w3=`, with SHARAR_W3_DECODE when the W3 checkpoint carries this block's experts): the only
  expert copy (the NVFP4 experts are not loaded), every call, decode to prefill, through `kernels/w3_moe.py`'s ReLU^2
  geometry (E4M3 activations per 32-group, GPTQ-calibrated codes, `weights/nemotron_w3_experts.py`).
  routed_scaling_factor is folded into fc2's row scales at load; padded and unfilled slots name no expert (id 512),
  which the kernel skips. Calls of at most SMALL_ROWS rows keep the small-call kernels below for everything but the
  routed sum (`nemotron_moe.w3_block`: router, shared up, w3_moe on the router's BF16 latent and ids, then fc2_latent
  plus the shared down).

Calls of at most SMALL_ROWS rows (decode, verify, the MTP head's draft steps) run the block through Sharar's own kernels
(`kernels/nemotron_moe.py`) instead, on the same tensors: the router|fc1 GEMV, the routing kernel below, then four
launches (prep: the call's experts and pairs, the latent and x quantized; up: routed and shared, relu² and the NVFP4
quantization in the epilogue; down: routed, summed per row; out: fc2_latent plus the shared expert's down), each
streaming its weights once, every distinct expert once for all of its pairs, with programmatic dependent launches so
a kernel's weight stream starts under the previous one's tail.

Routing for decode and verify forwards follows Flash-Next (`model/flashnext/moe.py`): a CUDA graph's padded rows (dummy
state slot 0) take row 0's top expert at weight 0, which reads no further expert (3-bit experts: no expert, id 512).
MoE sharing (off by default) with SHARAR_MOE_KEEP=F: every real token keeps its own top F experts by choice score (its
floor, which no cap cuts); its other top-22 experts ride along when they are in the call's set S, the union of the real
tokens' floors (free riders: no extra bytes); SHARAR_MOE_BUDGET=[P:]CAP grows S past that union, when the cap is larger,
by the experts with the most routing weight over the call's real tokens' top 22 (more free riders), and never shrinks
it; a cap the floors' union exceeds guarantees each token only its top `fallback` and fills to the cap. Each token's
weights renormalize over its kept experts (`route_budget`). Every path is capture-safe: static shapes, no host syncs.
`ctx=None` (the MTP head's draft steps) routes every row plainly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sharar.kernels.flashnext_fused import _or
from triton.language.extra import libdevice

from sharar.model.nemotron import config as c

@triton.jit
def _sigmoid_exact(x):
    # torch's FP32 sigmoid, 1 / (1 + exp(-x)) with the accurate exp and a rounded division: the choice scores are the
    # reference's bits (exact FP32 ties of sigmoid + bias do occur; the kernel takes the lower expert)
    return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _cast_latent(L, Y, t, stride_l, E: tl.constexpr, LAT: tl.constexpr):
    """Row t's latent half of the fused router|fc1 GEMM output (FP32 columns E..E+LAT-1) -> BF16 Y [t, :]."""
    cols = tl.arange(0, LAT)
    v = tl.load(L + t * stride_l + E + cols)
    tl.store(Y + t * LAT + cols, v.to(tl.bfloat16))


@triton.jit
def _choice_keys(L, Bias, t, stride_l, E: tl.constexpr, BLOCK: tl.constexpr):
    """Row t's choice scores sigmoid(logit) + bias as distinct int64 keys: the FP32 bits made order-preserving above
    BLOCK - 1 - expert, so one integer max finds the largest score and, among equal scores, the lower expert."""
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    choice = _sigmoid_exact(tl.load(L + t * stride_l + cols, mask=mask, other=0.0)) + tl.load(Bias + cols, mask=mask, other=0.0)
    bits = choice.to(tl.int32, bitcast=True)
    ordered = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.int64)
    return tl.where(mask, (ordered << 32) | (BLOCK - 1 - cols).to(tl.int64), _NO_KEY(BLOCK))


@triton.jit
def _NO_KEY(BLOCK: tl.constexpr):
    return tl.full([BLOCK], -2147483648, tl.int64) << 32  # below every key


@triton.jit
def _route_kernel(L, Bias, Slots, Ids, W, Y, stride_l, E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                  BLOCK: tl.constexpr, LAT: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr,
                  REPEAT: tl.constexpr = True):
    t = tl.program_id(0)
    _cast_latent(L, Y, t, stride_l, E, LAT)
    cols = tl.arange(0, BLOCK)
    key = _choice_keys(L, Bias, t, stride_l, E, BLOCK)
    j = tl.arange(0, KP)
    ids = tl.zeros([KP], dtype=tl.int32)
    for k in tl.static_range(K):  # largest choice score first: one warp-wide integer max a rank
        e = BLOCK - 1 - (tl.max(key, 0) & 0xFFFFFFFF).to(tl.int32)
        ids = tl.where(j == k, e, ids)
        key = tl.where(cols == e, _NO_KEY(BLOCK), key)
    # the chosen experts' plain sigmoid weights, from their logits again (the row is in L1)
    s = tl.where(j < K, _sigmoid_exact(tl.load(L + t * stride_l + ids, mask=j < K, other=0.0)), 0.0)
    wts = tl.math.div_rn(s, tl.zeros([KP], dtype=tl.float32) + (tl.sum(s, 0) + 1e-20))
    if HAS_SLOTS:
        if tl.load(Slots + t // STEPS) == 0:
            # a padded row (a graph's dummy slot 0) names row 0's top expert at weight 0: an expert the call reads anyway
            # (FlashInfer's NVFP4 MoE needs a real expert in every slot); row 0 is real, padding follows the requests
            if REPEAT:
                top0 = BLOCK - 1 - (tl.max(_choice_keys(L, Bias, 0, stride_l, E, BLOCK), 0) & 0xFFFFFFFF).to(tl.int32)
                ids = tl.zeros([KP], dtype=tl.int32) + top0
            else:  # no expert (the 3-bit kernel skips id E)
                ids = tl.zeros([KP], dtype=tl.int32) + E
            wts = tl.zeros([KP], dtype=tl.float32)
    tl.store(Ids + t * K + j, ids, mask=j < K)
    tl.store(W + t * K + j, wts, mask=j < K)


def route_topk(fused: torch.Tensor, bias: torch.Tensor, k: int, latent: int, slots: torch.Tensor | None = None,
               steps: int = 1, repeat: bool = True):
    """fused: FP32 [M, E + latent] (router logits | fc1 latent) -> (int32 ids [M, k] by descending choice score, FP32
    renormalized sigmoid weights [M, k] (without routed_scaling_factor), BF16 latent [M, latent]). With `slots`, row r
    is real iff slots[r // steps] != 0; a padded row takes row 0's top expert at weight 0 (without `repeat`: id E, no
    expert, for the 3-bit experts)."""
    m = fused.shape[0]
    e = fused.shape[1] - latent
    dev = fused.device
    ids = torch.empty((m, k), dtype=torch.int32, device=dev)
    w = torch.empty((m, k), dtype=torch.float32, device=dev)
    lat = torch.empty((m, latent), dtype=torch.bfloat16, device=dev)
    _route_kernel[(m,)](fused, bias, ids if slots is None else slots, ids, w, lat, fused.stride(0), e, k,
                        KP=triton.next_power_of_2(k), BLOCK=triton.next_power_of_2(e), LAT=latent, STEPS=steps,
                        HAS_SLOTS=slots is not None, REPEAT=repeat, num_warps=1)  # one warp a row: shuffle-only reductions
    return ids, w, lat


# ---- budgeted expert selection ('MoE sharing', SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET): Flash-Next's route_budget with this
# model's ranking. The rank kernel is this model's (sigmoid + bias choice, plain-sigmoid weights); the score, select and
# pick kernels are Flash-Next's, which take any ranking: a call's set is every real row's top `keep` plus the experts
# with the most routing weight over its real rows' top k, up to the cap; each row then takes its top k in the set.


@triton.jit
def _route_rank_kernel(L, Bias, Slots, Cand, P, Bits, Dense, Y, stride_l, Bits2, E: tl.constexpr, BLOCK: tl.constexpr,
                       KEEP: tl.constexpr, K: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, LAT: tl.constexpr,
                       STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr, G: tl.constexpr = 0):
    t = tl.program_id(0)
    _cast_latent(L, Y, t, stride_l, E, LAT)
    cols = tl.arange(0, BLOCK)
    key = _choice_keys(L, Bias, t, stride_l, E, BLOCK)
    j = tl.arange(0, CP)
    ids = tl.zeros([CP], dtype=tl.int32)
    w = tl.arange(0, BLOCK // 32)
    bits = tl.zeros([BLOCK // 32], dtype=tl.int32)
    bits2 = tl.zeros([BLOCK // 32], dtype=tl.int32)  # the row's top G: the guarantee when the floors overflow the cap
    top = cols < 0  # the row's top K, by expert
    for r in tl.static_range(C):
        e = BLOCK - 1 - (tl.max(key, 0) & 0xFFFFFFFF).to(tl.int32)
        ids = tl.where(j == r, e, ids)
        if r < K:
            top = top | (cols == e)
        if r < KEEP:
            bits = tl.where(w == e // 32, bits | (1 << (e % 32)), bits)
        if r < G:
            bits2 = tl.where(w == e // 32, bits2 | (1 << (e % 32)), bits2)
        key = tl.where(cols == e, _NO_KEY(BLOCK), key)
    p = tl.where(j < C, _sigmoid_exact(tl.load(L + t * stride_l + ids, mask=j < C, other=0.0)), 0.0)  # 0 past rank C
    tl.store(Cand + t * CP + j, ids)
    tl.store(P + t * CP + j, p)
    real = True
    if HAS_SLOTS:  # a padded row adds nothing to the guaranteed set or the scores
        real = tl.load(Slots + t // STEPS) != 0
        bits = tl.where(real, bits, 0)
        bits2 = tl.where(real, bits2, 0)
    tl.store(Bits + t * (BLOCK // 32) + w, bits)
    if G > 0:
        tl.store(Bits2 + t * (BLOCK // 32) + w, bits2)
    # the row's routing weight within its top K by expert, for the call's expert scores
    dense = tl.where(top, _sigmoid_exact(tl.load(L + t * stride_l + cols, mask=cols < E, other=0.0)), 0.0)
    tl.store(Dense + t * BLOCK + cols, tl.where(real, dense / tl.sum(dense, 0), 0.0))


@triton.jit
def _share_select_kernel(Score, Bits, Slots, Set, R, Budget, PerRequest, Bits2, E: tl.constexpr,
                         BLOCK: tl.constexpr, NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr,
                         HAS_SLOTS: tl.constexpr, G: tl.constexpr = 0):
    """Flash-Next's `_budget_select_kernel`: word `program_id` of the call's set, every real row's top KEEP (the OR of
    Bits), then the highest-scoring other experts (equal scores: the lower expert), as many as the cap leaves (Budget,
    or min(Budget, PerRequest x real requests); 0: none). With G, a cap the floors' union exceeds
    guarantees each real row only its top G (Bits2), and the cap fills from the scores. One program per 32 experts,
    each ranking its own against all."""
    g = tl.program_id(0)
    reqs = R // STEPS
    if HAS_SLOTS:  # a graph's padded requests (dummy slot 0) do not count
        reqs = 0
        rq = tl.arange(0, RB)
        for q0 in range(0, R // STEPS, RB):
            q = q0 + rq
            reqs += tl.sum((tl.load(Slots + q, mask=q < R // STEPS, other=0) != 0).to(tl.int32), 0)
    if PerRequest > 0:
        Budget = tl.minimum(Budget, PerRequest * reqs)
    w = tl.arange(0, NW)
    rr = tl.arange(0, RB)
    core = tl.zeros([NW], dtype=tl.int32)
    for r0 in range(0, R, RB):
        b = tl.load(Bits + (r0 + rr)[:, None] * NW + w[None, :], mask=(r0 + rr)[:, None] < R, other=0)
        core = core | tl.reduce(b, 0, _or)
    cols = tl.arange(0, BLOCK)
    if G > 0:  # the floors overflow the cap: each row's top G only
        incore = ((tl.gather(core, cols // 32, 0) >> (cols % 32)) & 1) != 0
        if (Budget > 0) & (tl.sum(incore.to(tl.int32), 0) > Budget):
            core = tl.zeros([NW], dtype=tl.int32)
            for r0 in range(0, R, RB):
                b = tl.load(Bits2 + (r0 + rr)[:, None] * NW + w[None, :], mask=(r0 + rr)[:, None] < R, other=0)
                core = core | tl.reduce(b, 0, _or)
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


@triton.jit
def _share_pick_kernel(Slots, Cand, P, Set, Ids, W, R, K: tl.constexpr, C: tl.constexpr, CP: tl.constexpr,
                       NW: tl.constexpr, RB: tl.constexpr, STEPS: tl.constexpr, HAS_SLOTS: tl.constexpr,
                       KP: tl.constexpr, NONE: tl.constexpr, REPEAT: tl.constexpr):
    """Flash-Next's `_route_pick_kernel` with route_topk's weight arithmetic (the warp's butterfly sum, a rounded
    division by it plus 1e-20): a row that keeps all its top k gets route_topk's weights bit for bit, so sharing
    changes a forward only where it drops an expert."""
    rr = tl.arange(0, RB)
    core = tl.load(Set + tl.arange(0, NW))
    rows = tl.program_id(0) * RB + rr
    live = rows < R
    j = tl.arange(0, CP)[None, :]
    cand = tl.load(Cand + rows[:, None] * CP + j, mask=live[:, None], other=0)
    p = tl.load(P + rows[:, None] * CP + j, mask=live[:, None], other=1.0)
    word = tl.reshape(tl.gather(core, tl.reshape(cand // 32, [RB * CP]), 0), [RB, CP])
    pull = (j < C) & (((word >> (cand % 32)) & 1) != 0)
    real = tl.full([RB], 1, tl.int1)
    if HAS_SLOTS:
        real = tl.load(Slots + rows // STEPS, mask=live, other=0) != 0
    a = pull.to(tl.int32)
    slot = tl.cumsum(a, 1) - 1
    chosen = pull & (slot < K) & real[:, None]
    used = tl.where(real, tl.minimum(tl.sum(a, 1), K), 0)
    # the row's sum folded as a warp's butterfly sums 32 lanes (route_topk's tl.sum over one warp): halves added pairwise
    v = tl.where(chosen, p, 0.0)
    if CP == 64:
        v = tl.sum(tl.reshape(v, [RB, 2, 32]), 1)
    v16 = tl.sum(tl.reshape(v, [RB, 2, 16]), 1)
    v8 = tl.sum(tl.reshape(v16, [RB, 2, 8]), 1)
    v4 = tl.sum(tl.reshape(v8, [RB, 2, 4]), 1)
    v2 = tl.sum(tl.reshape(v4, [RB, 2, 2]), 1)
    total = tl.sum(v2, 1) + 1e-20
    smask = live[:, None] & chosen
    tl.store(Ids + rows[:, None] * K + slot, cand, mask=smask)
    tl.store(W + rows[:, None] * K + slot, tl.math.div_rn(p, tl.zeros([RB, CP], dtype=tl.float32) + total[:, None]), mask=smask)
    q = tl.arange(0, KP)[None, :]
    hole = live[:, None] & (q >= used[:, None]) & (q < K)
    if REPEAT:
        first = tl.where(real, tl.load(Cand + rows * CP, mask=live, other=0), tl.load(Cand))
        fill_id = tl.broadcast_to(first[:, None], [RB, KP])
    else:
        fill_id = tl.full([RB, KP], NONE, tl.int32)
    tl.store(Ids + rows[:, None] * K + q, fill_id, mask=hole)
    tl.store(W + rows[:, None] * K + q, tl.zeros([RB, KP], dtype=tl.float32), mask=hole)


def route_budget(fused: torch.Tensor, bias: torch.Tensor, k: int, latent: int, keep: int, budget: int,
                 slots: torch.Tensor | None = None, steps: int = 1, per_request: int = 0, repeat: bool = True,
                 fallback: int = 0):
    """Budgeted top-k over fused FP32 [R, E + latent] -> (ids [R, k], renormalized weights [R, k], BF16 latent): the
    call reads at most min(budget, per_request x real requests) distinct experts (or `budget` without
    `per_request`), more only when the rows' guaranteed top `keep` exceed it (a budget of 0: the guaranteed experts
    alone). Each real row takes those of its top k (by choice score) that are in the set, in rank order, its top
    `keep` first (the others ride along free); its weights are its plain sigmoid scores renormalized over those. Slots
    left over (all of a padded row's) name the row's top expert (a padded row: row 0's) at weight 0, or without
    `repeat` no expert (id E). `fallback` (< keep): when the guaranteed experts' union exceeds the cap, each real row is
    guaranteed only its top `fallback` and the cap fills by routing weight (rows keep whichever of their top k the set
    holds). Five launches, no atomics, no host sync."""
    from sharar.kernels.flashnext_fused import _budget_score_kernel
    r = fused.shape[0]
    e = fused.shape[1] - latent
    block, cp = triton.next_power_of_2(e), triton.next_power_of_2(k)
    nw = block // 32
    dev = fused.device
    cand = torch.empty((r, cp), dtype=torch.int32, device=dev)
    p = torch.empty((r, cp), dtype=torch.float32, device=dev)
    bits = torch.empty((r, nw), dtype=torch.int32, device=dev)
    bits2 = torch.empty((r, nw), dtype=torch.int32, device=dev) if fallback else bits
    dense = torch.empty((r, block), dtype=torch.float32, device=dev)
    score = torch.empty((block,), dtype=torch.float32, device=dev)
    chosen = torch.empty((nw,), dtype=torch.int32, device=dev)
    ids = torch.empty((r, k), dtype=torch.int32, device=dev)
    w = torch.empty((r, k), dtype=torch.float32, device=dev)
    lat = torch.empty((r, latent), dtype=torch.bfloat16, device=dev)
    sl = cand if slots is None else slots
    common = dict(C=k, CP=cp, STEPS=steps, HAS_SLOTS=slots is not None)
    _route_rank_kernel[(r,)](fused, bias, sl, cand, p, bits, dense, lat, fused.stride(0), bits2, e, BLOCK=block,
                             KEEP=keep, K=k, LAT=latent, G=fallback, num_warps=1, **common)
    _budget_score_kernel[(block // 32,)](dense, score, r, BLOCK=block, EB=32, RB=16, num_warps=1)
    rb = min(64, max(16, triton.next_power_of_2(r)))
    _share_select_kernel[(nw,)](score, bits, sl, chosen, r, budget, per_request, bits2, E=e, BLOCK=block,
                                NW=nw, RB=rb, STEPS=steps, HAS_SLOTS=slots is not None, G=fallback, num_warps=4)
    _share_pick_kernel[(triton.cdiv(r, rb),)](sl, cand, p, chosen, ids, w, r, k, NW=nw, RB=rb, num_warps=4,
                                              KP=triton.next_power_of_2(k), NONE=e, REPEAT=repeat, **common)
    return ids, w, lat


@triton.jit
def _relu2_kernel(X, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    v = tl.maximum(tl.load(X + t * stride_x + cols, mask=mask, other=0.0).to(tl.float32), 0.0)
    tl.store(X + t * stride_x + cols, (v * v).to(tl.bfloat16), mask=mask)


def relu2_(x: torch.Tensor) -> torch.Tensor:
    """relu(x)^2 in place for BF16 [M, N] (FP32 square, one rounding: torch's relu(x) ** 2)."""
    m, n = x.shape
    blk = 2048
    _relu2_kernel[(m, triton.cdiv(n, blk))](x, x.stride(0), n, BLOCK=blk)
    return x


def expert_names(prefix: str, experts: int) -> list[str]:
    """Every tensor `load_experts` reads for the block at `prefix`."""
    kinds = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    return [f"{prefix}.experts.{e}.{p}.{k}" for e in range(experts) for p in ("up_proj", "down_proj") for k in kinds]


def load_experts(w, prefix: str, experts: int, device) -> dict[str, torch.Tensor]:
    """A block's routed NVFP4 experts as stored: up [E, I, L/2] and down [E, L, I/2] uint8 (low nibble first), their
    float8_e4m3fn block scales (unswizzled), FP32 [E] global scales and input scales. Stacked on the host from views of
    the block's shard as read (`SnapshotWeights.get_whole` on the CPU), then copied to the device: only the experts'
    bytes reach it."""
    t = w.get_whole(expert_names(prefix, experts), "cpu")

    def each(proj: str, kind: str) -> torch.Tensor:
        return torch.stack([t[f"{prefix}.experts.{e}.{proj}.{kind}"] for e in range(experts)]).to(device)

    def scalars(proj: str, kind: str) -> torch.Tensor:  # [] in the trunk's blocks, [1] in the MTP head's
        return each(proj, kind).float().reshape(experts)

    return {"w1": each("up_proj", "weight"), "s1": each("up_proj", "weight_scale"),
            "g1": scalars("up_proj", "weight_scale_2"), "in1": scalars("up_proj", "input_scale"),
            "w2": each("down_proj", "weight"), "s2": each("down_proj", "weight_scale"),
            "g2": scalars("down_proj", "weight_scale_2"), "in2": scalars("down_proj", "input_scale")}


class NemotronMoe:
    SHARE_MIN = 8  # budgeted selection: under a binding cap each token keeps at least its top SHARE_MIN
    # calls of at most this many rows take Sharar's kernels, the fastest at every M of 1-16 (block 1, 16 rows: 3481 us
    # against b12x's 3591; `kernels/csrc/nemotron_moe.cu`)
    SMALL_ROWS = 16

    def __init__(self, w, prefix: str, device: torch.device, target, w3=None):
        """`prefix`: the block's mixer ("language_model.backbone.layers.1.mixer", or the MTP head's
        "language_model.mtp.layers.1.mixer"); `target`: geometry (hidden_size, num_experts, experts_per_token,
        moe_intermediate_size, shared_expert_intermediate_size), the latent width and routed scaling from config.
        `w3`: this block's 3-bit experts (kernels/w3_moe.W3Experts from the W3 checkpoint); the NVFP4 experts are then
        not read."""
        from sharar.engine.config import current
        from sharar.model.flashnext.moe import swizzle_block_scales_batched
        from sharar.weights.modelopt import load_nvfp4_linear

        self.device = device
        self.hidden, self.experts, self.topk = target.hidden_size, target.num_experts, target.experts_per_token
        self.inter, self.latent = target.moe_intermediate_size, c.MOE_LATENT_SIZE
        router = w.get(f"{prefix}.gate.weight").to(device)  # [E, H] bf16
        fc1 = w.get(f"{prefix}.fc1_latent_proj.weight").to(device)  # [L, H] bf16
        self.gate_fc1 = torch.cat([router, fc1]).contiguous()  # [E + L, H]: one GEMM for both readers of x
        self.bias = w.get(f"{prefix}.gate.e_score_correction_bias").to(device).float().contiguous()  # [E]
        self.fc2 = w.get(f"{prefix}.fc2_latent_proj.weight").to(device).contiguous()  # [H, L] bf16
        self.shared_up = load_nvfp4_linear(w, [f"{prefix}.shared_experts.up_proj"], device)
        self.shared_down = load_nvfp4_linear(w, [f"{prefix}.shared_experts.down_proj"], device)

        cfg = current()
        self.keep = cfg.moe_keep if cfg.moe_keep < self.topk else 0  # each token's floor (0: every token's top 22)
        self.budget = cfg.moe_budget if self.keep else 0
        self.per_request = cfg.moe_budget_per_request

        self.w3 = w3
        if w3 is not None:
            import dataclasses
            if (w3.base, w3.experts) != (0, self.experts) or w3.row13.shape[1] != self.inter or w3.row2.shape[1] != self.latent:
                raise ValueError(f"{prefix}: 3-bit experts of the wrong geometry")
            # routed_scaling_factor in fc2's row scales (a scaled copy: the loaded experts stay as stored), as NVFP4's
            # alpha carries it
            self.w3 = dataclasses.replace(w3, row2=w3.row2 * c.ROUTED_SCALING_FACTOR)
            # calls of at most SMALL_ROWS rows keep the small-call kernels around w3_moe (`small`): no NVFP4 experts
            none = torch.empty(0, dtype=torch.uint8, device=device)
            self.w1 = self.s1 = self.w2 = self.s2 = none
            self.b12x = None
            one = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device).reshape(1).contiguous()  # noqa: E731
            su, sd = self.shared_up, self.shared_down
            self.small_scales = [one(0.0), one(0.0), one(0.0), one(0.0), one(su.alpha), one(sd.alpha),
                                 one(su.input_scale_inv), one(sd.input_scale_inv)]
            self._small = self._small_weights()
            return
        nv = load_experts(w, prefix, self.experts, device)
        self.w1, self.w2 = nv["w1"], nv["w2"]  # [E, I, L/2], [E, L, I/2] uint8
        self.s1, self.s2 = swizzle_block_scales_batched(nv["s1"]), swizzle_block_scales_batched(nv["s2"])
        in1, in2 = nv["in1"].max(), nv["in2"].max()  # one activation scale per layer (all experts share it here)
        self.g1, self.g2 = nv["g1"], nv["g2"]  # the checkpoint's global weight scales, for `reference`
        self.in1, self.in2 = in1, in2
        # quant_scales: fc1 input quantization (1 / scale), block scales, dequantization alphas per expert; the
        # routed scaling factor joins fc2's alpha
        self.quant_scales = [1.0 / in1, self.s1, in1 * nv["g1"], 1.0 / in2, self.s2,
                             in2 * nv["g2"] * c.ROUTED_SCALING_FACTOR]
        one = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device).reshape(1).contiguous()  # noqa: E731
        su, sd = self.shared_up, self.shared_down
        # the small-call kernels' scales: routed alphas [E], activation global scales, the shared expert's
        self.small_scales = [self.quant_scales[2].float().contiguous(), self.quant_scales[5].float().contiguous(),
                             one(self.quant_scales[0]), one(self.quant_scales[3]), one(su.alpha), one(sd.alpha),
                             one(su.input_scale_inv), one(sd.input_scale_inv)]
        # calls past SMALL_ROWS rows: FlashInfer's b12x MoE on the same tensors
        from sharar.kernels.b12x_moe import B12xExperts
        self.b12x = B12xExperts(self.w1, self.s1, nv["g1"], in1, self.w2, self.s2, nv["g2"], in2, c.ROUTED_SCALING_FACTOR)
        self._small = self._small_weights()

    def small_weights(self) -> list[torch.Tensor]:
        """The tensors `kernels/nemotron_moe.py` reads, in its order."""
        return self._small

    def _small_weights(self) -> list[torch.Tensor]:
        from sharar.kernels import nemotron_moe
        su, sd = self.shared_up, self.shared_down
        return [self.gate_fc1, self.fc2, self.w1, self.s1, self.w2, self.s2, su.weight, su.weight_scale_swizzled,
                sd.weight, sd.weight_scale_swizzled, *self.small_scales, nemotron_moe.counters(self.device)]

    @property
    def expert_bytes(self) -> int:
        """Resident bytes of all routed experts (packed weights and their block scales)."""
        if self.w3 is not None:
            return self.w3.bytes
        return self.w1.numel() + self.w2.numel() + (self.s1.numel() + self.s2.numel()) * 4

    @property
    def dense_bytes(self) -> int:
        """Bytes every call reads whatever its routing: router, latent projections, shared expert."""
        return (self.gate_fc1.numel() + self.fc2.numel()) * 2 + self.bias.numel() * 4 + self.shared_up.bytes + self.shared_down.bytes

    @property
    def bytes(self) -> int:
        return self.expert_bytes + self.dense_bytes

    @property
    def bytes_per_expert(self) -> int:
        return self.expert_bytes // self.experts

    def route(self, fused: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FP32 router|fc1 rows -> (ids, renormalized weights, BF16 latent)."""
        if ctx is None or ctx.mode not in ("decode", "verify"):
            return route_topk(fused, self.bias, self.topk, self.latent)
        # A graph pads with requests on the dummy state slot 0. A verify forward's rows are request-major, verify_steps
        # per request, and every request drafts the call's depth, so every row of a real request is real.
        slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
        repeat = self.w3 is None  # the 3-bit kernel skips id E: a padded row reads no expert
        if not self.keep:
            return route_topk(fused, self.bias, self.topk, self.latent, slots, steps, repeat=repeat)
        # each real token's top `keep` (its floor); its other top-k experts in the call's set ride along
        return route_budget(fused, self.bias, self.topk, self.latent, self.keep, self.budget, slots, steps,
                            per_request=self.per_request, repeat=repeat,
                            fallback=self.fallback(fused.shape[0], steps))

    def fallback(self, rows: int, steps: int) -> int:
        """The guarantee when the floors' union exceeds SHARAR_MOE_BUDGET's cap: cap / rows within [SHARE_MIN, floor),
        Flash-Next's rule from the call's shape (a graph's bucket, every request counted); 0 without a cap or when the
        rule gives the floor itself."""
        if not self.budget:
            return 0
        cap = min(self.budget, self.per_request * (rows // steps)) if self.per_request else self.budget
        g = max(min(self.SHARE_MIN, self.keep), min(self.keep, cap // rows))
        return g if g < self.keep else 0

    def routed(self, latent: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """The routed experts' weighted sum (times routed_scaling_factor) for BF16 latent [M, L] -> BF16 [M, L], for a
        call past SMALL_ROWS rows (3-bit experts: any size)."""
        if self.w3 is not None:
            from sharar.kernels.w3_moe import w3_moe
            return w3_moe(self.w3, latent, ids, weights)
        return self.b12x(latent, ids, weights)

    def small(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """The block through `kernels/nemotron_moe.py` (at most SMALL_ROWS rows): the router kernel also routes and
        plans as `route` would with route_topk; under budgeted selection it leaves the routing to `route_budget`."""
        from sharar.kernels import nemotron_moe
        w = self.small_weights()
        k, slots, steps = self.topk, None, 1
        if ctx is not None and ctx.mode in ("decode", "verify"):
            slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
            if self.keep:  # route_budget's routing
                fused, _, _, ws = nemotron_moe.router(x, w, self.bias, self.topk, route=False)
                ids, weights, lat = self.route(fused, ctx)
                if self.w3 is not None:
                    return nemotron_moe.w3_block(x, ws, lat, ids, weights, w, self.w3)
                return nemotron_moe.block(x, ws, ids, weights, w, plan=True)
        if self.w3 is not None:  # padded rows name no expert (id E), which w3_moe skips
            _, ids, weights, ws, lat = nemotron_moe.router(x, w, self.bias, k, slots, steps, pad_id=self.experts, latent=True)
            return nemotron_moe.w3_block(x, ws, lat, ids, weights, w, self.w3)
        _, ids, weights, ws = nemotron_moe.router(x, w, self.bias, k, slots, steps)
        return nemotron_moe.block(x, ws, ids, weights, w)

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        from sharar.kernels import ops
        return ops.nvfp4_linear(relu2_(ops.nvfp4_linear(x, self.shared_up)), self.shared_down)

    def forward(self, x: torch.Tensor, ctx=None, x_q=None) -> torch.Tensor:
        """Post-norm rows x [M, H] BF16 -> the block's mixer output [M, H] BF16 (the residual add is the caller's).
        `x_q`: the mixers' common signature; this block reads BF16 x."""
        x = x.contiguous()
        if x.shape[0] <= self.SMALL_ROWS:
            return self.small(x, ctx)
        fused = torch.mm(x, self.gate_fc1.T, out_dtype=torch.float32)  # [M, E + L]
        ids, weights, latent = self.route(fused, ctx)
        routed = self.routed(latent, ids, weights)
        return torch.addmm(self.shared(x), routed, self.fc2.T)

    __call__ = forward

    @torch.no_grad()
    def reference(self, x: torch.Tensor, quantize_activations: bool = False) -> torch.Tensor:
        """The checkpoint's definition in FP32 torch from this block's resident weights (experts dequantized as used):
        x [M, H] -> FP32 [M, H]. `quantize_activations` rounds the expert and shared-expert inputs to NVFP4 with their
        calibrated scales, as the served kernels do; otherwise activations stay FP32 (the HF module's math).
        NVFP4 experts only."""
        if self.w3 is not None:
            raise NotImplementedError("reference(): the block serves 3-bit experts (kernels/w3_moe.reference)")
        from sharar.kernels.w3_moe import unswizzle_batched
        from sharar.weights.modelopt import unpack_fp4, unswizzle_block_scales

        def qdq(v: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            if not quantize_activations:
                return v
            from sharar.kernels.nvfp4_quant import nvfp4_quantize
            gs = (1.0 / scale).float()
            q, sf = nvfp4_quantize(v.bfloat16().contiguous(), gs)
            sf = unswizzle_block_scales(sf.reshape(-1).view(torch.float8_e4m3fn), v.shape[0], v.shape[1] // 16)
            return unpack_fp4(q, sf) / gs

        xf = x.float()
        fused = xf @ self.gate_fc1.float().T
        s = fused[:, :self.experts].sigmoid()
        ids = torch.topk(s + self.bias, self.topk, dim=-1).indices
        wt = s.gather(1, ids)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * c.ROUTED_SCALING_FACTOR
        lat = fused[:, self.experts:]
        if quantize_activations:
            lat = lat.bfloat16().float()
        lat_q = qdq(lat, self.in1)
        s1 = unswizzle_batched(self.s1, self.inter, self.latent // 16)
        s2 = unswizzle_batched(self.s2, self.latent, self.inter // 16)
        acc = torch.zeros(x.shape[0], self.latent, dtype=torch.float32, device=x.device)
        for e in ids.unique().tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            up = unpack_fp4(self.w1[e], s1[e]) * self.g1[e]
            down = unpack_fp4(self.w2[e], s2[e]) * self.g2[e]
            h = torch.relu(lat_q[rows] @ up.T) ** 2
            acc.index_add_(0, rows, (qdq(h, self.in2) @ down.T) * wt[rows, slot, None])
        if quantize_activations:
            acc = acc.bfloat16().float()
        up, down = self.shared_up.dequantize(), self.shared_down.dequantize()
        shared_in = 1.0 / self.shared_up.input_scale_inv
        shared_mid = 1.0 / self.shared_down.input_scale_inv
        h = torch.relu(qdq(xf, shared_in) @ up.T) ** 2
        return acc @ self.fc2.float().T + qdq(h, shared_mid) @ down.T
