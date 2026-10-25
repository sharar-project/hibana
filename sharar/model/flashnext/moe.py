# SPDX-License-Identifier: Apache-2.0
"""Routed experts (512, top-10 softmax routing) plus the sigmoid-gated shared expert.

Two expert formats, never both resident:
- NVFP4 W4A4 (the checkpoint's ModelOpt packing) through FlashInfer's CUTLASS fused MoE, with w13 = [up rows;
  gate rows] per expert, block scales swizzled per expert, activations quantized once per layer with the
  loudest expert's calibrated scale.
- 3-bit when the 3-bit copies are on and the W3 checkpoint carries this layer's experts: the layer then
  never reads the NVFP4 experts, and every call, decode to prefill, runs `kernels/w3_moe.py` (W3 format per
  expert, E4M3 activations). Under tensor parallelism each rank holds a range of whole experts (expert parallel)
  and sums the pairs routed to them: a partial sum, as the split NVFP4 experts give.
"""

from __future__ import annotations

import torch

from sharar.engine.config import current
from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel


def swizzle_block_scales_batched(scales: torch.Tensor) -> torch.Tensor:
    """[E, M, K] float8_e4m3fn block scales -> CUTLASS SM120 interleave, viewed as int32 [E, M_pad, K_pad/4]."""
    e, m, k = scales.shape
    m_pad, k_pad = (m + 127) // 128 * 128, (k + 3) // 4 * 4
    padded = torch.zeros((e, m_pad, k_pad), dtype=scales.dtype, device=scales.device)
    padded[:, :m, :k] = scales
    padded = padded.reshape(e, m_pad // 128, 4, 32, k_pad // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return padded.reshape(e, m_pad, k_pad).view(torch.int32)


def nvfp4_expert_names(prefix: str, experts: int) -> list[str]:
    """Every tensor `load_nvfp4_experts` reads for the layer at `prefix`."""
    kinds = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    return [f"{prefix}.experts.{e}.{proj}.{kind}" for e in range(experts) for proj in ("gate_proj", "up_proj", "down_proj")
            for kind in kinds]


def load_nvfp4_experts(w, prefix: str, experts: int, device, rows=None, cols=None) -> dict[str, torch.Tensor]:
    """A layer's NVFP4 experts as stored: w13 [E, 2I, H/2] = [up rows; gate rows] and w2 [E, H, I/2] uint8 (low nibble
    first), their float8_e4m3fn block scales s13 / s2 (unswizzled), the per-expert global scales scale2_13 / scale2_2
    and input scales in13 / in2 (FP32 [E]). `rows` / `cols` cut this rank's tensor-parallel part (default: whole).
    The layer's expert files are read whole and copied to the device once (`SnapshotWeights.get_whole`); the stacks
    are built there."""
    rows = rows or (lambda t: t)
    cols = cols or (lambda t: t)
    t = w.get_whole(nvfp4_expert_names(prefix, experts), device)

    def each(proj: str, kind: str, cut=lambda x: x) -> torch.Tensor:
        return torch.stack([cut(t[f"{prefix}.experts.{e}.{proj}.{kind}"]) for e in range(experts)])

    # FlashInfer's CUTLASS fused MoE reads fc1 as [up rows; gate rows] (measured: the other order
    # gives cosine 0.79 against the dequantized reference, this one 0.999).
    w13 = torch.cat([each("up_proj", "weight", rows), each("gate_proj", "weight", rows)], dim=1)
    s13 = torch.cat([each("up_proj", "weight_scale", rows), each("gate_proj", "weight_scale", rows)], dim=1)
    # packed [K, I/2] and block scales [K, I/16]: this rank's columns
    w2, s2 = each("down_proj", "weight", cols), each("down_proj", "weight_scale", cols)
    g2, u2 = each("gate_proj", "weight_scale_2").float(), each("up_proj", "weight_scale_2").float()
    differ = (g2 != u2).nonzero().flatten().tolist()
    if differ:
        raise ValueError(f"expert {differ[0]}: gate and up global scales differ; the fused kernel needs one")
    out = {"w13": w13, "w2": w2, "s13": s13, "s2": s2, "scale2_13": g2, "scale2_2": each("down_proj", "weight_scale_2").float(),
           "in13": torch.maximum(each("gate_proj", "input_scale").float(), each("up_proj", "input_scale").float()),
           "in2": each("down_proj", "input_scale").float()}
    return {k: v.contiguous() for k, v in out.items()}


class MoeLayer:
    def __init__(self, w, prefix: str, device: torch.device, target, linear_cls, precision: str, w3=None):
        """`w3`: this layer's 3-bit experts (kernels/w3_moe.W3Experts, from the W3 checkpoint); the NVFP4 experts are
        then not read at all."""
        self.device = device
        self.experts, self.topk, self.hidden = target.num_experts, target.experts_per_token, target.hidden_size
        self.inter = target.moe_intermediate_size
        self.router = w.get(f"{prefix}.gate.weight").to(device).contiguous()  # [E, H] bf16
        self.shared_gate = w.get(f"{prefix}.shared_expert_gate.weight").to(device).float().reshape(1, -1)  # [1, H]
        self.shared_gate_vec = self.shared_gate.reshape(-1).contiguous()
        # Tensor parallel (`Target.shard`): every expert's intermediate is split, gate|up column-parallel and down
        # row-parallel, routed and shared alike, so each rank streams half of every selected expert and returns a
        # partial sum the decoder layer all-reduces. Router and gates stay whole.
        shared_inter = target.shared_expert_intermediate_size
        self.shared_gate_up = linear_cls.load(w, [f"{prefix}.shared_expert.gate_proj.weight", f"{prefix}.shared_expert.up_proj.weight"], device, precision,
                                              ColumnParallel((shared_inter, shared_inter)))
        self.shared_down = linear_cls.load(w, [f"{prefix}.shared_expert.down_proj.weight"], device, precision, ROW_PARALLEL)
        cfg = current()
        # Expert selection in decode and verify forwards (SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET, route_budget): the call
        # reads at most a budget of distinct experts, chosen for all its tokens; each token keeps its own top experts
        # (`guarantee`, at most `keep`), then takes others from the set, up to topk. Slots left over name no expert,
        # which the w3 kernel skips, or for the NVFP4 experts (an expert in every slot) the token's own top 1 at
        # weight 0, the same bits. SHARAR_MOE_KEEP alone: each token's top `keep`. Keeping topk or more is the served route.
        self.keep = cfg.moe_keep if cfg.moe_keep < self.topk else 0
        self.budget = cfg.moe_budget if self.keep else 0
        self.per_request = cfg.moe_budget_per_request
        self.w3 = w3
        if w3 is not None:
            if w3.base + w3.experts > self.experts:
                raise ValueError(f"3-bit experts {w3.base}..{w3.base + w3.experts - 1}: the model has {self.experts}")
            return
        nv = load_nvfp4_experts(w, prefix, self.experts, device, ColumnParallel((self.inter,)).rows, ROW_PARALLEL.cols)
        self.w13, self.w2 = nv["w13"], nv["w2"]  # [E, 2I, K/2], [E, K, I/2] uint8
        self.s13 = swizzle_block_scales_batched(nv["s13"])
        self.s2 = swizzle_block_scales_batched(nv["s2"])
        w13_in, w2_in = nv["in13"].max(), nv["in2"].max()
        self.a13 = 1.0 / w13_in
        self.a2 = 1.0 / w2_in
        self.g1 = w13_in * nv["scale2_13"]  # [E]
        self.g2 = w2_in * nv["scale2_2"]
        self.quant_scales = [self.a13, self.s13, self.g1, self.a2, self.s2, self.g2]
        self.w13_long, self.w2_long = self.w13.view(torch.long), self.w2.view(torch.long)

    @property
    def expert_bytes(self) -> int:
        """Resident bytes of all routed experts (weights and their scales)."""
        if self.w3 is not None:
            return self.w3.bytes
        return self.w13.numel() + self.w2.numel() + self.s13.numel() * 4 + self.s2.numel() * 4

    @property
    def bytes(self) -> int:
        return self.expert_bytes + self.router.numel() * 2 + self.shared_gate_up.bytes + self.shared_down.bytes

    def route(self, x: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.router.T
        if self.keep and ctx.mode in ("decode", "verify"):
            if not self.budget:
                from sharar.kernels.flashnext_fused import route_topk
                return route_topk(logits, self.keep)  # ids and weights [rows, keep]: the routed experts take any k <= 10
            from sharar.kernels.flashnext_fused import route_budget
            # A graph pads with requests on the dummy state slot 0, which route_budget leaves out. A verify forward's
            # rows are request-major, verify_steps per request, and every request drafts the call's depth (adaptive
            # depth verifies the deepest any request wants), so every row of a real request is real.
            slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
            return route_budget(logits, self.topk, self.guarantee(logits.shape[0], steps), self.budget, slots, steps,
                                repeat=self.w3 is None, per_request=self.per_request, min_cap=self.MIN_CAP)
        if x.shape[0] <= 64:
            from sharar.kernels.flashnext_fused import route_topk
            if ctx.mode in ("decode", "verify"):  # a graph's padded requests (dummy slot 0) read no extra experts
                slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
                return route_topk(logits, self.topk, slots, steps, repeat=self.w3 is None)
            return route_topk(logits, self.topk)
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return ids.to(torch.int32), weights.contiguous()

    MIN_GUARANTEE = 2  # below two experts a token, GSM8K fell 3.3 points at 16 requests (keep 1 + budget 64)
    # The per-request cap's floor: one request alone at 12 experts (a depth-8 draft: 9 rows, each guaranteed 2) lost 4.7
    # points of GSM8K-300 one request at a time (3 better / 17 worse); at 16 (guaranteed 3) -1.3 (4 / 8, p 0.39), as at 20
    MIN_CAP = 16

    def guarantee(self, rows: int, steps: int) -> int:
        """Each token's guaranteed top experts under SHARAR_MOE_BUDGET: 2 cap / rows (about half of the guaranteed slots
        are distinct experts, so they fill about the cap and it binds), within [MIN_GUARANTEE, SHARAR_MOE_KEEP]; from the
        call's shape (a graph's bucket). At 4 and 12:64: decode and verify forwards of up to 4 tokens a request keep 4
        (one request, whose cap is MIN_CAP: up to 8 tokens, a depth-8 draft 3), wider drafts and bigger batches fewer,
        64 tokens and more 2."""
        cap = min(self.budget, max(self.MIN_CAP, self.per_request * (rows // steps))) if self.per_request else self.budget
        return max(min(self.MIN_GUARANTEE, self.keep), min(self.keep, 2 * cap // rows))

    def routed(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """The routed experts' weighted sum for x [M, H] bf16 -> bf16 [M, H]."""
        if self.w3 is not None:
            from sharar.kernels.w3_moe import w3_moe
            return w3_moe(self.w3, x, ids, weights)
        from flashinfer.fused_moe import cutlass_fused_moe
        m = x.shape[0]
        routed = cutlass_fused_moe(
            input=x, token_selected_experts=ids, token_final_scales=weights,
            fc1_expert_weights=self.w13_long, fc2_expert_weights=self.w2_long,
            output_dtype=torch.bfloat16, quant_scales=self.quant_scales,
            # The tactic is chosen for this token count; the default (8192) picks a large-M
            # configuration that runs the one-row decode GEMMs at half speed (profiled).
            tune_max_num_tokens=max(1, 1 << (m - 1).bit_length()),
        )
        return routed[0] if isinstance(routed, (list, tuple)) else routed  # flashinfer 0.6 returns [output]

    def __call__(self, x: torch.Tensor, ctx) -> torch.Tensor:
        x = x.contiguous()
        ids, weights = self.route(x, ctx)
        from sharar.kernels.flashnext_fused import ACT_SILU_MUL, moe_combine, moe_combine_pairs
        if self.w3 is not None:
            # the 3-bit experts' per-pair outputs, summed by the combine itself: the same bits without the routed sum's round trip
            from sharar.kernels.w3_moe import w3_moe
            from sharar.kernels.w3_moe import WIDE_ROWS
            # prefill calls return the pairs in BF16: half the traffic of their widest tensor
            pairs = w3_moe(self.w3, x, ids, weights, pairs=True, pairs_bf16=x.shape[0] >= WIDE_ROWS)
            shared = self.shared_down(self.shared_gate_up(x), act=ACT_SILU_MUL)
            return moe_combine_pairs(pairs, ids.to(torch.int32).contiguous(), weights.float().contiguous(), shared, x, self.shared_gate_vec,
                                     self.w3.base, self.w3.experts)
        routed = self.routed(x, ids, weights)
        shared = self.shared_down(self.shared_gate_up(x), act=ACT_SILU_MUL)  # SiLU-mul inside the down projection's quantization
        return moe_combine(routed, shared, x, self.shared_gate_vec)
