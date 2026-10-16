# SPDX-License-Identifier: Apache-2.0
"""Routed experts (512, top-10 softmax routing) plus the sigmoid-gated shared expert.

Two expert formats, never both resident:
- NVFP4 W4A4 (the checkpoint's ModelOpt packing) through FlashInfer's CUTLASS fused MoE, with w13 = [up rows;
  gate rows] per expert, block scales swizzled per expert, activations quantized once per layer with the
  loudest expert's calibrated scale.
- 3-bit (#73) when the 3-bit copies are on and the W3 checkpoint carries this layer's experts: the layer then
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


def load_nvfp4_experts(w, prefix: str, experts: int, device, rows=None, cols=None) -> dict[str, torch.Tensor]:
    """A layer's NVFP4 experts as stored: w13 [E, 2I, H/2] = [up rows; gate rows] and w2 [E, H, I/2] uint8 (low nibble
    first), their float8_e4m3fn block scales s13 / s2 (unswizzled), the per-expert global scales scale2_13 / scale2_2
    and input scales in13 / in2 (FP32 [E]). `rows` / `cols` cut this rank's tensor-parallel part (default: whole)."""
    rows = rows or (lambda t: t)
    cols = cols or (lambda t: t)
    p = f"{prefix}.experts"
    w13, w2, s13, s2, scale2_13, scale2_2, in13, in2 = [], [], [], [], [], [], [], []
    for e in range(experts):
        g, u, d = (f"{p}.{e}.gate_proj", f"{p}.{e}.up_proj", f"{p}.{e}.down_proj")
        # FlashInfer's CUTLASS fused MoE reads fc1 as [up rows; gate rows] (measured: the other order
        # gives cosine 0.79 against the dequantized reference, this one 0.999).
        w13.append(torch.cat([rows(w.get(f"{u}.weight")), rows(w.get(f"{g}.weight"))], dim=0))
        w2.append(cols(w.get(f"{d}.weight")))  # packed [K, I/2] and block scales [K, I/16]: this rank's columns
        s13.append(torch.cat([rows(w.get(f"{u}.weight_scale")), rows(w.get(f"{g}.weight_scale"))], dim=0))
        s2.append(cols(w.get(f"{d}.weight_scale")))
        g2, u2 = w.get(f"{g}.weight_scale_2").float().reshape(()), w.get(f"{u}.weight_scale_2").float().reshape(())
        if not torch.equal(g2, u2):
            raise ValueError(f"expert {e}: gate and up global scales differ; the fused kernel needs one")
        scale2_13.append(g2)
        scale2_2.append(w.get(f"{d}.weight_scale_2").float().reshape(()))
        in13.append(torch.maximum(w.get(f"{g}.input_scale").float().reshape(()), w.get(f"{u}.input_scale").float().reshape(())))
        in2.append(w.get(f"{d}.input_scale").float().reshape(()))
    out = {"w13": w13, "w2": w2, "s13": s13, "s2": s2, "scale2_13": scale2_13, "scale2_2": scale2_2, "in13": in13, "in2": in2}
    return {k: torch.stack(v).to(device) for k, v in out.items()}


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
        # Expert selection in decode and verify forwards: each token keeps its own top `keep` experts, then reuses up
        # to `reuse` more from those the call's other real tokens keep (route_subst), and with `fallback` fills the
        # reuse slots nothing fills from its own next ranks. Without it they name no expert (id = experts), which the
        # w3 kernel skips, or for the NVFP4 experts, which need an expert in every slot, the token's own top 1 at
        # weight 0 (the same bits as leaving the slot out). Keeping topk or more is the served route.
        for name, v in (("SHARAR_MOE_KEEP", cfg.moe_keep), ("SHARAR_MOE_REUSE", cfg.moe_reuse)):
            if v < 0:
                raise ValueError(f"{name}={v}: a count of experts, or 0 (off)")
        keep = cfg.moe_keep if cfg.moe_keep < self.topk else 0
        reuse = cfg.moe_reuse if keep else 0  # SHARAR_MOE_KEEP=0 turns the selection off, whatever the reuse
        if keep + reuse > self.topk:
            raise ValueError(f"SHARAR_MOE_KEEP + SHARAR_MOE_REUSE = {keep + reuse}: at most {self.topk} experts a token")
        self.keep, self.reuse, self.fallback = keep, reuse, cfg.moe_reuse_fallback
        self.w3 = w3
        if w3 is not None:
            if w3.base + w3.experts > self.experts:
                raise ValueError(f"3-bit experts {w3.base}..{w3.base + w3.experts - 1}: the model has {self.experts}")
            return
        nv = load_nvfp4_experts(w, prefix, self.experts, device, ColumnParallel((self.inter,)).rows, ROW_PARALLEL.cols)
        self.sim3_error = None
        group = cfg.flashnext_experts_sim3
        if group:  # simulated low-bit experts (weights/expert_sim3.py): the same kernel, the format's values (a screen)
            from sharar.weights.expert_sim3 import requantize
            nv["w13"], nv["s13"], e13 = requantize(nv["w13"], nv["s13"], group)
            nv["w2"], nv["s2"], e2 = requantize(nv["w2"], nv["s2"], group)
            self.sim3_error = (e13, e2)
            torch.cuda.empty_cache()
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
            from sharar.kernels.flashnext_fused import route_subst, route_topk
            if not self.reuse:
                return route_topk(logits, self.keep)  # ids and weights [rows, keep]: the routed experts take any k <= 10
            # A graph pads with requests on the dummy state slot 0, which route_subst leaves out of the core. A
            # verify forward's rows are request-major, verify_steps per request, and every request drafts the
            # call's depth (adaptive depth verifies the deepest any request wants), so every row of a real request
            # is real.
            width = self.keep + self.reuse
            if ctx.mode == "decode":
                return route_subst(logits, width, self.keep, ctx.slots, fallback=self.fallback, repeat=self.w3 is None)
            return route_subst(logits, width, self.keep, ctx.seq_slots, ctx.verify_steps, fallback=self.fallback,
                               repeat=self.w3 is None)
        if x.shape[0] <= 64:
            from sharar.kernels.flashnext_fused import route_topk
            return route_topk(logits, self.topk)
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return ids.to(torch.int32), weights.contiguous()

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
            # configuration that runs the one-row decode GEMMs at half speed (profile 2026-09-19).
            tune_max_num_tokens=max(1, 1 << (m - 1).bit_length()),
        )
        return routed[0] if isinstance(routed, (list, tuple)) else routed  # flashinfer 0.6 returns [output]

    def __call__(self, x: torch.Tensor, ctx) -> torch.Tensor:
        x = x.contiguous()
        ids, weights = self.route(x, ctx)
        routed = self.routed(x, ids, weights)
        from sharar.kernels.flashnext_fused import ACT_SILU_MUL, moe_combine
        shared = self.shared_down(self.shared_gate_up(x), act=ACT_SILU_MUL)  # SiLU-mul inside the down projection's quantization
        return moe_combine(routed, shared, x, self.shared_gate_vec)
