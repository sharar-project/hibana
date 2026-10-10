# SPDX-License-Identifier: Apache-2.0
"""Routed experts (512 x NVFP4 W4A4, top-10 softmax routing) plus the sigmoid-gated shared expert.

The expert matrices keep the checkpoint's ModelOpt packing and run through FlashInfer's CUTLASS
fused MoE with w13 = [up rows; gate rows] per expert, block scales swizzled per
expert, activations quantized once per layer with the loudest expert's calibrated scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel


def swizzle_block_scales_batched(scales: torch.Tensor) -> torch.Tensor:
    """[E, M, K] float8_e4m3fn block scales -> CUTLASS SM120 interleave, viewed as int32 [E, M_pad, K_pad/4]."""
    e, m, k = scales.shape
    m_pad, k_pad = (m + 127) // 128 * 128, (k + 3) // 4 * 4
    padded = torch.zeros((e, m_pad, k_pad), dtype=scales.dtype, device=scales.device)
    padded[:, :m, :k] = scales
    padded = padded.reshape(e, m_pad // 128, 4, 32, k_pad // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return padded.reshape(e, m_pad, k_pad).view(torch.int32)


class MoeLayer:
    def __init__(self, w, prefix: str, device: torch.device, target, linear_cls, precision: str):
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
        rows = ColumnParallel((self.inter,)).rows
        cols = ROW_PARALLEL.cols
        p = f"{prefix}.experts"
        w13, w2, s13, s2, scale2_13, scale2_2, in13, in2 = [], [], [], [], [], [], [], []
        for e in range(self.experts):
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
        w13, w2 = torch.stack(w13).to(device), torch.stack(w2).to(device)  # [E, 2I, K/2], [E, K, I/2] uint8
        s13, s2 = torch.stack(s13).to(device), torch.stack(s2).to(device)
        self.sim3_error = None
        from sharar.engine.config import current
        group = current().flashnext_experts_sim3
        if group:  # simulated 3-bit experts (weights/expert_sim3.py): the same kernel, the 3-bit format's values
            from sharar.weights.expert_sim3 import requantize
            w13, s13, e13 = requantize(w13, s13, group)
            w2, s2, e2 = requantize(w2, s2, group)
            self.sim3_error = (e13, e2)
            del e13, e2
            torch.cuda.empty_cache()  # the rounding's temporaries would otherwise shrink the pools sized after load
        self.w13, self.w2 = w13, w2
        self.s13 = swizzle_block_scales_batched(s13)
        self.s2 = swizzle_block_scales_batched(s2)
        w13_in, w2_in = torch.stack(in13).max(), torch.stack(in2).max()
        self.a13 = (1.0 / w13_in).to(device)
        self.a2 = (1.0 / w2_in).to(device)
        self.g1 = (w13_in * torch.stack(scale2_13)).to(device)  # [E]
        self.g2 = (w2_in * torch.stack(scale2_2)).to(device)
        self.quant_scales = [self.a13, self.s13, self.g1, self.a2, self.s2, self.g2]
        self.w13_long, self.w2_long = self.w13.view(torch.long), self.w2.view(torch.long)

    @property
    def bytes(self) -> int:
        return (self.w13.numel() + self.w2.numel() + self.s13.numel() * 4 + self.s2.numel() * 4 + self.router.numel() * 2
                + self.shared_gate_up.bytes + self.shared_down.bytes)

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.router.T
        if x.shape[0] <= 64:
            from sharar.kernels.flashnext_fused import route_topk
            return route_topk(logits, self.topk)
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        return ids.to(torch.int32), weights.contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe
        x = x.contiguous()
        ids, weights = self.route(x)
        m = x.shape[0]
        routed = cutlass_fused_moe(
            input=x, token_selected_experts=ids, token_final_scales=weights,
            fc1_expert_weights=self.w13_long, fc2_expert_weights=self.w2_long,
            output_dtype=torch.bfloat16, quant_scales=self.quant_scales,
            # The tactic is chosen for this token count; the default (8192) picks a large-M
            # configuration that runs the one-row decode GEMMs at half speed (profile 2026-09-19).
            tune_max_num_tokens=max(1, 1 << (m - 1).bit_length()),
        )
        if isinstance(routed, (list, tuple)):  # flashinfer 0.6 returns [output]
            routed = routed[0]
        from sharar.kernels.flashnext_fused import ACT_SILU_MUL, moe_combine
        shared = self.shared_down(self.shared_gate_up(x), act=ACT_SILU_MUL)  # SiLU-mul inside the down projection's quantization
        return moe_combine(routed, shared, x, self.shared_gate_vec)
