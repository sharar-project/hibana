"""Dense SwiGLU MLP with NVFP4 gate/up and down projections."""

from __future__ import annotations

import torch

from hibana.kernels import mlp_fusion, ops
from hibana.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize
from hibana.weights.modelopt import Nvfp4Linear, interleave_pairs, load_nvfp4_linear
from hibana.weights.sharding import ROW_PARALLEL, ColumnParallel
from hibana.weights.safetensors import SnapshotWeights


class Mlp:
    def __init__(self, w: SnapshotWeights, prefix: str, device: torch.device, intermediate_size: int, fused_silu_quant: bool = True):
        """`intermediate_size`: this rank's MLP columns (gate|up column-parallel, down row-parallel)."""
        # gate and up interleaved in blocks of 64 rows, so the fused prefill GEMM's tiles hold matching columns
        self.gate_up: Nvfp4Linear = interleave_pairs(load_nvfp4_linear(w, [f"{prefix}.gate_proj", f"{prefix}.up_proj"], device,
                                                                         ColumnParallel((intermediate_size, intermediate_size))),
                                                       mlp_fusion.PAIRS)
        self.down: Nvfp4Linear = load_nvfp4_linear(w, [f"{prefix}.down_proj"], device, ROW_PARALLEL)
        self.fused_silu_quant = fused_silu_quant
        if fused_silu_quant:
            mlp_fusion.available(self.gate_up.weight.device)  # compile the fused prefill kernel at load (once)

    def __call__(self, x: torch.Tensor | None, prequantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """x [rows, H] BF16, or None with `prequantized`: x already quantized for gate_up (the fused prefill norm)."""
        rows = x.shape[0] if x is not None else prequantized[0].shape[0]
        if self.fused_silu_quant and mlp_fusion.applies(rows, self.gate_up):
            # Prefill: SiLU(gate) * up quantized for the down projection in the gate/up GEMM's epilogue (the same bytes)
            pre = prequantized if prequantized is not None else nvfp4_quantize(x, self.gate_up.input_scale_inv)
            pre = mlp_fusion.gate_up_silu_quantize(pre, self.gate_up, self.down.input_scale_inv)
            return ops.nvfp4_linear(None, self.down, prequantized=pre)
        gate_up = ops.nvfp4_linear(x, self.gate_up, prequantized=prequantized)
        pairs = self.gate_up.pairs
        if ops.w3_decode(self.down, gate_up.shape[0]):
            # 3-bit decode: the SiLU product in BF16 quantized per 32-group for the 3-bit GEMM, in one launch (the bits of
            # silu_and_mul and quant_groups); no NVFP4 activation quantization
            from hibana.kernels import w3
            from hibana.kernels.flashnext_fused import silu_mul_quant_groups
            x_q, x_scale = silu_mul_quant_groups(gate_up, pairs, torch_chain=True)
            return w3.w3_linear(x_q, self.down.w3, x_scale)
        if self.fused_silu_quant:
            # Compute in FP32, preserve the BF16 product boundary, then pack
            # directly to NVFP4 without a materialized activation tensor.
            pre = silu_mul_nvfp4_quantize(gate_up, self.down.input_scale_inv, pairs)
            return ops.nvfp4_linear(None, self.down, prequantized=pre)
        return ops.nvfp4_linear(ops.silu_and_mul(gate_up, pairs), self.down)
