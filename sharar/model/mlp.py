"""Dense SwiGLU MLP with NVFP4 gate/up and down projections."""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.kernels.nvfp4_quant import silu_mul_nvfp4_quantize
from sharar.weights.modelopt import Nvfp4Linear, load_nvfp4_linear
from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel
from sharar.weights.safetensors import SnapshotWeights


class Mlp:
    def __init__(self, w: SnapshotWeights, prefix: str, device: torch.device, intermediate_size: int, fused_silu_quant: bool = True):
        """`intermediate_size`: this rank's MLP columns (gate|up column-parallel, down row-parallel)."""
        self.gate_up: Nvfp4Linear = load_nvfp4_linear(w, [f"{prefix}.gate_proj", f"{prefix}.up_proj"], device,
                                                       ColumnParallel((intermediate_size, intermediate_size)))
        self.down: Nvfp4Linear = load_nvfp4_linear(w, [f"{prefix}.down_proj"], device, ROW_PARALLEL)
        self.fused_silu_quant = fused_silu_quant

    def __call__(self, x: torch.Tensor | None, prequantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """x [rows, H] BF16, or None with `prequantized`: x already quantized for gate_up (the fused prefill norm)."""
        gate_up = ops.nvfp4_linear(x, self.gate_up, prequantized=prequantized)
        if ops.w3_decode(self.down, gate_up.shape[0]):
            # 3-bit decode: keep the SiLU product in BF16, no NVFP4 activation quantization
            return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)
        if self.fused_silu_quant:
            # Compute in FP32, preserve the BF16 product boundary, then pack
            # directly to NVFP4 without a materialized activation tensor.
            pre = silu_mul_nvfp4_quantize(gate_up, self.down.input_scale_inv)
            return ops.nvfp4_linear(None, self.down, prequantized=pre)
        return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)
