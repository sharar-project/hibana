"""Dense SwiGLU MLP with NVFP4 gate/up and down projections."""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.kernels.nvfp4_quant import silu_mul_nvfp4_quantize
from sharar.weights.modelopt import Nvfp4Linear, load_nvfp4_linear
from sharar.weights.safetensors import SnapshotWeights


class Mlp:
    def __init__(self, w: SnapshotWeights, prefix: str, device: torch.device, fused_silu_quant: bool = True):
        self.gate_up: Nvfp4Linear = load_nvfp4_linear(w, [f"{prefix}.gate_proj", f"{prefix}.up_proj"], device)
        self.down: Nvfp4Linear = load_nvfp4_linear(w, [f"{prefix}.down_proj"], device)
        self.fused_silu_quant = fused_silu_quant
        self._down_scale_inv_1d = self.down.input_scale_inv.reshape(1)
        self._masked_m: dict[int, torch.Tensor] = {}

    def forward_prequantized(self, pre: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """MLP on an input already quantized to NVFP4 (by the fused norm)."""
        gate_up = ops.nvfp4_linear(None, self.gate_up, prequantized=pre)
        if self.fused_silu_quant:
            return ops.nvfp4_linear(None, self.down, prequantized=silu_mul_nvfp4_quantize(gate_up, self.down.input_scale_inv))
        return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = ops.nvfp4_linear(x, self.gate_up)
        if ops.w3_decode(self.down, x.shape[0]):
            # 3-bit decode: keep the SiLU product in BF16, no NVFP4 activation quantization
            return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)
        if self.fused_silu_quant:
            # Compute in FP32, preserve the BF16 product boundary, then pack
            # directly to NVFP4 without a materialized activation tensor.
            pre = silu_mul_nvfp4_quantize(gate_up, self.down.input_scale_inv)
            return ops.nvfp4_linear(None, self.down, prequantized=pre)
        return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)
