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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = ops.nvfp4_linear(x, self.gate_up)
        if self.fused_silu_quant:
            # silu(gate) * up in fp32, quantized straight to NVFP4 (Triton; the
            # FlashInfer experts kernel it replaces ran at ~120 GB/s here).
            pre = silu_mul_nvfp4_quantize(gate_up, self.down.input_scale_inv)
            return ops.nvfp4_linear(None, self.down, prequantized=pre)
        return ops.nvfp4_linear(ops.silu_and_mul(gate_up), self.down)
