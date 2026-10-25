"""Bolt's checkpoint-native weight-only projections; activations stay BF16."""

from dataclasses import dataclass

import torch

from sharar.kernels import ggml_bf16
from sharar.weights.ggml import dequantize
from sharar.weights.modelopt import unpack_fp4


@dataclass
class PackedLinear:
    weight: torch.Tensor
    quant: str
    scales: torch.Tensor | None = None
    gain: torch.Tensor | None = None
    input_scale: object = None  # the common norm path emits BF16, without an FP8 copy
    w3: object = None

    @classmethod
    def load(cls, w, prefixes, device, shard=None):
        weights = [w.get(f"{p}.weight").to(device) for p in prefixes]
        if weights[0].ndim == 3:
            if any(v.ndim != 3 or v.shape[1:] != weights[0].shape[1:] for v in weights):
                raise ValueError("fused Bolt projections must share their encoding and input width")
            quant = {34: "Q8_0", 66: "IQ2_XXS"}[weights[0].shape[-1]]
            return cls(torch.cat(weights, dim=0).contiguous(), quant)
        if len(prefixes) != 1:
            raise ValueError("Bolt's NVFP4 projections are separate")
        p = prefixes[0]
        return cls(weights[0].contiguous(), "NVFP4", w.get(f"{p}.weight_scale").to(device).contiguous(),
                   w.get(f"{p}.weight_scale_2").to(device).float().reshape(()))

    @property
    def out_features(self):
        return self.weight.shape[0]

    @property
    def in_features(self):
        return self.weight.shape[1] * {"Q8_0": 32, "IQ2_XXS": 256, "NVFP4": 2}[self.quant]

    @property
    def bytes(self):
        return self.weight.numel() + sum(t.numel() * t.element_size() for t in (self.scales, self.gain) if t is not None)

    def dequantize(self):
        if self.quant == "NVFP4":
            return unpack_fp4(self.weight, self.scales) * self.gain
        return dequantize(self.weight, self.quant)

    def __call__(self, x, *, relu2=False):
        return ggml_bf16.matmul(x, self.weight, self.quant, scales=self.scales, gain=self.gain, relu2=relu2)
