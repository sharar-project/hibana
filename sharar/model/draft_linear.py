"""Dynamic-activation FP8 linears for the parallel draft's large matrices."""
from dataclasses import dataclass

import torch


@dataclass
class DraftFp8Linear:
    weight: torch.Tensor
    scale: torch.Tensor
    w3: object = None  # 3-bit decode copy (sharar.kernels.w3.W3Weight) when SHARAR_W3_DRAFT=1

    @classmethod
    def from_bf16(cls, weight):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError('draft FP8 requires a BF16 matrix')
        scale = (weight.float().abs().amax() / 448).clamp_min(1e-12).reshape(1, 1)
        packed = (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return cls(packed, scale)

    @property
    def bytes(self):
        return self.weight.numel() + self.scale.numel() * self.scale.element_size()

    def __call__(self, x):
        from sharar.kernels.draft_fp8 import quant_dynamic
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if self.w3 is not None:
            from sharar.kernels import w3
            if flat.shape[0] <= w3.MAX_ROWS:
                return w3.w3_linear(flat, self.w3).reshape(*x.shape[:-1], self.weight.shape[0])
        packed, scale = quant_dynamic(flat)
        out = torch._scaled_mm(packed, self.weight.T, scale_a=scale, scale_b=self.scale,
                               out_dtype=torch.bfloat16)
        return out.reshape(*x.shape[:-1], self.weight.shape[0])
