"""Packed IQ2_XS, IQ2_XXS and Q8_0 weights for the grouped FP8-compute IQ2 MoE (`kernels/iq2_moe.py`).

Packing reorders encoded bytes without requantization: payloads tiled eight output rows at a time, block bases and
scales split into compact metadata. In the MMA each decoded integer is the exact sum of two E4M3 values, selected
through the execution table built here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from hibana.weights.ggml import FORMATS, _validate, dequantize
from hibana.weights.iq2_grid import GRID
from hibana.weights.iq2_xs_grid import GRID_XS

_CODEC = Path(__file__).resolve().parent / "csrc" / "ggml_codec.cuh"


@lru_cache(maxsize=None)
def _execution_lut(quant: str, device: torch.device) -> torch.Tensor:
    """IQ2 byte selectors, or two exact E4M3 components per Q8 integer."""
    if quant == "Q8_0":
        values = torch.arange(256).to(torch.uint8).view(torch.int8).float()
    else:
        grid = torch.tensor(GRID_XS if quant == "IQ2_XS" else GRID, dtype=torch.int64)
        codes = (grid == 25).long() + 2 * (grid == 43).long()
        # Eight nibbles select {8,24,44} and {0,1,-1} from register constants.
        # Their sums recover {8,25,43} exactly. One uint32 per eight weights.
        return (codes << (4 * torch.arange(8))).sum(1).to(device=device, dtype=torch.uint32)
    high = values.to(torch.float8_e4m3fn)
    low = (values - high.float()).to(torch.float8_e4m3fn)
    if quant == "Q8_0":
        table = high.view(torch.uint8).int() | (low.view(torch.uint8).int() << 8)
        return table.to(device=device, dtype=torch.uint32)


@dataclass
class GgmlWeight:
    """Owned compressed payload [N/8,K/Block,8,Bytes] and compact metadata.

    The cached execution table is shared by all weights of a format on a device.
    Encoded tensors must remain unmodified during execution.
    """
    values: torch.Tensor
    metadata: torch.Tensor
    quant: str
    out_features: int
    in_features: int
    lut: torch.Tensor

    @property
    def bytes(self) -> int:
        return self.values.numel() + self.metadata.numel()

    def blocks(self) -> torch.Tensor:
        """Reconstruct native blocks for reference/export, outside execution."""
        block, size = FORMATS[self.quant]
        kb, n = self.in_features // block, self.out_features
        payload = self.values.reshape(n // 8, kb, 8, -1).permute(0, 2, 1, 3).reshape(n, kb, -1)
        meta = self.metadata.permute(1, 0, 2)
        return torch.cat((meta[..., :2], payload, meta[..., 2:]), -1).reshape(n, kb, size)

    def dequantize(self) -> torch.Tensor:
        return dequantize(self.blocks(), self.quant)


def pack_weight(blocks: torch.Tensor, quant: str) -> GgmlWeight:
    """Losslessly retile native uint8 blocks; N must be divisible by eight."""
    block = _validate(blocks, quant)
    n, kb, _ = blocks.shape
    if not n or n % 8:
        raise ValueError("packed GGML weights require positive N divisible by eight")
    if blocks.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GGML weight preparation must precede CUDA graph capture")
    if not torch.isfinite(blocks[..., :2].contiguous().view(torch.float16)).all().item():
        raise ValueError("GGML block bases must be finite")
    payload = 32 if quant == "Q8_0" else 64
    values = blocks[..., 2:2 + payload].reshape(n // 8, 8, kb, payload).permute(0, 2, 1, 3)
    values = values.clone(memory_format=torch.contiguous_format)
    metadata = torch.cat((blocks[..., :2], blocks[..., 2 + payload:]), -1).permute(1, 0, 2).contiguous()
    return GgmlWeight(values, metadata, quant, n, kb * block, _execution_lut(quant, blocks.device))
