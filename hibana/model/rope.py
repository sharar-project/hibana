"""Partial rotary embedding (NeoX style) over the first rotary_dim of each head.

For text the multimodal rope sections collapse to one 1-D position, so this is
plain RoPE with rotary_dim = head_dim * partial_rotary_factor.
"""

from __future__ import annotations

import torch


class PartialRotary:
    def __init__(self, head_dim: int, rotary_dim: int, base: float, max_positions: int, device: torch.device):
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
        t = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        # Keep the target's cache and rotary products in FP32. Rounding either
        # to BF16 changes attention, including the first prefill chunk.
        self.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def __call__(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x: [T, heads, head_dim] bf16. Returns a new tensor."""
        cos_sin = self.cos_sin[positions]  # [T, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rot = x[..., : self.rotary_dim]
        rest = x[..., self.rotary_dim :]
        x1, x2 = rot.float().chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1.to(x.dtype), o2.to(x.dtype), rest), dim=-1)
