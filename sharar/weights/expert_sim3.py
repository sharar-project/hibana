# SPDX-License-Identifier: Apache-2.0
"""Simulated 3-bit routed experts for Flash-Next (a quality screen: no new kernel, no fewer bytes).

A 3-bit expert format with eight levels {±0.5, ±1.5, ±3, ±4} times one E4M3 scale per `group` weights
(3.25 bits per weight at group 32, 3.5 at 16, against NVFP4's 4.5) is exactly representable in NVFP4:
the levels are E2M1 values and the group scale is the E4M3 block scale of each of its 16-value blocks.
`requantize` rewrites the checkpoint's NVFP4 experts into that format in place, so FlashInfer's CUTLASS
fused MoE computes what a 3-bit expert kernel would. The level set is the E2M1 subset closest to the
MSE-optimal eight-level codebook for bell-shaped weights (spacing 1:3:6:8 against Lloyd's 1:3.1:5.5:8.8).

Enabled with SHARAR_FLASHNEXT_EXPERTS_SIM3=<group> (16 or 32); the lossy-change gate judges it.
"""

from __future__ import annotations

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitude of code & 7; code & 8 is the sign
LEVELS = (0.5, 1.5, 3.0, 4.0)  # the eight-level set: these magnitudes, both signs
LEVEL_CODES = (1, 3, 5, 6)  # their E2M1 magnitude codes
SCALE_GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05)  # group scale = amax / 4 * f, rounded to E4M3
# A 2-bit screen (SHARAR_EXPERTS_SIM_BITS=2): four levels {+-1, +-3}, the E2M1 pair closest to the MSE-optimal
# four-level codebook for bell-shaped weights (spacing 1:3 against Lloyd's 1:3.3); 2.25 bits per weight at group 32.
LEVELS_2BIT, LEVEL_CODES_2BIT = (1.0, 3.0), (2, 5)
SCALE_GRID_2BIT = tuple(round(0.40 + 0.05 * i, 2) for i in range(14))  # four levels clip harder: amax / 3 * f


def _level_set():
    from sharar.engine.config import current
    if current().experts_sim_bits == 2:
        return LEVELS_2BIT, LEVEL_CODES_2BIT, SCALE_GRID_2BIT
    return LEVELS, LEVEL_CODES, SCALE_GRID


def _decode(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """uint8 [R, K/2] low-nibble-first E2M1 and float8_e4m3fn [R, K/16] -> float32 [R, K] (block-scaled)."""
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1).long()
    mags = torch.tensor(E2M1, device=packed.device)[codes & 7]
    return torch.where(codes & 8 != 0, -mags, mags) * scales.float().repeat_interleave(16, -1)


def _quantize(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 [R, K] -> (uint8 [R, K/2] E2M1 codes from the level set, float8_e4m3fn [R, K/16] block scales)."""
    r, k = w.shape
    g = w.reshape(r, k // group, group)
    mag = g.abs()
    amax = mag.amax(-1, keepdim=True)
    level_values, level_codes, scale_grid = _level_set()
    levels = torch.tensor(level_values, device=w.device)
    bounds = (levels[1:] + levels[:-1]) / 2
    best_err = best_idx = best_scale = None
    for f in scale_grid:
        scale = (amax / level_values[-1] * f).to(torch.float8_e4m3fn).float()
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        idx = torch.bucketize(mag / safe, bounds)
        err = (mag - levels[idx] * scale).square().sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_idx, best_scale = err, idx, scale
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_idx = torch.where(better, idx, best_idx)
            best_scale = torch.where(better, scale, best_scale)
    codes = torch.tensor(level_codes, device=w.device, dtype=torch.uint8)[best_idx]
    codes = torch.where(g < 0, codes | 8, codes).reshape(r, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scales = best_scale.expand(r, k // group, group // 16).reshape(r, k // 16).to(torch.float8_e4m3fn)
    return packed, scales


@torch.no_grad()
def requantize(packed: torch.Tensor, scales: torch.Tensor, group: int = 32, chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor, float]:
    """NVFP4 experts [E, R, K/2] uint8 and [E, R, K/16] float8_e4m3fn (unswizzled) -> the same tensors in the
    3-bit format, and the relative weight error against the source (Frobenius). The global scales are unchanged."""
    if group not in (16, 32):
        raise ValueError("the simulated 3-bit expert group is 16 or 32 weights")
    e, r, _ = packed.shape
    out_p, out_s = torch.empty_like(packed), torch.empty_like(scales)
    num = den = 0.0
    for lo in range(0, e, chunk):
        hi = min(e, lo + chunk)
        w = _decode(packed[lo:hi].reshape(-1, packed.shape[-1]), scales[lo:hi].reshape(-1, scales.shape[-1]))
        p, s = _quantize(w, group)
        num += (_decode(p, s) - w).square().sum().item()
        den += w.square().sum().item()
        out_p[lo:hi] = p.reshape(hi - lo, r, -1)
        out_s[lo:hi] = s.reshape(hi - lo, r, -1)
    return out_p, out_s, (num / den) ** 0.5 if den else 0.0
