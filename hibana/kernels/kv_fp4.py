# SPDX-License-Identifier: Apache-2.0
"""NVFP4 KV values: E2M1 codes with one E4M3 scale per 16 values along head_dim (global scale 1).

`nvfp4_round` is the reference rounding, FlashInfer's NVFP4 KV writer's arithmetic (`page.cuh`
`nvfp4_append_quantize_block`, global scale 1): a value becomes scale * e2m1(value * (1 / scale)), scale = e4m3(amax / 6)
of its block of 16, the code rounded to nearest with ties to the even code and saturating at 6. A block of zeros stays
zero. Every result is exactly representable in BF16 (a code's two significant bits times a scale's four), so a BF16
pool holding rounded values is an FP4 pool's exact numerics (`SHARAR_KV_FORMAT=sim-fp4`, the accuracy screen).
"""
import torch

BLOCK = 16
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # E2M1 magnitudes; codes 0..7, even codes 0, 1.0, 2.0, 4.0


def nvfp4_round(x: torch.Tensor) -> torch.Tensor:
    """x [..., D] (D a multiple of 16) -> the NVFP4 value of each element, in x's dtype."""
    d = x.shape[-1]
    if d % BLOCK:
        raise ValueError(f"NVFP4 rounding needs a last dimension divisible by {BLOCK}, not {d}")
    v = x.float().reshape(*x.shape[:-1], d // BLOCK, BLOCK)
    amax = v.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    # times the reciprocal, as FlashInfer's NVFP4 KV writer (page.cuh nvfp4_append_quantize_block) scales a block
    q = v * torch.where(scale > 0, 1.0 / torch.where(scale > 0, scale, 1.0), torch.zeros_like(scale))
    a = q.abs()
    # nearest E2M1 magnitude, a tie to the even code (0, 1, 2, 4), saturating at 6; constants only (graph capture)
    mag = torch.zeros_like(a)
    for lower, inclusive, value in ((0.25, False, 0.5), (0.75, True, 1.0), (1.25, False, 1.5), (1.75, True, 2.0),
                                    (2.5, False, 3.0), (3.5, True, 4.0), (5.0, False, 6.0)):
        mag = torch.where(a >= lower if inclusive else a > lower, value, mag)
    return (torch.copysign(mag, q) * scale).reshape(x.shape).to(x.dtype)


def fp8_round(x: torch.Tensor) -> torch.Tensor:
    """x through E4M3 (scale 1, as the FP8 pool stores K/V) and back to x's dtype."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def _same(x: torch.Tensor) -> torch.Tensor:
    return x


SIMULATED = {
    # SHARAR_KV_FORMAT: (key rounding, value rounding) applied on every write into a BF16 pool
    "sim-bf16": (_same, _same),  # the unrounded reference the formats are measured against
    "sim-fp4": (nvfp4_round, nvfp4_round),
    "sim-k8v4": (fp8_round, nvfp4_round),
}
