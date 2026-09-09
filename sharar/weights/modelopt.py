"""ModelOpt checkpoint layouts: per-tensor FP8 and block-scaled NVFP4 linears.

Ported in spirit from SGLang's ModelOptFp8LinearMethod / ModelOptFp4LinearMethod
(Apache-2.0); only the pieces this checkpoint uses. Fused projections (q/k/v,
qkv/z, gate/up) are concatenated along the output dimension exactly as the
reference does: FP8 shards are requantized to the maximum weight scale, NVFP4
shards are required to share one global scale. FP8 dequantization passes
through FP16 before requantization, including single-shard projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar.weights.safetensors import SnapshotWeights

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448


@dataclass
class Fp8Linear:
    weight: torch.Tensor  # [N, K] float8_e4m3fn, one scale for all rows
    weight_scale: torch.Tensor  # [] float32
    input_scale: torch.Tensor  # [] float32
    alpha: torch.Tensor = None  # [] float32 = input_scale * weight_scale

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = (self.input_scale * self.weight_scale).float()

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]


@dataclass
class Nvfp4Linear:
    weight: torch.Tensor  # [N, K/2] uint8, two FP4 values per byte
    weight_scale_swizzled: torch.Tensor  # [N_pad128, (K/16)_pad4] float8_e4m3fn in CUTLASS layout
    alpha: torch.Tensor  # [] float32 = input_scale * weight_scale_2
    input_scale_inv: torch.Tensor  # [] float32
    out_features: int
    in_features: int


def _to_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def load_fp8_linear(w: SnapshotWeights, prefixes: list[str], device: torch.device) -> Fp8Linear:
    """Load one FP8 linear, or several fused along the output dim."""
    weights = [w.get(f"{p}.weight") for p in prefixes]
    scales = [w.get(f"{p}.weight_scale").float().reshape(()) for p in prefixes]
    input_scales = [w.get(f"{p}.input_scale").float().reshape(()) for p in prefixes]
    max_scale = torch.stack(scales).max()
    parts = []
    for wt, sc in zip(weights, scales):
        wt = wt.to(device)
        # The checkpoint's reference loader rounds the dequantized values to
        # FP16. Skipping this boundary changes packed weights in fused shards.
        wt = _to_fp8(wt.half() * sc.to(device), max_scale.to(device))
        parts.append(wt)
    weight = torch.cat(parts, dim=0).contiguous()
    return Fp8Linear(
        weight=weight,
        weight_scale=max_scale.to(device),
        input_scale=torch.stack(input_scales).max().to(device),
    )


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """CUTLASS/FlashInfer SM120 NVFP4 block-scale layout (SGLang's interleave)."""
    m, k = scales.shape
    m_pad = (m + 127) // 128 * 128
    k_pad = (k + 3) // 4 * 4
    padded = torch.zeros((m_pad, k_pad), dtype=scales.dtype, device=scales.device)
    padded[:m, :k] = scales
    padded = padded.reshape(1, m_pad // 128, 4, 32, k_pad // 4, 4)
    padded = padded.permute(0, 1, 4, 3, 2, 5).contiguous()
    return padded.reshape(m_pad, k_pad)


def load_nvfp4_linear(w: SnapshotWeights, prefixes: list[str], device: torch.device) -> Nvfp4Linear:
    weights = [w.get(f"{p}.weight") for p in prefixes]
    scales = [w.get(f"{p}.weight_scale") for p in prefixes]
    scale2 = torch.stack([w.get(f"{p}.weight_scale_2").float().reshape(()) for p in prefixes])
    input_scale = torch.stack([w.get(f"{p}.input_scale").float().reshape(()) for p in prefixes])
    if len(prefixes) > 1 and not torch.all(scale2 == scale2[0]):
        raise ValueError(f"NVFP4 fused shards must share weight_scale_2: {prefixes} {scale2.tolist()}")
    weight = torch.cat([x.to(device) for x in weights], dim=0).contiguous()
    scale = torch.cat([x.to(device) for x in scales], dim=0)
    n, k_half = weight.shape
    k = k_half * 2
    if n % 32 or k % 32:
        raise ValueError(f"NVFP4 GEMM needs N and K divisible by 32: {prefixes} {n}x{k}")
    if scale.shape != (n, k // 16):
        raise ValueError(f"unexpected block-scale shape {tuple(scale.shape)} for {n}x{k}")
    input_scale_max = input_scale.max()
    weight_scale_2 = scale2.max()
    return Nvfp4Linear(
        weight=weight,
        weight_scale_swizzled=swizzle_block_scales(scale).view(torch.float8_e4m3fn),
        alpha=(input_scale_max * weight_scale_2).to(device),
        input_scale_inv=(1.0 / input_scale_max).to(device),
        out_features=n,
        in_features=k,
    )
