"""Kernel entry points: FlashInfer GEMM/quantization/attention plus torch fallbacks.

Every backend choice here is explicit. The measurement or reference behavior
that chose it is named in a comment.
"""

from __future__ import annotations

import torch
import flashinfer
from flashinfer import fp4_quantize as _fp4_quantize, mm_fp4 as _mm_fp4

from sharar.weights.modelopt import FP8_MAX, Fp8Linear, Nvfp4Linear

# The reference resolves fp4_gemm_runner_backend=auto to flashinfer_cutlass on
# SM121 (fp4_utils.initialize_fp4_gemm_config) and quantizes activations with
# the "cuda" fp4_quantize backend (SM100 gets cute-dsl; SM121 does not).
FP4_GEMM_BACKEND = "cutlass"
FP4_QUANT_BACKEND = "cuda"


def fp4_quantize(x: torch.Tensor, input_scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _fp4_quantize(
        input=x,
        global_scale=input_scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
        is_sf_8x4_layout=False,
        backend=FP4_QUANT_BACKEND,
    )


def nvfp4_linear(x: torch.Tensor, lin: Nvfp4Linear, prequantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    if prequantized is None:
        x_fp4, x_sf = fp4_quantize(x, lin.input_scale_inv)
        m = x.shape[0]
        out_dtype = x.dtype
    else:
        x_fp4, x_sf = prequantized
        m = x_fp4.shape[0]
        out_dtype = torch.bfloat16
    out = _mm_fp4(
        x_fp4,
        lin.weight.T,
        x_sf,
        lin.weight_scale_swizzled.T,
        lin.alpha,
        out_dtype,
        backend=FP4_GEMM_BACKEND,
    )
    return out.view(m, lin.out_features)


def silu_mul_fp4_quantize(gate_up: torch.Tensor, input_scale_inv_1d: torch.Tensor, masked_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU(gate)*up + NVFP4 quantization, the reference's dense-MLP path.

    Reshapes copied from SGLang's Qwen2MoeMLP._silu_fp4_quant_fused, which
    verified the resulting layout bit-exact against fp4_quantize.
    """
    from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize

    y_fp4, y_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gate_up.unsqueeze(0), masked_m, input_scale_inv_1d
    )
    y_fp4 = y_fp4.squeeze(-1).view(torch.uint8)
    m_padded = y_sf.shape[2] * y_sf.shape[0] * y_sf.shape[3]
    y_sf = y_sf.view(torch.uint8).permute(2, 4, 0, 1, 3, 5).reshape(m_padded, -1)
    return y_fp4, y_sf


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    half = gate_up.shape[-1] // 2
    return torch.nn.functional.silu(gate_up[..., :half]) * gate_up[..., half:]


def fp8_linear(x: torch.Tensor, lin: Fp8Linear) -> torch.Tensor:
    """Static per-tensor FP8 GEMM. The reference uses cuBLAS-backed scaled GEMMs
    (flashinfer bmm_fp8 / torch._scaled_mm) with the same scales."""
    x2 = x.reshape(-1, x.shape[-1])
    x_q = (x2.float() / lin.input_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        x_q,
        lin.weight.t(),
        scale_a=lin.input_scale.reshape(1, 1),
        scale_b=lin.weight_scale.reshape(1, 1),
        out_dtype=x.dtype,
    )
    return out.view(*x.shape[:-1], lin.out_features)


class RaggedAttention:
    """FlashInfer causal ragged prefill (bf16 K/V), as the reference uses for
    prefill without a cached prefix."""

    def __init__(self, device: torch.device, num_qo_heads: int, num_kv_heads: int, head_dim: int):
        self._workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self._workspace, "NHD")
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._planned: tuple | None = None

    def plan(self, seq_lens: list[int], device: torch.device) -> None:
        key = tuple(seq_lens)
        if self._planned == key:
            return
        indptr = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
        self._wrapper.plan(
            indptr,
            indptr,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
            sm_scale=self.head_dim**-0.5,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        self._planned = key

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = q.shape[0]
        return self._wrapper.run(
            q.view(t, self.num_qo_heads, self.head_dim),
            k.view(t, self.num_kv_heads, self.head_dim),
            v.view(t, self.num_kv_heads, self.head_dim),
        ).view(t, self.num_qo_heads * self.head_dim)
