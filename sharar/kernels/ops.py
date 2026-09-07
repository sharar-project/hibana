"""Kernel entry points: FlashInfer GEMM/quantization/attention plus torch fallbacks.

Every backend choice here is explicit. The measurement or reference behavior
that chose it is named in a comment.
"""

from __future__ import annotations

import torch
import flashinfer
from flashinfer import fp4_quantize as _fp4_quantize, mm_fp4 as _mm_fp4

from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda
from sharar.kernels.fp8_quant import quant_fp8
from sharar.kernels.nvfp4_quant import nvfp4_quantize
from sharar.weights.modelopt import FP8_MAX, Fp8Linear, Nvfp4Linear

# cuBLAS FP8 GEMM reaches ~60% of DRAM bandwidth at M<=4 on SM121 (profile in
# docs/phase2.md); the streaming GEMV is used up to this many rows.
FP8_GEMV_MAX_M = 4

# The reference resolves fp4_gemm_runner_backend=auto to flashinfer_cutlass on
# SM121 and quantizes activations with FlashInfer's "cuda" fp4_quantize. Here
# the backend is chosen per shape from tools/bench_gemm_backends.py (cold L2,
# GB10): b12x wins for the wide gate/up GEMM at prefill (+10%), cudnn for the
# down GEMM at prefill (+8%) and for small M; cutlass is the parity oracle.
FP4_GEMM_BACKEND = "cutlass"
FP4_QUANT_BACKEND = "cuda"
FP4_SMALL_M = 64  # cuDNN beats b12x up to 64 rows (gate/up at M=64: 0.51 vs 0.62 ms); b12x wins at prefill M


def fp4_backend(m: int, n: int, k: int) -> str:
    if m <= FP4_SMALL_M:
        return "cudnn"
    if n > k:  # gate/up: N=34816, K=5120
        return "b12x"
    return "cudnn"


def fp4_quantize(x: torch.Tensor, input_scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's quantizer; kept as the oracle for sharar.kernels.nvfp4_quant."""
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
        x_fp4, x_sf = nvfp4_quantize(x, lin.input_scale_inv)
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
        backend=fp4_backend(m, lin.out_features, lin.in_features),
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


def fp8_linear(x: torch.Tensor | None, lin: Fp8Linear, x_q: torch.Tensor | None = None) -> torch.Tensor:
    """Static per-tensor FP8 GEMM. Pass either bf16 `x` (quantized here with
    the layer's input scale) or an already quantized `x_q`."""
    if x_q is None:
        x2 = x.reshape(-1, x.shape[-1])
        x_q = quant_fp8(x2, lin.input_scale)
        lead = x.shape[:-1]
    else:
        lead = x_q.shape[:-1]
    m = x_q.shape[0]
    if m <= FP8_GEMV_MAX_M:
        out = fp8_gemv_cuda(x_q, lin.weight, lin.alpha)
    else:
        out = torch._scaled_mm(
            x_q,
            lin.weight.t(),
            scale_a=lin.input_scale.reshape(1, 1),
            scale_b=lin.weight_scale.reshape(1, 1),
            out_dtype=torch.bfloat16,
        )
    return out.view(*lead, lin.out_features)


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
        indptr = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int32)  # host: no sync in plan()
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
