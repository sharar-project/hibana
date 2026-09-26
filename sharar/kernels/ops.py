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
from sharar.kernels import stable, verify_gemm, w3

# cuBLAS FP8 GEMM reaches ~60% of DRAM bandwidth at M<=4 on SM121 (profile in
# docs/phase2.md); the streaming GEMV is used up to this many rows.
FP8_GEMV_MAX_M = 4

# CUTLASS preserves the pinned target's arithmetic. Faster backend candidates
# must also preserve full target outputs before replacing this dispatch.
FP4_GEMM_BACKEND = "cutlass"
FP4_QUANT_BACKEND = "cuda"

# Validated 16-row target shapes only. Benchmarks may disable this before
# graph capture to retain the original CUTLASS/cuBLAS arithmetic oracle.
VERIFY_GEMM_ENABLED = True


def fp4_backend(m: int, n: int, k: int) -> str:
    return FP4_GEMM_BACKEND


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


def w3_decode(lin, m: int) -> bool:
    """True when this call runs the 3-bit decode copy instead of the NVFP4/FP8 weights."""
    return lin.w3 is not None and m <= w3.MAX_ROWS and w3.enabled()


def nvfp4_linear(x: torch.Tensor, lin: Nvfp4Linear, prequantized: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    if prequantized is None and x.dtype == torch.bfloat16 and w3_decode(lin, x.shape[0]):
        return w3.w3_linear(x, lin.w3)
    if prequantized is None:
        x_fp4, x_sf = nvfp4_quantize(x, lin.input_scale_inv)
        m = x.shape[0]
        out_dtype = x.dtype
    else:
        x_fp4, x_sf = prequantized
        m = x_fp4.shape[0]
        out_dtype = torch.bfloat16
    if (VERIFY_GEMM_ENABLED and FP4_GEMM_BACKEND == "cutlass" and out_dtype == torch.bfloat16
            and verify_gemm.is_mlp(m, lin.out_features, lin.in_features)):
        return verify_gemm.fp4_mlp(x_fp4, x_sf, lin)
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
    if w3_decode(lin, m):
        return w3.w3_linear(x_q, lin.w3, x_scale=lin.input_scale).view(*lead, lin.out_features)
    if stable.enabled():
        rows = stable.padded_rows(m, (lin.out_features, lin.weight.shape[1]))
        if rows != m:  # keep cuBLASLt on its wide-M algorithm; the padded rows are discarded
            padded = torch.zeros((rows, x_q.shape[1]), dtype=x_q.dtype, device=x_q.device)
            padded[:m] = x_q
            return fp8_linear(None, lin, x_q=padded)[:m].view(*lead, lin.out_features)
    if m <= FP8_GEMV_MAX_M:
        out = fp8_gemv_cuda(x_q, lin.weight, lin.alpha)
    elif VERIFY_GEMM_ENABLED and verify_gemm.is_attention_qkv(m, lin.out_features, lin.weight.shape[1]):
        out = flashinfer.bmm_fp8(x_q[None], lin.weight.t()[None], lin.input_scale,
                                lin.weight_scale, torch.bfloat16, backend="cublas")[0]
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
    """Causal prefill over the new BF16 K/V, with optional prefix merging."""

    def __init__(self, device: torch.device, num_qo_heads: int, num_kv_heads: int, head_dim: int):
        self._workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self._workspace, "NHD")
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
        self._planner = AttentionPlanner(AttentionGeometry(num_qo_heads, num_kv_heads, head_dim))
        self._planned: tuple | None = None

    def plan(self, seq_lens: list[int], device: torch.device) -> None:
        key = tuple(seq_lens)
        if self._planned == key:
            return
        from sharar.kernels.attention_plan import host_indptr
        self._planner.ragged_prefill(self._wrapper, host_indptr(seq_lens))  # host: no sync in plan()
        self._planned = key

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = q.shape[0]
        return self._wrapper.run(
            q.view(t, self.num_qo_heads, self.head_dim),
            k.view(t, self.num_kv_heads, self.head_dim),
            v.view(t, self.num_kv_heads, self.head_dim),
        ).view(t, self.num_qo_heads * self.head_dim)

    def run_return_lse(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self._wrapper.run(q, k, v, return_lse=True)
