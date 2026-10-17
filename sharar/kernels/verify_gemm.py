"""Exact-output GEMM specializations for GB10's 16-row verifier.

Only the attention QKV and the two target MLP shapes are selected. Other
batch sizes and prefill shapes retain the ordinary dispatch. The FP4 path
uses the pinned FlashInfer 0.6.17 CuTe compiler with a larger K tile; packed
weights/scales are borrowed, and each call owns its BF16 output. Compilation
is cached before CUDA graph capture by the normal executor warmup.
"""

import torch


def is_attention_qkv(m: int, n: int, k: int) -> bool:
    return (m, n, k) == (16, 14336, 5120)


def is_mlp(m: int, n: int, k: int) -> bool:
    return m == 16 and (n, k) in ((34816, 5120), (5120, 17408))


_compiled = {}


def fp4_mlp(q, sf, lin):
    """[16,K/2] packed activations and swizzled scales -> BF16 [16,N]."""
    import cutlass
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import DenseGemmKernel

    m, n, k = len(q), lin.out_features, lin.in_features
    if not is_mlp(m, n, k):
        raise ValueError('unsupported 16-row MLP GEMM shape')
    sf_m, sf_n, sf_k = (m + 127) // 128, (n + 127) // 128, (k // 16 + 3) // 4
    out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    tile = (64, 128)
    key = (q.device.index, tile, 512, sf_m, sf_n, sf_k)
    compiled, _ = _compile_block_scaled_gemm(
        _compiled, key, lambda: DenseGemmKernel(16, tile, (1, 1), tile_k=512),
        ab_cutlass_dtype=cutlass.Uint8, sf_dtype=cutlass.Float8E4M3FN,
        c_cutlass_dtype=cutlass.BFloat16, ab_assumed_align=32,
        cluster_shape_mn=(1, 1), swap_ab=False,
        sf_m=sf_m, sf_n=sf_n, sf_k=sf_k, batch_size=1)
    compiled(q, lin.weight, out, sf_m, sf_n, sf_k, sf.data_ptr(),
             lin.weight_scale_swizzled.data_ptr(), lin.alpha.reshape(1))
    return out
