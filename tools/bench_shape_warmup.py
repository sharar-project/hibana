#!/usr/bin/env python3
"""First-call cost of a new activation row count M per mm_fp4 backend and for
the FP8 GEMM / Triton kernels: is there a per-shape compile or plan step?"""
import sys
import time

import torch

sys.path.insert(0, "/workspace/sharar")
from flashinfer import mm_fp4  # noqa: E402

from sharar.kernels import ops  # noqa: E402
from sharar.kernels.fused_norm import gemma_rmsnorm  # noqa: E402
from sharar.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize  # noqa: E402
from sharar.weights.modelopt import Nvfp4Linear, swizzle_block_scales  # noqa: E402


def t(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


n, k = 34816, 5120
w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
sc = (torch.rand(n, k // 16, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn)
lin = Nvfp4Linear(weight=w, weight_scale_swizzled=swizzle_block_scales(sc).view(torch.float8_e4m3fn), alpha=torch.tensor(1e-4, device="cuda"), input_scale_inv=torch.tensor(700.0, device="cuda"), out_features=n, in_features=k)
for backend in ("cutlass", "cudnn", "b12x"):
    for m in (129, 130, 777, 1001, 1002):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        xq, sf = nvfp4_quantize(x, lin.input_scale_inv)
        f = lambda: mm_fp4(xq, lin.weight.T, sf, lin.weight_scale_swizzled.T, lin.alpha, torch.bfloat16, backend=backend)  # noqa: E731
        first, second = t(f), t(f)
        print(f"mm_fp4 {backend:8s} M={m:5d}: first {first:7.1f} ms  second {second:6.2f} ms")
wf = (torch.randn(14336, 5120, device="cuda") * 40).to(torch.float8_e4m3fn)
a = torch.tensor(0.2, device="cuda")
for m in (129, 130, 777, 1001):
    xq = (torch.randn(m, 5120, device="cuda") * 40).to(torch.float8_e4m3fn)
    f = lambda: torch._scaled_mm(xq, wf.t(), scale_a=a.reshape(1, 1), scale_b=a.reshape(1, 1), out_dtype=torch.bfloat16)  # noqa: E731
    print(f"_scaled_mm M={m:5d}: first {t(f):7.1f} ms  second {t(f):6.2f} ms")
wt = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
for m in (129, 130, 777):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    print(f"triton norm M={m}: first {t(lambda: gemma_rmsnorm(x, wt, 1e-6)):6.1f} ms second {t(lambda: gemma_rmsnorm(x, wt, 1e-6)):5.2f} ms; "
          f"nvfp4_quant first {t(lambda: nvfp4_quantize(x, lin.input_scale_inv)):6.1f} second {t(lambda: nvfp4_quantize(x, lin.input_scale_inv)):5.2f}")
