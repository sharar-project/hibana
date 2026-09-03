#!/usr/bin/env python3
"""Microbenchmark FlashInfer mm_fp4 backends and FP8 GEMM paths on model shapes.

Weights are rotated across 8 copies so L2 cannot serve them (cold DRAM stream).
"""
import sys
import time

import torch

sys.path.insert(0, "/workspace/sharar")
import flashinfer  # noqa: E402
from flashinfer import mm_fp4  # noqa: E402

from sharar.kernels import ops  # noqa: E402
from sharar.weights.modelopt import Nvfp4Linear, swizzle_block_scales  # noqa: E402

COPIES = 8


def timeit(fn, iters=32):
    fn(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fn(i % COPIES)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def make_nvfp4(n, k):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    sc = (torch.rand(n, k // 16, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn)
    return Nvfp4Linear(weight=w, weight_scale_swizzled=swizzle_block_scales(sc).view(torch.float8_e4m3fn), alpha=torch.tensor(1e-4, device="cuda"), input_scale_inv=torch.tensor(700.0, device="cuda"), out_features=n, in_features=k)


print("== NVFP4 mm_fp4 backends ==")
for n, k in ((34816, 5120), (5120, 17408), (248320, 5120)):
    lins = [make_nvfp4(n, k) for _ in range(COPIES if n < 100000 else 2)]
    gb = n * k / 2 / 1e9 + n * k / 16 / 1e9
    for m in (1, 4, 2048):
        if n > 100000 and m > 4:
            continue
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x_fp4, x_sf = ops.fp4_quantize(x, lins[0].input_scale_inv)
        for backend in ("cutlass", "cudnn", "b12x", "trtllm", "auto"):
            try:
                def fn(i, backend=backend):
                    lin = lins[i % len(lins)]
                    return mm_fp4(x_fp4, lin.weight.T, x_sf, lin.weight_scale_swizzled.T, lin.alpha, torch.bfloat16, backend=backend)
                ref = fn(0, "cutlass").float()
                out = fn(0).float()
                ok = torch.allclose(out, ref, rtol=5e-2, atol=1e-1)
                t = timeit(fn)
                flops = 2 * m * n * k
                print(f"N={n} K={k} M={m} {backend:8s}: {t * 1e6:8.0f} us  {gb / t:5.0f} GB/s  {flops / t / 1e12:6.1f} TFLOPS  {'ok' if ok else 'MISMATCH'}")
            except Exception as e:  # noqa: BLE001
                print(f"N={n} K={k} M={m} {backend:8s}: failed {type(e).__name__}: {str(e)[:80]}")

print("== FP8 GEMM paths ==")
has_bmm = hasattr(flashinfer, "bmm_fp8")
for n, k in ((16384, 5120), (14336, 5120), (5120, 6144)):
    ws = [(torch.randn(n, k, device="cuda") * 40).to(torch.float8_e4m3fn) for _ in range(COPIES)]
    a = torch.tensor(0.2, device="cuda")
    b = torch.tensor(1e-3, device="cuda")
    gb = n * k / 1e9
    for m in (1, 4, 2048):
        x = (torch.randn(m, k, device="cuda") * 40).to(torch.float8_e4m3fn)
        t = timeit(lambda i: torch._scaled_mm(x, ws[i].t(), scale_a=a.reshape(1, 1), scale_b=b.reshape(1, 1), out_dtype=torch.bfloat16))
        print(f"N={n} K={k} M={m} _scaled_mm : {t * 1e6:8.0f} us  {gb / t:5.0f} GB/s  {2 * m * n * k / t / 1e12:6.1f} TFLOPS")
        if has_bmm:
            try:
                t = timeit(lambda i: flashinfer.bmm_fp8(x.unsqueeze(0), ws[i].t().unsqueeze(0), a, b, torch.bfloat16))
                print(f"N={n} K={k} M={m} bmm_fp8    : {t * 1e6:8.0f} us  {gb / t:5.0f} GB/s  {2 * m * n * k / t / 1e12:6.1f} TFLOPS")
            except Exception as e:  # noqa: BLE001
                print(f"N={n} K={k} M={m} bmm_fp8    : failed {type(e).__name__}: {str(e)[:80]}")
        if m <= 4:
            from sharar.kernels.fp8_gemv import fp8_gemv
            t = timeit(lambda i: fp8_gemv(x, ws[i], a * b))
            print(f"N={n} K={k} M={m} fp8_gemv   : {t * 1e6:8.0f} us  {gb / t:5.0f} GB/s (cold L2)")
