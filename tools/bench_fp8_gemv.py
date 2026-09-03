#!/usr/bin/env python3
"""Microbenchmark FP8 GEMV configurations on the real decode shapes (no model)."""
import itertools
import sys
import time

import torch
import triton

sys.path.insert(0, "/workspace/sharar")
from sharar.kernels.fp8_gemv import _fp8_gemv_kernel, fp8_gemv_fma  # noqa: E402

SHAPES = [(16384, 5120), (14336, 5120), (5120, 6144), (5120, 6144)]  # qkvz, qkv, o/out proj


def run(x, w, alpha, bn, bk, warps, m_pad):
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device="cuda")
    _fp8_gemv_kernel[(triton.cdiv(w.shape[0], bn),)](
        x, w, y, alpha, x.shape[0], w.shape[0], w.shape[1], x.stride(0), w.stride(0), y.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, M_PAD=m_pad, num_warps=warps,
    )
    return y


def timeit(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
alpha = torch.tensor(1e-3, device="cuda")
for n, k in SHAPES[:3]:
    w = (torch.randn(n, k, device="cuda") * 40).to(torch.float8_e4m3fn)
    x = (torch.randn(m, k, device="cuda") * 40).to(torch.float8_e4m3fn)
    gb = n * k / 1e9
    t = timeit(lambda: torch._scaled_mm(x, w.t(), scale_a=alpha.reshape(1, 1), scale_b=alpha.reshape(1, 1), out_dtype=torch.bfloat16))
    print(f"N={n} K={k} M={m}: _scaled_mm {t * 1e6:.0f} us = {gb / t:.0f} GB/s")
    best = None
    for bn, bk, warps, mp in itertools.product((32, 64, 128), (256, 512, 1024), (4, 8), (16,)):
        if k % bk:
            continue
        try:
            t = timeit(lambda: run(x, w, alpha, bn, bk, warps, mp))
        except Exception as e:  # noqa: BLE001
            print(f"   bn={bn} bk={bk} w={warps}: failed {type(e).__name__}")
            continue
        if best is None or t < best[0]:
            best = (t, bn, bk, warps)
        print(f"   bn={bn:3d} bk={bk:4d} warps={warps}: {t * 1e6:6.0f} us = {gb / t:4.0f} GB/s")
    print(f"   best dot: bn={best[1]} bk={best[2]} warps={best[3]} -> {gb / best[0]:.0f} GB/s")
    ref = torch._scaled_mm(x, w.t(), scale_a=alpha.reshape(1, 1), scale_b=alpha.reshape(1, 1), out_dtype=torch.bfloat16)
    best = None
    for bn, bk, warps, stages in itertools.product((16, 32, 64), (256, 512, 1024), (4, 8), (2, 3, 4)):
        try:
            t = timeit(lambda: fp8_gemv_fma(x, w, alpha * alpha, bn, bk, warps, stages))
            out = fp8_gemv_fma(x, w, alpha * alpha, bn, bk, warps, stages)
            ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2)
        except Exception as e:  # noqa: BLE001
            print(f"   fma bn={bn} bk={bk} w={warps} st={stages}: failed {type(e).__name__}")
            continue
        if best is None or t < best[0]:
            best = (t, bn, bk, warps, stages)
        print(f"   fma bn={bn:3d} bk={bk:4d} warps={warps} stages={stages}: {t * 1e6:6.0f} us = {gb / t:4.0f} GB/s {'ok' if ok else 'WRONG'}")
    print(f"   best fma: bn={best[1]} bk={best[2]} warps={best[3]} stages={best[4]} -> {gb / best[0]:.0f} GB/s")
