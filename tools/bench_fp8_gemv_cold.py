#!/usr/bin/env python3
"""Cold-L2 FP8 GEMV benchmark: rotate weights across 8 copies (~640 MB)."""
import itertools
import sys
import time

import torch

sys.path.insert(0, "/workspace/sharar")
from sharar.kernels.fp8_gemv import fp8_gemv, fp8_gemv_fma, fp8_gemv_splitk  # noqa: E402
from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda  # noqa: E402

COPIES = 8


def timeit(fn, iters=32):
    fn(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fn(i % COPIES)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
alpha = torch.tensor(1e-3, device="cuda")
for n, k in ((16384, 5120), (14336, 5120), (5120, 6144), (10240, 5120), (6144, 5120)):
    ws = [(torch.randn(n, k, device="cuda") * 40).to(torch.float8_e4m3fn) for _ in range(COPIES)]
    x = (torch.randn(m, k, device="cuda") * 40).to(torch.float8_e4m3fn)
    gb = n * k / 1e9
    ref = torch._scaled_mm(x, ws[0].t(), scale_a=alpha.reshape(1, 1), scale_b=alpha.reshape(1, 1), out_dtype=torch.bfloat16)
    t = timeit(lambda i: torch._scaled_mm(x, ws[i].t(), scale_a=alpha.reshape(1, 1), scale_b=alpha.reshape(1, 1), out_dtype=torch.bfloat16))
    print(f"N={n} K={k} M={m}: _scaled_mm {gb / t:.0f} GB/s")
    results = []
    out = fp8_gemv_cuda(x, ws[0], alpha * alpha)
    ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2)
    t = timeit(lambda i: fp8_gemv_cuda(x, ws[i], alpha * alpha))
    results.append((gb / t, f"CUDA warp-per-row port {'ok' if ok else 'WRONG'} maxdiff {(out.float() - ref.float()).abs().max().item():.3f}"))
    for bn, bk, nw in ((32, 256, 4), (64, 512, 8), (32, 512, 4), (64, 256, 4)):
        t = timeit(lambda i: fp8_gemv(x, ws[i], alpha * alpha, bn, bk, nw))
        results.append((gb / t, f"dot bn={bn} bk={bk} w={nw}"))
    for bn, bk, nw, st in ((32, 1024, 8, 4), (64, 1024, 8, 4), (32, 512, 4, 3)):
        t = timeit(lambda i: fp8_gemv_fma(x, ws[i], alpha * alpha, bn, bk, nw, st))
        results.append((gb / t, f"fma bn={bn} bk={bk} w={nw} st={st}"))
    for bn, bk, nw, sp, ev in itertools.product((32, 64), (256,), (4,), (2, 4), (True, False)):
        if k % (sp * bk):
            continue
        out = fp8_gemv_splitk(x, ws[0], alpha * alpha, bn, bk, nw, sp, ev)
        ok = torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-2)
        t = timeit(lambda i: fp8_gemv_splitk(x, ws[i], alpha * alpha, bn, bk, nw, sp, ev))
        results.append((gb / t, f"splitk bn={bn} bk={bk} w={nw} splits={sp} evict={ev} {'ok' if ok else 'WRONG'}"))
    results.sort(reverse=True)
    for r in results[:6]:
        print(f"   {r[0]:4.0f} GB/s  {r[1]}")
