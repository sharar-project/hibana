"""F2 screen 2: Qwen3.8-Flash-Next's BF16 trunk through Sharar's FP8 GEMV and 3-bit kernels on the GB10.

Times every dense projection shape of the Flash-Next trunk (hidden 2,560) at 1, 4, 16 and
64 rows in BF16 (cuBLAS, the shipped precision), FP8 (Sharar's small-M GEMV up to 4 rows,
cuBLASLt beyond) and W3 (sharar/kernels/w3.py, up to 64 rows), inside CUDA graphs with
weight copies rotated past the 24 MiB L2, and sums them into a per-token trunk cost.

    python bench/flashnext_trunk_screen.py --rows 1 4 16 64
"""
from __future__ import annotations

import argparse
import json

import torch

from sharar.kernels import w3
from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda
from sharar.kernels.fp8_quant import quant_fp8

# (name, N, K, count per token, W3 eligible)  K must be a multiple of 128 for W3
SHAPES = [
    ("gdn.in_qkv|z", 16384, 2560, 36),
    ("gdn.out_proj", 2560, 6144, 36),
    ("attn.qkv", 13312, 2560, 12),
    ("attn.o_proj", 2560, 6144, 12),
    ("attn.indexer", 640, 2560, 12),
    ("hc.mix_up", 10240, 320, 96),
    ("hc.mix_down", 320, 10240, 96),
    ("shared.gate_up", 1280, 2560, 48),
    ("shared.down", 2560, 640, 48),
    ("router", 512, 2560, 48),
    ("lm_head", 248320, 2560, 1),
]
L2 = 24 << 20


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def copies_for(nbytes: int) -> int:
    return max(1, min(16, (3 * L2) // max(1, nbytes) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    results = []
    for name, n, k, count in SHAPES:
        bf16_bytes, fp8_bytes, w3_bytes = n * k * 2, n * k, int(n * k * 3.25 / 8)
        row = {"name": name, "n": n, "k": k, "count": count, "bf16": {}, "fp8": {}, "w3": {}}
        w = torch.randn((n, k), device=dev, dtype=torch.bfloat16) * 0.02
        c_bf16 = copies_for(bf16_bytes)
        ws = [w.clone() for _ in range(c_bf16)]
        c_fp8 = copies_for(fp8_bytes)
        w_fp8 = [(w.float().clamp(-1, 1) * 448).to(torch.float8_e4m3fn) for _ in range(c_fp8)]
        alpha = torch.tensor([1.0 / 448], device=dev)
        w3_ok = k % 128 == 0 and n % 128 == 0
        if w3_ok:
            c_w3 = copies_for(w3_bytes)
            packed = [w3.pack(w.float()) for _ in range(c_w3)]
        for m in args.rows:
            x = torch.randn((m, k), device=dev, dtype=torch.bfloat16)
            # BF16 cuBLAS
            def f_bf16(i=[0]):
                for c in range(c_bf16):
                    torch.matmul(x, ws[c].T)
            ms = timed(f_bf16) / c_bf16
            row["bf16"][m] = ms
            # FP8: Sharar's dispatch (GEMV for m <= 4, cuBLASLt scaled_mm beyond)
            x_q = (x.float().clamp(-1, 1) * 448).to(torch.float8_e4m3fn)
            scale = torch.tensor(1.0 / 448, device=dev)
            if m <= 4 and k % 16 == 0:
                def f_fp8():
                    for c in range(c_fp8):
                        fp8_gemv_cuda(x_q, w_fp8[c], alpha)
            else:
                def f_fp8():
                    for c in range(c_fp8):
                        torch._scaled_mm(x_q, w_fp8[c].t(), scale_a=scale.reshape(1, 1), scale_b=scale.reshape(1, 1), out_dtype=torch.bfloat16)
            try:
                row["fp8"][m] = timed(f_fp8) / c_fp8
            except Exception as exc:  # noqa: BLE001
                row["fp8"][m] = None
                print(f"  fp8 {name} m={m}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            if w3_ok and m <= w3.MAX_ROWS:
                def f_w3():
                    for c in range(c_w3):
                        w3.w3_linear(x, packed[c])
                try:
                    row["w3"][m] = timed(f_w3) / c_w3
                except Exception as exc:  # noqa: BLE001
                    row["w3"][m] = None
                    print(f"  w3 {name} m={m}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        results.append(row)
        line = f"{name:15} N={n:6} K={k:5} x{count:3}"
        for prec, nbytes in (("bf16", bf16_bytes), ("fp8", fp8_bytes), ("w3", w3_bytes)):
            for m in args.rows:
                ms = row[prec].get(m)
                if ms:
                    line += f" | {prec} m{m}: {ms*1000:6.0f}us {nbytes/ms/1e6:4.0f}GB/s"
        print(line, flush=True)
        del ws, w_fp8
        if w3_ok:
            del packed
        torch.cuda.empty_cache()
    print("\nper-token trunk cost (sum over layers), ms:")
    for m in args.rows:
        totals = {}
        for prec in ("bf16", "fp8", "w3"):
            t = 0.0; missing = []
            for r in results:
                ms = r[prec].get(m)
                if ms is None:
                    missing.append(r["name"]); ms = r["fp8"].get(m) or r["bf16"][m]  # fall back to the next precision
                t += ms * r["count"]
            totals[prec] = (t, missing)
        print(f"  m={m:2}: " + "  ".join(f"{p} {t:6.2f} ms" + (f" (no {p} for {','.join(mi)})" if mi else "") for p, (t, mi) in totals.items()))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
