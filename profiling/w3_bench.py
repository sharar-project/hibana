"""3-bit decode GEMM: correctness against the dequantized reference and bandwidth per target shape.

Runs from the Python environment (.venv) with the GPU idle. For each (N, K) of the target's
decode projections (plus the lm_head and the draft's q/o) and M in (16, 32, 64)
it packs a random matrix, checks the kernel against an FP32 reference on the
same E4M3 activations, sweeps the split-K factor and reports the achieved
bandwidth (bytes of packed weights per launch over the median time) next to
the times a serving profile measured for the same shapes in their checkpoint format.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from hibana.kernels import w3

SHAPES = {  # (N, K): (name, profile median us at M=16, at M=64)
    (34816, 5120): ("mlp gate_up (NVFP4 base)", 442, 472),
    (5120, 17408): ("mlp down (NVFP4 base)", 234, 250),
    (10240, 5120): ("gdn qkv (FP8 base)", 242, 283),
    (6144, 5120): ("gdn z / attn o_proj (FP8 base)", 159, 183),
    (5120, 6144): ("gdn out (FP8 base)", 151, 162),
    (14336, 5120): ("attention qkv (FP8 base)", 338, 359),
    (248320, 5120): ("lm_head (NVFP4 base)", None, None),
    (4096, 5120): ("draft q_proj (FP8 base)", None, None),
    (5120, 4096): ("draft o_proj (FP8 base)", None, None),
}


def time_us(fn, iters=15, launches=10):
    """Median per-launch time of `launches` back-to-back launches replayed as a CUDA graph (no host gaps, as in serving)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(launches):
                fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); graph.replay(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / launches)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("/proof/w3-bench.json"))
    ap.add_argument("--rows", default="16,32,64")
    ap.add_argument("--codebook", default=w3.DEFAULT_CODEBOOK)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    results = {"codebook": args.codebook, "shapes": {}}
    for (n, k), (name, us16, us64) in SHAPES.items():
        w = torch.randn(n, k, device=dev) * 0.02
        packed = w3.pack(w, args.codebook)
        ref = w3.dequantize(packed)
        # rotate through enough copies that back-to-back launches cannot be served from L2 (weights stream from DRAM as in serving)
        copies = [packed] + [w3.pack(torch.randn(n, k, device=dev) * 0.02, args.codebook) for _ in range(max(0, -(-(160 << 20) // packed.bytes) - 1))]
        entry = {"name": name, "bytes": packed.bytes, "quant_rel_error": ((ref - w).norm() / w.norm()).item(), "M": {}}
        del w
        for m in (int(v) for v in args.rows.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            y_exact = x.float() @ ref.t()
            x_q, xs = w3.quant_groups(x)
            y_q = (x_q.float() * xs.repeat_interleave(32, dim=1)) @ ref.t()  # what the kernel should compute exactly
            y = w3.w3_linear(x, packed)
            err_kernel = ((y.float() - y_q).norm() / y_q.norm()).item()
            err_act = ((y_q - y_exact).norm() / y_exact.norm()).item()
            # activation quantization error on rows with outlier channels (per-row scale versus per-group scale)
            xo = x.clone(); xo[:, torch.randperm(k, device=dev)[:8]] *= 200
            yo = xo.float() @ ref.t()
            xr, sr = w3.quant_rows(xo); xg, sg = w3.quant_groups(xo)
            err_row_outlier = (((xr.float() * sr[:, None]) @ ref.t() - yo).norm() / yo.norm()).item()
            err_group_outlier = (((xg.float() * sg.repeat_interleave(32, dim=1)) @ ref.t() - yo).norm() / yo.norm()).item()
            x_scale = torch.tensor(x.abs().max().item() / 448.0, device=dev)
            x8 = (x.float() / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            y8 = w3.w3_linear(x8, packed, x_scale=x_scale)
            err8 = ((y8.float() - (x8.float() * x_scale) @ ref.t()).norm() / y_exact.norm()).item()
            assert err_kernel < 4e-3 and err8 < 4e-3, (name, m, err_kernel, err8)
            sweep = {}
            chunks = k // 128
            for splits in (1, 2, 4, 8):
                if chunks % splits:
                    continue
                turn = [0]

                def launch(splits=splits):
                    w3.w3_linear(x8, copies[turn[0] % len(copies)], x_scale=x_scale, splits=splits)
                    turn[0] += 1
                sweep[splits] = time_us(launch, launches=max(10, len(copies)))
            best_splits = min(sweep, key=sweep.get)
            t = sweep[best_splits]
            default = w3.splits_for(n, k, m)
            entry["M"][m] = {"weight_copies": len(copies), "kernel_rel_error": err_kernel, "kernel_rel_error_static_fp8": err8, "activation_quant_rel_error": err_act,
                             "activation_quant_rel_error_outliers": {"per_row": err_row_outlier, "per_group": err_group_outlier},
                             "us_by_splits": sweep, "best_splits": best_splits, "default_splits": default, "us": t,
                             "us_default_splits": sweep.get(default), "GBs": packed.bytes / t / 1e3, "pct_of_239": 100 * packed.bytes / t / 1e3 / 239,
                             "today_us": us16 if m == 16 else (us64 if m == 64 else None)}
            print(f"{name:34s} N={n:6d} K={k:5d} M={m:2d}: {t:7.1f} us best splits {best_splits} (default {default}: {sweep.get(default, float('nan')):7.1f}) "
                  f"{packed.bytes / t / 1e3:6.1f} GB/s  kernel err {err_kernel:.4f} act-quant err {err_act:.4f} (outlier rows: per-row {err_row_outlier:.3f}, per-group {err_group_outlier:.3f})  sweep {sweep}", flush=True)
        results["shapes"][f"{n}x{k}"] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
