"""Sharar on Nemotron-3.5-Super: the 3-bit decode GEMM (kernels/w3.py) against the FP8 path `ops.fp8_linear` takes
without a copy (fp8_gemv at M <= 4, the FP8 GEMM above), on the real projection shapes, M = 1..8 and 16.

    flock $SHARAR_RESULTS/.gpu.lock python profiling/nemotron_w3_kernels.py [--rows 1,2,3,4,5,6,8,16]

Random weights (the speed does not depend on the values); each time is the median per launch of a CUDA graph of ten
back-to-back launches (profiling/w3_bench.time_us), each launch reading the next of enough copies of the weights to
span 512 MB, so no launch finds its weights in the L2 (a decode step reads every projection once). One JSON line per
(shape, M): microseconds and the achieved weight bandwidth of each.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3_bench import time_us  # noqa: E402

SHAPES = {"mamba in_proj": (18560, 4096), "mamba out_proj": (4096, 8192), "attention qkv": (4608, 4096),
          "attention o": (4096, 4096), "lm_head": (131072, 4096)}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,2,3,4,5,6,8,16")
    args = ap.parse_args()
    from sharar.engine import config
    from sharar.kernels import ops, w3
    with config.using(config.current().replace(w3_decode=True)):
        for label, (n, k) in SHAPES.items():
            weight = (torch.randn(n, k, device="cuda") * 0.02)
            scale = torch.tensor(weight.abs().max().item() / 448.0, device="cuda")
            copies = max(1, min(16, -(-512 * 2**20 // (n * k))))  # FP8 copies spanning 512 MB (the W3 ones 41% of it)
            lins = []
            for _ in range(copies):
                lin = SimpleNamespace(weight=(weight / scale).to(torch.float8_e4m3fn), weight_scale=scale.reshape(1),
                                      input_scale=torch.tensor([0.01], device="cuda"), out_features=n, w3=None)
                lin.alpha = (lin.weight_scale * lin.input_scale).reshape(1)
                lins.append(lin)
            packed = w3.pack(weight)
            packs = [packed] + [replace(packed, planes=packed.planes.clone(), scales=packed.scales.clone(),
                                        row_scale=packed.row_scale.clone())
                                for _ in range(max(0, min(15, -(-512 * 2**20 // packed.bytes) - 1)))]
            del weight
            for m in (int(x) for x in args.rows.split(",")):
                x_q = (torch.randn(m, k, device="cuda")).to(torch.float8_e4m3fn)
                turn = {"fp8": 0, "w3": 0}

                def run_fp8():
                    turn["fp8"] += 1
                    return ops.fp8_linear(None, lins[turn["fp8"] % len(lins)], x_q=x_q)

                def run_w3():
                    turn["w3"] += 1
                    return w3.w3_linear(x_q, packs[turn["w3"] % len(packs)], x_scale=lins[0].input_scale)
                fp8, three = time_us(run_fp8), time_us(run_w3)
                print(json.dumps({"engine": "Sharar", "machine": "spark-9931", "shape": label, "n": n, "k": k, "m": m,
                                  "fp8_us": round(fp8, 1), "w3_us": round(three, 1), "speedup": round(fp8 / three, 2),
                                  "fp8_gbs": round(n * k / fp8 / 1e3, 1), "w3_gbs": round(packed.bytes / three / 1e3, 1)}), flush=True)
            del lins, packs, packed
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
