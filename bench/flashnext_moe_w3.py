# SPDX-License-Identifier: Apache-2.0
"""3-bit routed experts (#73) against FlashInfer's cutlass_fused_moe on real Flash-Next expert weights: the go/no-go
microbenchmark (go only if the 3-bit path is faster at M = 5, the MTP k = 4 verify).

Loads `--layers` MoE layers of the checkpoint (NVFP4, the CUTLASS arm) and fits each layer's experts into the 3-bit
format from them (kernels/w3_moe.calibrate, as bench/flashnext_w3_experts.py writes the checkpoint).

Routing is synthetic, matched to the distinct experts per layer that recorded traffic touched in the small-M study
(#37: 10.0, 16.4, 25.1, 28.3, 40.1 at M = 1, 2, 4, 5, 8; M = 16 extrapolated at 3.9 new experts a row): row 0 takes
ten random experts, every further row a share of new ones and the rest from experts earlier rows already chose.
`--routing uniform` draws every row's ten experts independently instead (the most distinct experts, no overlap);
calls of more than 16 rows (prefill chunks, wide verification) always do.
Inputs are Gaussian rows scaled to the layer's calibrated fc1 input range.

For each M, `--windows` routings per layer run through all loaded layers inside one CUDA graph per window (each
layer's own weights, so an expert recurs only after the other layers' experts have gone through the 24 MB L2);
the graphs replay round-robin. Reported per layer: time, distinct experts, effective GB/s from the bytes of the
distinct experts (NVFP4 weights + block scales: 2.76 MB an expert; 3-bit codes + group scales + row scales:
2.01 MB). Both arms produce the routed sum only (no shared expert, no combine); the 3-bit arm includes its
activation quantization, routing plan and weighted sum, CUTLASS its own.

Numerics per M on two layers: the 3-bit kernel against `w3_moe.reference` (the same 3-bit weights dequantized,
FP32 matmuls in torch; the kernel's activation rounding), and both against CUTLASS (NVFP4 activations); CUTLASS
against the same reference with NVFP4 activations checks that both arms hold the same weights.

    SHARAR_TARGET=flashnext python3 -m bench.flashnext_moe_w3 --out moe_w3.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from sharar.kernels import w3_moe

NVFP4_BYTES = 2 * 640 * 1280 + 2560 * 320 + 2 * 640 * 160 + 2560 * 40  # w13 + w2 and their E4M3 block scales
DISTINCT = {1: 10.0, 2: 16.4, 4: 25.1, 5: 28.3, 8: 40.1}  # recorded traffic, #37 (docs/flashnext.md on feat/moe-small-m)
NEW_PER_ROW_PAST_8 = 3.9


def target_distinct(m: int) -> float:
    if m in DISTINCT:
        return DISTINCT[m]
    lo = max(k for k in DISTINCT if k < m)
    if m > max(DISTINCT):
        return DISTINCT[lo] + NEW_PER_ROW_PAST_8 * (m - lo)
    hi = min(k for k in DISTINCT if k > m)
    return DISTINCT[lo] + (DISTINCT[hi] - DISTINCT[lo]) * (m - lo) / (hi - lo)


def routing(m: int, kind: str, gen: torch.Generator, experts: int = 512, topk: int = 10) -> torch.Tensor:
    """ids [m, topk] int32: distinct per row; `matched` hits target_distinct(m) on average."""
    if kind == "uniform" or m > 16:  # prefill-sized calls touch most experts either way
        return torch.stack([torch.randperm(experts, generator=gen)[:topk] for _ in range(m)]).to(torch.int32)
    goal = target_distinct(m)
    d = int(goal) + int(torch.rand((), generator=gen).item() < goal - int(goal))
    extra = d - topk  # new experts spread over rows 1..m-1
    new = [topk] + [extra // (m - 1) + (1 if r - 1 < extra % (m - 1) else 0) for r in range(1, m)] if m > 1 else [topk]
    perm = torch.randperm(experts, generator=gen).tolist()
    union, rows = [], []
    for r in range(m):
        fresh = [perm.pop() for _ in range(new[r])]
        reuse = [union[i] for i in torch.randperm(len(union), generator=gen)[:topk - new[r]].tolist()]
        row = fresh + reuse
        rows.append([row[i] for i in torch.randperm(topk, generator=gen).tolist()])
        union += fresh
    return torch.tensor(rows, dtype=torch.int32)


def cutlass_routed(moe, x, ids, w):
    from flashinfer.fused_moe import cutlass_fused_moe
    m = x.shape[0]
    r = cutlass_fused_moe(input=x, token_selected_experts=ids, token_final_scales=w, fc1_expert_weights=moe.w13_long,
                          fc2_expert_weights=moe.w2_long, output_dtype=torch.bfloat16, quant_scales=moe.quant_scales,
                          tune_max_num_tokens=max(1, 1 << (m - 1).bit_length()))  # as MoeLayer.__call__ serves it
    return r[0] if isinstance(r, (list, tuple)) else r


def graph_us(steps, iters: int, calls: int) -> float:
    """steps: callables (one pass over the layers each), one graph per step replayed round-robin; us per call."""
    for s in steps:
        s()
    torch.cuda.synchronize()
    graphs = []
    for s in steps:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s()
        graphs.append(g)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        graphs[i % len(graphs)].replay()
    b.record()
    torch.cuda.synchronize()
    del graphs
    return 1000 * a.elapsed_time(b) / iters / calls


def quant_nvfp4(moe):
    """CUTLASS's activation rounding for w3_moe.reference: NVFP4 with the layer's static input scales, dequantized."""
    from sharar.kernels.nvfp4_quant import nvfp4_quantize
    from sharar.weights.modelopt import unpack_fp4, unswizzle_block_scales

    def quant(t, stage):
        gs = moe.a13 if stage == "x" else moe.a2
        q, sf = nvfp4_quantize(t.contiguous(), gs)
        k = t.shape[1]
        return unpack_fp4(q, unswizzle_block_scales(sf.reshape(-1).view(torch.float8_e4m3fn), t.shape[0], k // 16)) / gs
    return quant


def compare(a, b):
    a, b = a.float(), b.float()
    return {"rel_err": float((a - b).norm() / b.norm()), "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
            "max_abs": float((a - b).abs().max()), "ref_max": float(b.abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--first", type=int, default=0, help="first decoder layer loaded")
    ap.add_argument("--ms", default="1,2,5,8,16")
    ap.add_argument("--routing", choices=("matched", "uniform"), default="matched")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats per arm (arms interleaved); median reported")
    ap.add_argument("--skip-numerics", action="store_true")
    args = ap.parse_args()

    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.model.flashnext.moe import MoeLayer
    from sharar.target import FLASHNEXT
    from sharar.weights.safetensors import SnapshotWeights

    dev = torch.device("cuda")
    snap = FLASHNEXT.snapshot_dir()
    w = SnapshotWeights(snap)
    t0 = time.perf_counter()
    moes, experts, errors = [], [], []
    for i in range(args.first, args.first + args.layers):
        moe = MoeLayer(w, f"model.language_model.layers.{i}.mlp", dev, FLASHNEXT, TrunkLinear, "fp8")
        w.close()
        moes.append(moe)
        s13 = w3_moe.unswizzle_batched(moe.s13, 2 * moe.inter, moe.hidden // 16)
        s2 = w3_moe.unswizzle_batched(moe.s2, moe.hidden, moe.inter // 16)
        # g = input scale * global scale and a = 1 / input scale (moe.py): the weights' own global scales
        ex, report = w3_moe.calibrate(moe.w13, s13, moe.g1 * moe.a13, moe.w2, s2, moe.g2 * moe.a2)
        experts.append(ex)
        errors.append(report)
        del s13, s2
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    w3_bytes = experts[0].bytes / w3_moe.EXPERTS
    print(f"loaded and calibrated {args.layers} layers in {time.perf_counter() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB; bytes per expert NVFP4 {NVFP4_BYTES}, 3-bit {w3_bytes:.0f}", flush=True)
    w3_moe.w3._module()

    gen = torch.Generator().manual_seed(0)
    out = {"layers": args.layers, "first": args.first, "routing": args.routing, "windows": args.windows, "iters": args.iters,
           "nvfp4_bytes_per_expert": NVFP4_BYTES, "w3_bytes_per_expert": w3_bytes, "weight_error": errors,
           "gpu": torch.cuda.get_device_name(), "timing": [], "numerics": []}
    for m in [int(v) for v in args.ms.split(",")]:
        xs, routes = [], []
        for _ in range(args.windows):
            xw, rw = [], []
            for moe in moes:
                amax = float(1.0 / moe.a13) * 6 * 448  # the calibrated fc1 input range
                xw.append((torch.randn((m, 2560), generator=gen) * amax / 8).to(dev, torch.bfloat16))
                ids = routing(m, args.routing, gen).to(dev)
                wt = torch.softmax(torch.randn((m, 10), generator=gen), -1).to(dev)
                rw.append((ids, wt))
            xs.append(xw)
            routes.append(rw)
        distinct = sum(int(torch.unique(ids).numel()) for rw in routes for ids, _ in rw) / (args.windows * len(moes))
        row = {"m": m, "distinct_experts_per_layer": round(distinct, 2)}
        steps = {
            "cutlass": [(lambda i=i: [cutlass_routed(moe, x, ids, wt) for moe, x, (ids, wt) in zip(moes, xs[i], routes[i])]) for i in range(args.windows)],
            "w3": [(lambda i=i: [w3_moe.w3_moe(ex, x, ids, wt) for ex, x, (ids, wt) in zip(experts, xs[i], routes[i])]) for i in range(args.windows)],
        }
        runs = {arm: [] for arm in steps}
        for _ in range(args.repeats):
            for arm, s in steps.items():
                runs[arm].append(graph_us(s, args.iters, len(moes)))
        for arm, per_expert in (("cutlass", NVFP4_BYTES), ("w3", w3_bytes)):
            us = sorted(runs[arm])[len(runs[arm]) // 2]
            row[f"{arm}_us"] = round(us, 1)
            row[f"{arm}_us_runs"] = [round(v, 1) for v in runs[arm]]
            row[f"{arm}_gbs"] = round(distinct * per_expert / (us * 1e3), 1)
        row["speedup"] = round(row["cutlass_us"] / row["w3_us"], 3)
        out["timing"].append(row)
        print(json.dumps(row), flush=True)
        if args.skip_numerics or m > 64:
            continue
        for li in sorted({0, len(moes) - 1}):
            x, (ids, wt) = xs[0][li], routes[0][li]
            r_w3 = w3_moe.w3_moe(experts[li], x, ids, wt)
            r_c = cutlass_routed(moes[li], x, ids, wt)
            r_ref = w3_moe.reference(experts[li], x, ids, wt)
            r_ref4 = w3_moe.reference(experts[li], x, ids, wt, quant_nvfp4(moes[li]))
            rep = {"m": m, "layer": args.first + li, "w3_vs_reference": compare(r_w3, r_ref), "w3_vs_cutlass": compare(r_w3, r_c),
                   "cutlass_vs_reference": compare(r_c, r_ref), "cutlass_vs_reference_nvfp4_acts": compare(r_c, r_ref4),
                   "reference_nvfp4_acts_vs_reference": compare(r_ref4, r_ref),
                   "w3_repeat_bit_equal": bool(torch.equal(r_w3, w3_moe.w3_moe(experts[li], x, ids, wt)))}
            out["numerics"].append(rep)
            print(json.dumps(rep), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
