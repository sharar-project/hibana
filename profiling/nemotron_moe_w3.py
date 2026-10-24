# SPDX-License-Identifier: Apache-2.0
"""Nemotron's routed experts: the 3-bit kernel (`kernels/w3_moe.py`, ReLU^2 geometry) against FlashInfer's
cutlass_fused_moe on the checkpoint's NVFP4 experts (`NemotronMoe.routed`), per token count, on real expert weights.

Loads `--blocks` MoE blocks (NVFP4, the CUTLASS arm) and fits each block's experts into the 3-bit format from them
(`w3_moe.calibrate_relu2`, the cheap fit: speed does not depend on the codes; `--w3 DIR` takes a W3 checkpoint's
experts instead). Routing: every row's 22 experts drawn independently, distinct within the row (this router's tokens
share few experts: 16 consecutive tokens' top 22 are 64-67% distinct), renormalized weights times 5. Inputs: Gaussian
latent rows scaled to the block's calibrated fc1 input range.

For each M, `--windows` routings per block run through all loaded blocks inside one CUDA graph per window (each block's
own weights, so an expert recurs only after the others have gone through the 24 MB L2); the graphs replay round-robin.
Reported per block: time, distinct experts, effective GB/s from the distinct experts' bytes (NVFP4 3.10 MB an expert,
3-bit 2.25 MB). Both arms produce the routed sum only (latent [M, 1024] in and out, no shared expert or fc2_latent).

Numerics per M <= 64 on one block: the 3-bit kernel against `w3_moe.reference` (the same 3-bit weights dequantized,
FP32 matmuls, the kernel's activation rounding) and against CUTLASS; both arms against the checkpoint's own math
(`exact`: the NVFP4 experts dequantized, FP32 activations, relu^2 in FP32), which neither arm computes (CUTLASS rounds
the activations to NVFP4, the 3-bit arm to E4M3 per 32 and the weights to 3 bits).

    python3 -m profiling.nemotron_moe_w3 --out nemotron_moe_w3.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from sharar.kernels import w3_moe

NVFP4_BYTES = 2 * 2688 * 512 + 2 * 2688 * 64  # up + down codes and their E4M3 block scales: 3,096,576
TOPK = 22


def graph_us(steps, iters: int, calls: int) -> float:
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


def compare(a, b):
    a, b = a.float(), b.float()
    return {"rel_err": round(float((a - b).norm() / b.norm()), 5),
            "cos": round(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)), 6)}


@torch.no_grad()
def exact(moe, x, ids, wt):
    """The routed sum (times routed_scaling_factor) from the dequantized NVFP4 experts with FP32 activations."""
    from sharar.model.nemotron import config as c
    from sharar.weights.modelopt import unpack_fp4
    s1 = w3_moe.unswizzle_batched(moe.s1, moe.inter, moe.latent // 16)
    s2 = w3_moe.unswizzle_batched(moe.s2, moe.latent, moe.inter // 16)
    xf = x.float()
    acc = torch.zeros(x.shape[0], moe.latent, device=x.device)
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        up = unpack_fp4(moe.w1[e], s1[e]) * moe.g1[e]
        down = unpack_fp4(moe.w2[e], s2[e]) * moe.g2[e]
        acc.index_add_(0, rows, (torch.relu(xf[rows] @ up.T).square() @ down.T) * wt[rows, slot, None] * c.ROUTED_SCALING_FACTOR)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--first", type=int, default=1, help="first MoE block (checkpoint index) loaded")
    ap.add_argument("--ms", default="1,2,4,8,16,64,512,2048")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--w3", type=Path, help="a W3 checkpoint with the experts (default: the cheap fit here)")
    ap.add_argument("--machine", default="spark-9931")
    ap.add_argument("--numerics-only", action="store_true")
    args = ap.parse_args()

    from sharar.engine.config import current, install
    from sharar.model.nemotron import config as c
    from sharar.model.nemotron.moe import NemotronMoe
    from sharar.target import NEMOTRON
    from sharar.weights.safetensors import SnapshotWeights
    install(current().replace(target="nemotron"))
    dev = torch.device("cuda")
    w = SnapshotWeights(NEMOTRON.snapshot_dir())
    calibrated = None
    if args.w3:
        from sharar.weights.w3_checkpoint import CalibratedWeights
        calibrated = CalibratedWeights(args.w3, revision=NEMOTRON.model_revision)
    moe_ids = [i for i, kind in enumerate(c.BLOCK_TYPES) if kind == "moe" and i >= args.first][:args.blocks]
    t0 = time.perf_counter()
    moes, experts, errors = [], [], []
    for i in moe_ids:
        moe = NemotronMoe(w, f"language_model.backbone.layers.{i}.mixer", dev, NEMOTRON)
        w.close()
        moes.append(moe)
        if calibrated is not None:
            ex = calibrated.load_latent_experts(f"layers.{i}", NEMOTRON.num_experts, moe.inter, moe.latent, dev)
            errors.append(None)
        else:
            s1 = w3_moe.unswizzle_batched(moe.s1, moe.inter, moe.latent // 16)
            s2 = w3_moe.unswizzle_batched(moe.s2, moe.latent, moe.inter // 16)
            ex, report = w3_moe.calibrate_relu2(moe.w1, s1, moe.g1, moe.w2, s2, moe.g2)
            errors.append(report)
            del s1, s2
        # routed_scaling_factor in fc2's row scales, as NemotronMoe(w3=...) serves them
        import dataclasses
        experts.append(dataclasses.replace(ex, row2=ex.row2 * c.ROUTED_SCALING_FACTOR))
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    w3_bytes = experts[0].bytes / w3_moe.EXPERTS
    print(f"loaded {len(moes)} blocks {moe_ids} in {time.perf_counter() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
          f"bytes per expert NVFP4 {NVFP4_BYTES}, 3-bit {w3_bytes:.0f}; errors {errors}", flush=True)
    w3_moe.w3._module()

    gen = torch.Generator().manual_seed(0)
    out = {"engine": "Sharar", "machine": args.machine, "blocks": moe_ids, "windows": args.windows, "iters": args.iters,
           "nvfp4_bytes_per_expert": NVFP4_BYTES, "w3_bytes_per_expert": w3_bytes, "weight_error": errors,
           "gpu": torch.cuda.get_device_name(), "timing": [], "numerics": []}
    for m in [int(v) for v in args.ms.split(",")]:
        xs, routes = [], []
        for _ in range(args.windows):
            xw, rw = [], []
            for moe in moes:
                amax = float(moe.in1) * 6 * 448
                xw.append((torch.randn((m, moe.latent), generator=gen) * amax / 8).to(dev, torch.bfloat16))
                ids = torch.stack([torch.randperm(w3_moe.EXPERTS, generator=gen)[:TOPK] for _ in range(m)]).to(torch.int32).to(dev)
                wt = torch.rand((m, TOPK), generator=gen)
                rw.append((ids, (wt / wt.sum(-1, keepdim=True)).to(dev)))
            xs.append(xw)
            routes.append(rw)
        distinct = sum(int(torch.unique(ids).numel()) for rw in routes for ids, _ in rw) / (args.windows * len(moes))
        row = {"engine": f"Sharar ({args.machine})", "m": m, "distinct_experts_per_block": round(distinct, 1)}
        steps = {
            "nvfp4": [(lambda i=i: [moe.routed(x, ids, wt) for moe, x, (ids, wt) in zip(moes, xs[i], routes[i])]) for i in range(args.windows)],
            "w3": [(lambda i=i: [w3_moe.w3_moe(ex, x, ids, wt) for ex, x, (ids, wt) in zip(experts, xs[i], routes[i])]) for i in range(args.windows)],
        }
        runs = {arm: [] for arm in steps}
        for _ in range(0 if args.numerics_only else args.repeats):
            for arm, s in steps.items():
                runs[arm].append(graph_us(s, args.iters if m <= 64 else max(4, args.iters // 8), len(moes)))
        for arm, per_expert in (("nvfp4", NVFP4_BYTES), ("w3", w3_bytes)) if runs["w3"] else ():
            us = sorted(runs[arm])[len(runs[arm]) // 2]
            row[f"{arm}_us"] = round(us, 1)
            row[f"{arm}_gbs"] = round(distinct * per_expert / (us * 1e3), 1)
        if runs["w3"]:
            row["speedup"] = round(row["nvfp4_us"] / row["w3_us"], 3)
            out["timing"].append(row)
            print(json.dumps(row), flush=True)
        if m > 64:
            continue
        for b in range(len(moes)):
            x, (ids, wt) = xs[0][b], routes[0][b]
            r_w3 = w3_moe.w3_moe(experts[b], x, ids, wt)
            r_nv = moes[b].routed(x, ids, wt)
            r_ex = exact(moes[b], x, ids, wt)
            rep = {"m": m, "block": moe_ids[b], "w3_vs_nvfp4": compare(r_w3, r_nv), "w3_vs_exact": compare(r_w3, r_ex),
                   "nvfp4_vs_exact": compare(r_nv, r_ex),
                   "w3_repeat_bit_equal": bool(torch.equal(r_w3, w3_moe.w3_moe(experts[b], x, ids, wt)))}
            if b == 0:
                rep["w3_vs_reference"] = compare(r_w3, w3_moe.reference(experts[b], x, ids, wt))
            out["numerics"].append(rep)
            print(json.dumps(rep), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
