# SPDX-License-Identifier: Apache-2.0
"""Nemotron-3.5-Super's latent MoE block (`model/nemotron/moe.py`) by part and token count, on one real block.

Parts, each timed alone in CUDA graphs: the fused router|fc1_latent GEMM (FP32 out), the routing kernel (sigmoid + bias
top-22, which also casts the latent), the routed experts (`cutlass_fused_moe`, ReLU²), the shared expert (two NVFP4
linears and ReLU²), the fc2_latent addmm onto the shared output; and the whole block call. Calls of at most
`NemotronMoe.SMALL_ROWS` rows take Sharar's kernels (`kernels/nemotron_moe.py`) instead; for them the parts are its
launches (router: the router|fc1 GEMV with the routing, plan and quantizations; up; down; out), each timed as the
difference of the chain cut after it, and `large` is the same call through the parts above (`--large`: report only
those, as before the small-call kernels). Each M runs `--windows`
graphs, one per input window, replayed round-robin; the dense weights rotate over `--copies` copies (the block's
45.7 MB of them would stay half-hot in the 24 MB L2 otherwise) and every window routes differently, so the experts
stream from DRAM as in a forward over 40 blocks. Reported: median us per call over `--repeats`, bytes the call reads (dense
weights + distinct experts x 3.10 MB; activations not counted) and the GB/s that implies (the Spark's DRAM is ~273),
with nvidia-smi utilization around each M (the GPU may be shared).

Inputs: with `--capture N`, the block's real post-norm inputs for N tokens of a fixed mixed prompt (prose, code,
math), run through the blocks before it by `model/nemotron/reference.py` (Mamba and attention) and `NemotronMoe`
(earlier MoE blocks); saved to `--hidden` for later runs. Without them, Gaussian rows scaled by the block's norm weight
(which route far more uniformly than real text). Distinct experts per call on real rows: M consecutive positions
(a verify forward's rows) and M random positions (a decode batch of unrelated requests, approximated by positions of one
document), for the profiled block and every other captured block. `--budget KEEP:CAP` adds the budgeted selection
(route_budget) as one request of M rows: its distinct experts and the experts' time under it.

    python3 -m profiling.nemotron_moe --capture 2048 --hidden nemotron_moe_hidden.pt --out nemotron_moe.json
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

LAYER_PREFIX = "language_model.backbone.layers"
PROMPT = """# Notes on the 1918 influenza pandemic and modern surveillance

The 1918 pandemic killed an estimated 50 million people. Unlike seasonal flu, it struck young adults hardest; one
hypothesis is that older cohorts had partial immunity from an earlier H1-like strain. Public-health responses varied
by city: Philadelphia held a parade on September 28 and saw deaths peak within weeks, while St. Louis closed schools
and theaters early and had a flatter curve.

```python
import numpy as np

def sir(beta: float, gamma: float, s0: float, i0: float, days: int, dt: float = 0.1):
    \"\"\"Integrate the SIR model with forward Euler; returns arrays S, I, R sampled daily.\"\"\"
    steps = int(days / dt)
    s, i, r = s0, i0, 1.0 - s0 - i0
    out = []
    for k in range(steps):
        new_inf = beta * s * i * dt
        new_rec = gamma * i * dt
        s, i, r = s - new_inf, i + new_inf - new_rec, r + new_rec
        if k % int(1 / dt) == 0:
            out.append((s, i, r))
    return np.array(out).T
```

The basic reproduction number is R0 = beta / gamma. For R0 = 2.5 the final size z solves z = 1 - exp(-R0 z), so
z is about 0.89: let f(z) = 1 - exp(-2.5 z) - z; f(0.9) = 1 - e^{-2.25} - 0.9 = 1 - 0.1054 - 0.9 = -0.0054 and
f(0.88) = 1 - e^{-2.2} - 0.88 = 0.0092, so the root lies between 0.88 and 0.9.

Wastewater surveillance now tracks SARS-CoV-2 RNA concentrations days before clinical case counts rise. Les réseaux
de surveillance européens partagent leurs données chaque semaine; 日本でも下水サーベイランスが導入されている.
Q: Why might wastewater signals lead case counts? A: Shedding begins before symptoms, testing lags, and many
infections are never tested at all.
"""


def gpu_util() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return "?"


def graph_us(calls, iters: int, repeats: int) -> float:
    """calls: one callable per window, each captured in its own graph and replayed round-robin; median us per call."""
    for f in calls:
        f()
    torch.cuda.synchronize()
    graphs = []
    for f in calls:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        graphs.append(g)
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for i in range(iters):
            graphs[i % len(graphs)].replay()
        b.record()
        torch.cuda.synchronize()
        times.append(1000 * a.elapsed_time(b) / iters)
    del graphs
    return statistics.median(times)


def dense_copy(moe):
    """The block with its own copy of every dense weight (the experts shared): an L2-cold rotation partner."""
    other = copy.copy(moe)
    other.gate_fc1, other.fc2 = moe.gate_fc1.clone(), moe.fc2.clone()
    for name in ("shared_up", "shared_down"):
        lin = getattr(moe, name)
        setattr(other, name, dataclasses.replace(lin, weight=lin.weight.clone(), weight_scale_swizzled=lin.weight_scale_swizzled.clone()))
    return other


@torch.inference_mode()
def capture(w, snap: Path, layers: list[int], tokens: int, device) -> dict[int, torch.Tensor]:
    """Real post-norm inputs of the MoE blocks in `layers` for the first `tokens` tokens of PROMPT (repeated to length)."""
    from tokenizers import Tokenizer
    from sharar.model.nemotron import config as c
    from sharar.model.nemotron import reference as ref
    from sharar.model.nemotron.moe import NemotronMoe
    from sharar.target import NEMOTRON

    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    ids = tok.encode(PROMPT).ids
    ids = (ids * (tokens // len(ids) + 1))[:tokens]
    last = max(layers)
    other = [i for i in range(last + 1) if c.BLOCK_TYPES[i] != "moe"]
    model = ref.NemotronReference(snap, device=device, blocks=other, head=False, expert_cache_gib=0.1)
    x = w.get("language_model.backbone.embeddings.weight")[torch.tensor(ids)].to(device)
    out = {}
    for i in range(last + 1):
        if c.BLOCK_TYPES[i] != "moe":
            x = model.run_block(i, x, chunked=True)
            continue
        h = ref.rms_norm(x, w.get(f"{LAYER_PREFIX}.{i}.norm.weight").to(device), c.RMS_NORM_EPS).to(torch.bfloat16)
        if i in layers:
            out[i] = h.cpu()
        moe = NemotronMoe(w, f"{LAYER_PREFIX}.{i}.mixer", device, NEMOTRON)
        x = x + torch.cat([moe(h[s:s + 512]) for s in range(0, h.shape[0], 512)]).to(x.dtype)
        del moe
        torch.cuda.empty_cache()
    return out


def distinct_experts(gate_fc1: torch.Tensor, bias: torch.Tensor, hidden: torch.Tensor, ms, samples: int, gen) -> dict:
    """Mean distinct experts a call of M rows routes to: M consecutive positions, and M random ones."""
    from sharar.model.nemotron.moe import route_topk
    fused = torch.mm(hidden, gate_fc1.T, out_dtype=torch.float32)
    ids, _, _ = route_topk(fused, bias, 22, gate_fc1.shape[0] - bias.shape[0])
    ids = ids.cpu()
    t = ids.shape[0]
    res = {}
    for m in ms:
        if m > t:
            continue
        starts = torch.randint(0, t - m + 1, (samples,), generator=gen)
        consecutive = statistics.mean(len(torch.unique(ids[s:s + m])) for s in starts.tolist())
        rand = statistics.mean(len(torch.unique(ids[torch.randperm(t, generator=gen)[:m]])) for _ in range(samples))
        res[m] = {"consecutive": consecutive, "random": rand}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=1, help="the MoE block profiled (checkpoint numbering)")
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,128,512,2048")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--copies", type=int, default=4)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--hidden", default=None, help="captured inputs (.pt): read if present, written by --capture")
    ap.add_argument("--capture", type=int, default=0, help="capture this many tokens of real inputs first")
    ap.add_argument("--capture-layers", default="1,3,5,8,10,12,14")
    ap.add_argument("--budget", default=None, help="KEEP:CAP for the budgeted-selection columns")
    ap.add_argument("--large", action="store_true", help="every call through FlashInfer's and torch's kernels")
    args = ap.parse_args()

    from sharar.kernels import nemotron_moe
    from sharar.model.nemotron.moe import NemotronMoe, route_topk
    from sharar.target import NEMOTRON
    from sharar.weights.safetensors import SnapshotWeights

    dev = torch.device("cuda")
    from sharar.model.nemotron.reference import find_snapshot
    snap = find_snapshot()
    w = SnapshotWeights(snap)
    gen = torch.Generator().manual_seed(0)
    ms = [int(m) for m in args.ms.split(",")]

    hidden = {}
    if args.capture:
        layers = sorted({args.layer, *(int(i) for i in args.capture_layers.split(","))})
        hidden = capture(w, snap, layers, args.capture, dev)
        if args.hidden:
            torch.save(hidden, args.hidden)
    elif args.hidden and Path(args.hidden).exists():
        hidden = torch.load(args.hidden)
    report = {"layer": args.layer, "real_inputs": args.layer in hidden, "distinct": {}, "calls": []}

    if hidden:  # routing only, so every captured block is cheap
        for i, h in sorted(hidden.items()):
            p = f"{LAYER_PREFIX}.{i}.mixer"
            gf = torch.cat([w.get(f"{p}.gate.weight"), w.get(f"{p}.fc1_latent_proj.weight")]).to(dev)
            bias = w.get(f"{p}.gate.e_score_correction_bias").float().to(dev)
            report["distinct"][i] = distinct_experts(gf, bias, h.to(dev), ms, 64, gen)
        print("distinct experts per call (consecutive rows / random rows):")
        print("block " + " ".join(f"{m:>12d}" for m in ms))
        for i, d in report["distinct"].items():
            print(f"{i:5d} " + " ".join(f"{d[m]['consecutive']:5.1f}/{d[m]['random']:5.1f}" if m in d else " " * 12 for m in ms))

    moe = NemotronMoe(w, f"{LAYER_PREFIX}.{args.layer}.mixer", dev, NEMOTRON)
    if args.large:
        moe.SMALL_ROWS = 0
    mods = [moe] + [dense_copy(moe) for _ in range(args.copies - 1)]
    norm = w.get(f"{LAYER_PREFIX}.{args.layer}.norm.weight").to(dev).float()
    keep_cap = tuple(int(v) for v in args.budget.split(":")) if args.budget else None
    real = hidden.get(args.layer)
    print(f"block {args.layer}: {moe.bytes / 2**30:.2f} GiB ({moe.expert_bytes / 2**30:.2f} experts, "
          f"{moe.dense_bytes / 1e6:.1f} MB dense), {moe.bytes_per_expert / 1e6:.2f} MB an expert; "
          f"inputs {'real' if real is not None else 'Gaussian'}")
    header = f"{'M':>5} {'gemm':>8} {'route':>7} {'experts':>9} {'shared':>8} {'fc2':>7} {'total':>9} {'distinct':>8} {'MB':>7} {'GB/s':>6} util"
    if keep_cap:
        header += f"   budget {keep_cap[0]}:{keep_cap[1]}: distinct  experts_us"
    print(header)
    for m in ms:
        u0 = gpu_util()
        win = []
        for i in range(args.windows):
            mod = mods[i % len(mods)]
            if real is not None and real.shape[0] >= m:
                s = int(torch.randint(0, real.shape[0] - m + 1, (), generator=gen))
                x = real[s:s + m].to(dev)
            else:
                x = (torch.randn(m, NEMOTRON.hidden_size, generator=gen).to(dev) * norm).bfloat16()
            ctx = SimpleNamespace(mode="decode", slots=torch.arange(1, m + 1, dtype=torch.int32, device=dev)) if m <= 128 else None
            fused = torch.mm(x, mod.gate_fc1.T, out_dtype=torch.float32)
            ids, wt, lat = route_topk(fused, mod.bias, mod.topk, mod.latent)
            large = copy.copy(mod)
            large.SMALL_ROWS = 0
            routed = large.routed(lat, ids, wt)
            win.append(SimpleNamespace(mod=mod, large=large, x=x, ctx=ctx, fused=fused, ids=ids, wt=wt, lat=lat,
                                       routed=routed, shared=large.shared(x)))
        t = lambda f: graph_us([lambda v=v: f(v) for v in win], args.iters, args.repeats)  # noqa: E731
        if m <= moe.SMALL_ROWS:
            def chain(v, stages):
                w_ = v.mod.small_weights()
                _, ids, wt, ws = nemotron_moe.router(v.x, w_, v.mod.bias, v.mod.topk)
                return nemotron_moe.block(v.x, ws, ids, wt, w_, stages=stages)
            cut = [t(lambda v: nemotron_moe.router(v.x, v.mod.small_weights(), v.mod.bias, v.mod.topk))]
            cut += [t(lambda v, s=s: chain(v, s)) for s in (2, 3, 4)]
            row = {"m": m, "router_us": cut[0], "up_us": cut[1] - cut[0], "down_us": cut[2] - cut[1], "out_us": cut[3] - cut[2],
                   "total_us": t(lambda v: v.mod(v.x, v.ctx)), "large_us": t(lambda v: v.large(v.x, v.ctx)),
                   "distinct": statistics.mean(len(torch.unique(v.ids)) for v in win)}
            row["bytes"] = moe.dense_bytes + row["distinct"] * moe.bytes_per_expert
            row["gbps"] = row["bytes"] / (row["total_us"] * 1e-6) / 1e9
            row["util"] = f"{u0}->{gpu_util()}"
            report["calls"].append(row)
            print(f"{m:5d} small: router {row['router_us']:6.1f} up {row['up_us']:7.1f} down {row['down_us']:7.1f} "
                  f"out {row['out_us']:6.1f} total {row['total_us']:8.1f} (large {row['large_us']:8.1f}) "
                  f"{row['distinct']:8.1f} {row['bytes'] / 1e6:7.1f} {row['gbps']:6.0f} {row['util']}", flush=True)
            del win
            torch.cuda.empty_cache()
            continue
        row = {
            "m": m,
            "gemm_us": t(lambda v: torch.mm(v.x, v.mod.gate_fc1.T, out_dtype=torch.float32)),
            "route_us": t(lambda v: route_topk(v.fused, v.mod.bias, v.mod.topk, v.mod.latent)),
            "experts_us": t(lambda v: v.mod.routed(v.lat, v.ids, v.wt)),
            "shared_us": t(lambda v: v.mod.shared(v.x)),
            "fc2_us": t(lambda v: torch.addmm(v.shared, v.routed, v.mod.fc2.T)),
            "total_us": t(lambda v: v.mod(v.x, v.ctx)),
            "distinct": statistics.mean(len(torch.unique(v.ids)) for v in win),
        }
        row["bytes"] = moe.dense_bytes + row["distinct"] * moe.bytes_per_expert
        row["gbps"] = row["bytes"] / (row["total_us"] * 1e-6) / 1e9
        row["experts_gbps"] = row["distinct"] * moe.bytes_per_expert / (row["experts_us"] * 1e-6) / 1e9
        if keep_cap and m <= 128:
            for v in win:
                v.mod = copy.copy(v.mod)
                v.mod.keep, v.mod.budget, v.mod.per_request = keep_cap[0], keep_cap[1], 0
                v.bctx = SimpleNamespace(mode="verify", seq_slots=torch.ones(1, dtype=torch.int32, device=dev), verify_steps=m)
                v.bids, v.bwt, _ = v.mod.route(v.fused, v.bctx)
            row["budget_distinct"] = statistics.mean(len(torch.unique(v.bids)) for v in win)
            row["budget_experts_us"] = t(lambda v: v.mod.routed(v.lat, v.bids, v.bwt))
            row["budget_route_us"] = t(lambda v: v.mod.route(v.fused, v.bctx))
        row["util"] = f"{u0}->{gpu_util()}"
        report["calls"].append(row)
        line = (f"{m:5d} {row['gemm_us']:8.1f} {row['route_us']:7.1f} {row['experts_us']:9.1f} {row['shared_us']:8.1f} "
                f"{row['fc2_us']:7.1f} {row['total_us']:9.1f} {row['distinct']:8.1f} {row['bytes'] / 1e6:7.1f} "
                f"{row['gbps']:6.0f} {row['util']}")
        if "budget_distinct" in row:
            line += f"   {row['budget_distinct']:8.1f} {row['budget_experts_us']:10.1f} (route {row['budget_route_us']:.1f})"
        print(line, flush=True)
        del win
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
