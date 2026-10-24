"""Nsight Compute view of Sharar's Nemotron speculative round: every kernel of one eager round (graphs off, so each
kernel is its own launch) at a depth and batch, its DRAM bytes and throughput, occupancy and grid.

  run    loads the model, runs warm eager rounds, then one round (or `--rounds`) between cudaProfilerStart/Stop:
           flock ~/.local/share/sharar/.gpu.lock ncu --profile-from-start off --cache-control none --clock-control none \
             --metrics <METRICS> --csv --log-file OUT.csv python -m profiling.nemotron_ncu_profile run --depth 3 --batch 1
  table  aggregates such a CSV by kernel (and grid): calls per round, us per call, GB/s, % of the DRAM peak, MB per
         call read and written, achieved occupancy; sorted by total time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# GB10's LPDDR5X is system memory to the L2 (no dram__ counters): DRAM reads are the L2's sysmem fills, writes the
# L2's write sectors (all reach DRAM; nothing stays resident between rounds), 32 bytes a sector.
METRICS = ",".join([
    "gpu__time_duration.sum", "lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_read.sum", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
])
PROMPT = "Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example."


def run(args) -> None:
    import torch
    from sharar import target as targets
    from sharar.engine import config
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.tokenizer import SnapshotTokenizer
    if args.headroom_slack_gib:
        from sharar.cache import ledger
        check, slack = ledger.check_load_headroom, int(args.headroom_slack_gib * 2**30)
        ledger.check_load_headroom = lambda peak, *a, **k: check(max(0, peak - slack), *a, **k)
    t = targets.get("nemotron")
    cfg = config.current().replace(target="nemotron", hf_cache=str(Path.home() / ".cache/huggingface/hub"),
                                   w3_decode=False, w3_checkpoint=None)
    with config.using(cfg):
        model = targets.load_model(t, t.snapshot_dir(), log=None)
        tok = SnapshotTokenizer(t.snapshot_dir())
        ids = tok.encode(f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think></think>")
        kw = {}
        if args.depth:
            from sharar.cache.spec_budget import SpecConfig
            kw["spec_config"] = SpecConfig("mtp", args.depth, "reconstruct", "bf16", None)
        from sharar.kernels import nemotron_fused
        nemotron_fused.ENABLED = args.fused == "on"
        ex = Executor(model, log=None, max_batch=4, max_context=args.context, num_pages=args.pages, num_slots=args.slots, **kw)
        ex.capture_graphs(log=None)
        ex.use_graphs = args.graphs
        for b in [int(x) for x in str(args.batch).split(",")]:  # one profiler range per batch, in this order
            ex.flush_cache()
            reqs = [ex.admit(ids, SamplingParams(max_new_tokens=args.context - 200, ignore_eos=True)) for _ in range(b)]
            for r in reqs:
                ex.prefill(r)
            for _ in range(4):
                ex.decode_step(reqs)
            torch.cuda.synchronize()
            r0 = ex.spec.rounds if ex.spec else 0
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(args.rounds):
                ex.decode_step(reqs)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(json.dumps({"engine": "Sharar", "depth": args.depth, "batch": b,
                              "rounds": (ex.spec.rounds - r0) if ex.spec else args.rounds}), flush=True)
            for r in reqs:
                ex.release(r)


def _num(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def table(args) -> None:
    lines = Path(args.csv).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(lines[start:]))
    # one row per (kernel, metric) in ncu's long CSV
    kernels: dict[str, dict] = {}
    for r in rows:
        k = kernels.setdefault(r["ID"], {"name": r["Kernel Name"], "grid": r.get("Grid Size", ""), "block": r.get("Block Size", "")})
        k[r["Metric Name"]] = (_num(r["Metric Value"]), r.get("Metric Unit", ""))
    peak = args.peak_gbs

    def val(k, m, unit_scale=None):
        v, u = k.get(m, (0.0, ""))
        if unit_scale:
            return v * unit_scale.get(u, 1.0)
        return v
    to_bytes = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    to_ns = {"nsecond": 1, "usecond": 1e3, "msecond": 1e6, "ns": 1, "us": 1e3, "ms": 1e6}
    agg = defaultdict(lambda: defaultdict(float))
    for k in kernels.values():
        key = (k["name"][:90], k["grid"] if args.by_grid else "")
        a = agg[key]
        ns = val(k, "gpu__time_duration.sum", to_ns)
        rd = 32 * val(k, "lts__d_sectors_fill_sysmem.sum")
        wr = 32 * val(k, "lts__t_sectors_op_write.sum")
        a["calls"] += 1
        a["ns"] += ns
        a["read"] += rd
        a["write"] += wr
        a["l2"] += 32 * val(k, "lts__t_sectors_op_read.sum")
        a["mem_pct"] += val(k, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed") * ns
        a["occ"] += val(k, "sm__warps_active.avg.pct_of_peak_sustained_active") * ns
        a["sm"] += val(k, "sm__throughput.avg.pct_of_peak_sustained_elapsed") * ns
    total = sum(a["ns"] for a in agg.values())
    n = args.rounds
    out = []
    for (name, grid), a in sorted(agg.items(), key=lambda kv: -kv[1]["ns"]):
        gbs = (a["read"] + a["write"]) / a["ns"] if a["ns"] else 0.0
        out.append({"kernel": name, "grid": grid, "calls_per_round": round(a["calls"] / n, 1),
                    "us_per_round": round(a["ns"] / n / 1e3, 1), "us_per_call": round(a["ns"] / a["calls"] / 1e3, 2),
                    "GB/s": round(gbs, 1), "pct_peak": round(100 * gbs / peak, 1),
                    "MB_read_per_call": round(a["read"] / a["calls"] / 1e6, 3),
                    "MB_write_per_call": round(a["write"] / a["calls"] / 1e6, 3),
                    "L2_read_MB_per_call": round(a["l2"] / a["calls"] / 1e6, 3),
                    "mem_pipe_pct": round(a["mem_pct"] / a["ns"], 1) if a["ns"] else 0.0,
                    "occupancy_pct": round(a["occ"] / a["ns"], 1) if a["ns"] else 0.0,
                    "sm_pct": round(a["sm"] / a["ns"], 1) if a["ns"] else 0.0})
    summary = {"kernels": len(kernels), "kernels_per_round": round(len(kernels) / n, 1), "gpu_ms_per_round": round(total / n / 1e6, 3),
               "dram_GB_per_round": round(sum(a["read"] + a["write"] for a in agg.values()) / n / 1e9, 3),
               "tiny_under_5us_per_round": round(sum(1 for k in kernels.values() if val(k, "gpu__time_duration.sum", to_ns) < 5000) / n, 1),
               "tiny_ms_per_round": round(sum(val(k, "gpu__time_duration.sum", to_ns) for k in kernels.values()
                                              if val(k, "gpu__time_duration.sum", to_ns) < 5000) / n / 1e6, 3)}
    print(json.dumps(summary))
    for r in out[:args.top]:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--depth", type=int, default=3)
    r.add_argument("--batch", default="1", help="a comma list: one profiler range each")
    r.add_argument("--context", type=int, default=2048, help="executor max_context")
    r.add_argument("--pages", type=int, default=32768, help="KV pages (fixed: no plan from what other processes leave)")
    r.add_argument("--slots", type=int, default=24, help="state slots")
    r.add_argument("--headroom-slack-gib", type=float, default=0.0,
                   help="lower check_load_headroom's requirement by this much (other processes share the Spark's memory)")
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--fused", default="on", help="Nemotron fusions on or off")
    r.add_argument("--graphs", action="store_true", help="replay graphs (ncu --graph-profiling node) instead of eager")
    t = sub.add_parser("table")
    t.add_argument("csv")
    t.add_argument("--rounds", type=int, default=1)
    t.add_argument("--peak-gbs", type=float, default=273.0)
    t.add_argument("--by-grid", action="store_true")
    t.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    (run if a.cmd == "run" else table)(a)


if __name__ == "__main__":
    main()
