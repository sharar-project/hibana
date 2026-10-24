"""Where Sharar's Nemotron decode round spends its GPU and host time: graph replays, eager GPU work between them, GPU
idle gaps (inside a replay and between replays), kernels per round, tiny kernels.

Two modes, one model load:

  run   steady-state rounds at each depth and batch (CUDA graphs, exact acceptance, greedy). Every graph replay and
        every host phase (draft step, verify, accept, commit) is an NVTX range and has CUDA events before and after it,
        so without a profiler the events give the idle time between replays (`gaps`), and under
        `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node` the measured window is the profiler
        range (`--nsys`).
  nsys  reads the `nsys export --type sqlite` of such a capture: per round, GPU busy (the union of kernels, memcpys,
        memsets) against the round's span, kernels by phase, gaps between consecutive activities by where they fall
        (inside one graph launch: device launch latency between nodes; between launches or eager work: the host), the
        kernels under 5 us, and the top kernels.

  flock ~/.local/share/sharar/.gpu.lock python -m profiling.nemotron_gap_profile run --depths 0,3 --batches 1,4
  flock ~/.local/share/sharar/.gpu.lock nsys profile -o OUT --capture-range=cudaProfilerApi --cuda-graph-trace=node \
      -t cuda,nvtx python -m profiling.nemotron_gap_profile run --depths 3 --batches 1 --nsys
  nsys export --type sqlite OUT.nsys-rep && python -m profiling.nemotron_gap_profile nsys OUT.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROMPTS = ["Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.",
           "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
           "Explain step by step how to compute 37 * 48 mentally, then verify with long multiplication.",
           "Describe the causes of the French Revolution in five bullet points, with one sentence each."]


# ---------------------------------------------------------------------------------------------------------------------
# run: instrumented steady-state rounds

class Marks:
    """CUDA events before and after each instrumented call, with NVTX ranges of the same name."""

    def __init__(self):
        self.on = False
        self.events: list[tuple[str, object, object]] = []
        self.host: dict[str, float] = defaultdict(float)

    def wrap(self, obj, attr: str, name: str, device: bool = True):
        import torch
        orig = getattr(obj, attr)

        def call(*a, **kw):
            if not self.on:
                return orig(*a, **kw)
            torch.cuda.nvtx.range_push(name)
            t0 = time.perf_counter()
            if device:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record()
            try:
                return orig(*a, **kw)
            finally:
                if device:
                    e1 = torch.cuda.Event(enable_timing=True)
                    e1.record()
                    self.events.append((name, e0, e1))
                self.host[name] += time.perf_counter() - t0
                torch.cuda.nvtx.range_pop()
        setattr(obj, attr, call)


def _graphs(ex):
    """Every captured graph a round replays, by kind."""
    out = []
    if ex.spec is not None:
        spec = ex.spec
        out += [("draft", g) for g in spec.proposer.graphs._graphs.values()]
        for graphs in spec.width_graphs.values():
            out += [("verify", g) for g in graphs._graphs.values()]
            accept = getattr(graphs, "accept", None)
            if accept is not None:
                out += [("accept", g) for g in accept._graphs.values()]
    if ex.graphs is not None:
        out += [("decode", g) for g in ex.graphs._graphs.values()]
    return out


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
        ids = [tok.encode(f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think></think>")
               for p in PROMPTS]
        started = [False]
        from sharar.kernels import nemotron_fused
        for variant, dacc, d in [(v, a, int(x)) for x in args.depths.split(",") for v in args.fused.split(",")
                                 for a in args.device_accept.split(",")]:
            nemotron_fused.ENABLED = variant == "on"  # resolved when the graphs are captured
            vcfg = config.current().replace(device_accept=dacc == "on")  # SHARAR_DEVICE_ACCEPT, read when built
            kw = {}
            if d:
                from sharar.cache.spec_budget import SpecConfig
                kw["spec_config"] = SpecConfig("mtp", d, "reconstruct", "bf16", None)
            with config.using(vcfg):
                ex = Executor(model, log=None, max_batch=4, max_context=args.context, num_pages=args.pages,
                              num_slots=args.slots, **kw)
                ex.capture_graphs(log=None)
                ex.warmup(log=None)
            marks = Marks()
            for kind, g in _graphs(ex):
                marks.wrap(g, "replay", f"replay:{kind}")
            if ex.spec is not None:
                marks.wrap(ex.spec, "prepare", "host:prepare", device=False)
                marks.wrap(ex.spec.proposer, "_step", "host:draft_step", device=False)
                marks.wrap(ex.spec, "_accept_batch", "eager:accept")
                marks.wrap(ex.state, "commit_reconstructed", "eager:commit")
            marks.wrap(ex, "decode_step", "round", device=True)
            for b in [int(x) for x in args.batches.split(",")]:
                ex.flush_cache()
                reqs = [ex.admit(ids[j % 4], SamplingParams(max_new_tokens=args.context - 200, ignore_eos=True)) for j in range(b)]
                for r in reqs:
                    ex.prefill(r)
                for _ in range(args.warm):
                    ex.decode_step(reqs)
                torch.cuda.synchronize()
                r0, e0 = (ex.spec.rounds, ex.spec.emitted) if ex.spec else (0, 0)
                marks.events.clear()
                marks.host.clear()
                marks.on = True
                if args.nsys and not started[0]:
                    torch.cuda.cudart().cudaProfilerStart()  # once: nsys's default capture range ends at the first stop
                    started[0] = True
                torch.cuda.nvtx.range_push(f"window:fused-{variant}:accept-{dacc}:depth{d}:batch{b}")
                t0 = time.perf_counter()
                for _ in range(args.rounds):
                    ex.decode_step(reqs)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                torch.cuda.nvtx.range_pop()
                marks.on = False
                n = args.rounds
                tokens = ((ex.spec.emitted - e0) if ex.spec else n * b)
                # idle between consecutive device-timed marks (replays and eager phases, rounds excluded)
                seq = [m for m in marks.events if m[0] != "round"]
                busy, gaps = defaultdict(float), defaultdict(float)
                for k, a, z in seq:
                    busy[k] += a.elapsed_time(z)
                for (k1, a1, z1), (k2, a2, z2) in zip(seq, seq[1:]):
                    gaps[f"{k1} -> {k2}"] += max(0.0, z1.elapsed_time(a2))
                rounds = [m for m in marks.events if m[0] == "round"]
                row = {"engine": "Sharar", "fused": variant, "device_accept": dacc, "depth": d, "batch": b, "rounds": n,
                       "spec_rounds": (ex.spec.rounds - r0) if ex.spec else n,
                       "tokens_per_round": round(tokens / max(1, (ex.spec.rounds - r0) if ex.spec else n) / (b if not ex.spec else b), 3),
                       "tok_s": round(tokens / wall, 2), "wall_ms_per_round": round(1000 * wall / n, 2),
                       "round_event_ms": round(sum(a.elapsed_time(z) for _, a, z in rounds) / n, 2),
                       "device_ms_per_round": {k: round(v / n, 3) for k, v in busy.items()},
                       "between_ms_per_round": {k: round(v / n, 3) for k, v in sorted(gaps.items(), key=lambda kv: -kv[1])},
                       "idle_between_ms_per_round": round(sum(gaps.values()) / n, 3),
                       "host_ms_per_round": {k: round(1000 * v / n, 3) for k, v in marks.host.items()}}
                print(json.dumps(row), flush=True)
                for r in reqs:
                    ex.release(r)
            del ex, marks, reqs
            import gc
            gc.collect()  # the executor's graphs and the instrumentation's closures form cycles
            torch.cuda.empty_cache()
            print(f"freed: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", file=sys.stderr, flush=True)
        if started[0]:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------------------------------------------------------------
# nsys: analysis of an exported capture

TINY_US = 5.0


def _union(iv):
    total, cur = 0, None
    for s, e in sorted(iv):
        if cur is None or s > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0)


def nsys(args) -> None:
    db = sqlite3.connect(args.sqlite)
    q = lambda s: db.execute(s).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    nvtx = [(s, e, txt if txt is not None else strings.get(tid)) for s, e, txt, tid in
            q("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL")]
    acts = []  # (start, end, name, correlation, kind)
    for s, e, n, c in q("SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        acts.append((s, e, strings.get(n, str(n)), c, "kernel"))
    for tab, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if tab in tables:
            for s, e, c in q(f"SELECT start, end, correlationId FROM {tab}"):
                acts.append((s, e, label, c, label))
    acts.sort()
    runtime = q("SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")
    rt_name = {c: strings.get(n, "") for _, _, c, n in runtime}
    rt_start = {c: s for s, _, c, _ in runtime}
    # each graph launch's kind: the innermost replay:* NVTX range around its runtime call
    replays = sorted((s, e, t.split(":", 1)[1]) for s, e, t in nvtx if t and t.startswith("replay:"))
    phases = sorted((s, e, t) for s, e, t in nvtx if t and (t.startswith("eager:") or t.startswith("host:")))

    def kind_of(corr):
        name = rt_name.get(corr, "")
        ts = rt_start.get(corr)
        if ts is None:
            return "unknown"
        if "GraphLaunch" in name:
            for s, e, k in replays:
                if s <= ts <= e:
                    return f"graph:{k}"
            return "graph:?"
        inner = [(s, e, t) for s, e, t in phases if s <= ts <= e]
        if inner:
            return min(inner, key=lambda x: x[1] - x[0])[2]
        return "eager:other"
    kinds = {}
    for a in acts:
        if a[3] not in kinds:
            kinds[a[3]] = kind_of(a[3])
    windows = [(s, e, t) for s, e, t in nvtx if t and t.startswith("window:")]
    rounds = sorted((s, e) for s, e, t in nvtx if t == "round")
    report = {}
    for ws, we, wt in windows:
        rr = [(s, e) for s, e in rounds if ws <= s and e <= we]
        n = max(1, len(rr))
        wa = [a for a in acts if ws <= rt_start.get(a[3], -1) <= we]  # launched inside the window
        # device span: first to last activity launched inside the window
        if not wa:
            continue
        span = wa[-1][1] - wa[0][0]
        busy = _union([(a[0], a[1]) for a in wa])
        by_phase = defaultdict(lambda: [0, 0.0])
        tiny = defaultdict(lambda: [0, 0.0])
        top = defaultdict(lambda: [0, 0.0])
        for s, e, name, c, k in wa:
            p = kinds[c]
            by_phase[p][0] += 1
            by_phase[p][1] += (e - s) / 1e3
            if e - s < TINY_US * 1e3:
                tiny[p][0] += 1
                tiny[p][1] += (e - s) / 1e3
            top[name][0] += 1
            top[name][1] += (e - s) / 1e3
        # gaps between consecutive activities (device idle), by location
        gaps = defaultdict(lambda: [0, 0.0])
        end_so_far = wa[0][1]
        prev = wa[0]
        for a in wa[1:]:
            g = a[0] - end_so_far
            if g > 0:
                k1, k2 = kinds[prev[3]], kinds[a[3]]
                where = f"inside {k1}" if prev[3] == a[3] else f"{k1} -> {k2}"
                gaps[where][0] += 1
                gaps[where][1] += g / 1e3
            if a[1] >= end_so_far:
                end_so_far, prev = a[1], a
        report[wt] = {
            "rounds": len(rr), "round_ms": round(sum(e - s for s, e in rr) / n / 1e6, 3),
            "device_span_ms_per_round": round(span / n / 1e6, 3), "busy_ms_per_round": round(busy / n / 1e6, 3),
            "idle_ms_per_round": round((span - busy) / n / 1e6, 3),
            "activities_per_round": round(len(wa) / n, 1),
            "by_phase_per_round": {p: {"count": round(v[0] / n, 1), "ms": round(v[1] / n / 1e3, 3)}
                                   for p, v in sorted(by_phase.items(), key=lambda kv: -kv[1][1])},
            "tiny_under_5us_per_round": {p: {"count": round(v[0] / n, 1), "ms": round(v[1] / n / 1e3, 3)} for p, v in tiny.items()},
            "gaps_per_round": {w: {"count": round(v[0] / n, 1), "ms": round(v[1] / n / 1e3, 3),
                                   "mean_us": round(v[1] / max(1, v[0]), 2)}
                               for w, v in sorted(gaps.items(), key=lambda kv: -kv[1][1])},
            "top_kernels_per_round": [{"name": k[:140], "count": round(v[0] / n, 1), "us": round(v[1] / n, 1),
                                       "mean_us": round(v[1] / v[0], 2)}
                                      for k, v in sorted(top.items(), key=lambda kv: -kv[1][1])[:args.top]],
        }
    json.dump(report, sys.stdout, indent=1)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--depths", default="0,3")
    r.add_argument("--batches", default="1,4")
    r.add_argument("--context", type=int, default=2048, help="executor max_context")
    r.add_argument("--pages", type=int, default=32768, help="KV pages (fixed: no plan from what other processes leave)")
    r.add_argument("--slots", type=int, default=24, help="state slots")
    r.add_argument("--headroom-slack-gib", type=float, default=0.0,
                   help="lower check_load_headroom's requirement by this much (other processes share the Spark's memory)")
    r.add_argument("--rounds", type=int, default=40)
    r.add_argument("--device-accept", default="off", help="SHARAR_DEVICE_ACCEPT per executor: off, on or off,on")
    r.add_argument("--fused", default="off,on", help="Nemotron fusions (SHARAR_NEMOTRON_FUSED) per executor, in order")
    r.add_argument("--warm", type=int, default=8)
    r.add_argument("--nsys", action="store_true", help="bracket each measured window with cudaProfilerStart/Stop")
    n = sub.add_parser("nsys")
    n.add_argument("sqlite")
    n.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    (run if a.cmd == "run" else nsys)(a)


if __name__ == "__main__":
    main()
