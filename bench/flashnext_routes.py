# SPDX-License-Identifier: Apache-2.0
"""Flash-Next router traces over a frozen prompt mix, in-process (#32).

Runs the pinned prompts (`bench/workloads/flashnext_routes_v1.json`: the 22 h2h decode prompts
and 6 tool-calling agent conversations) one at a time through the scheduler with the trace sink
and router traces on (`SHARAR_TRACE_ROUTER=1`, every request keeps every route row), then turns
the traces into one self-describing dataset per arm (`dataset`).

    # in the runtime image, the checkout on PYTHONPATH, the GPU lock held:
    SHARAR_TARGET=flashnext SHARAR_TRACE_ROUTER=1 SHARAR_TRACE_ROUTER_SAMPLE=1 SHARAR_DRAFT_HEAD=w3 \\
      SHARAR_BENCH_VERIFY_STATE=reconstruct [SHARAR_FLASHNEXT_W3_CHECKPOINT=/w3] \\
      python3 -m bench.flashnext_routes capture --arm fast-k4 --profile fast --k 4 --modes greedy,t07 --out RUN
    python3 -m bench.flashnext_routes dataset --run RUN --out DATA     # CPU
    python3 -m bench.flashnext_routes ab --k 4 --out ab.json           # outputs and round time on fixed prompts

`ab` only uses interfaces that main already has, so the same file measures main's checkout
(PYTHONPATH pointing at it) for the bit-identity and round-time gate. Analysis: bench/route_stats.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

WORKLOAD = Path(__file__).parent / "workloads" / "flashnext_routes_v1.json"
PROFILES = {  # the acceptance rule of each serving profile (run-sharar.sh); the trunk comes from the environment
    "fast": {"accept_top_k": 3, "accept_margin": 2.0},
    "exact": {"accept_top_k": 1, "accept_margin": 0.0},
}
AB_PROMPTS = ("code-lru", "prose-history", "math-finance", "de-essay", "json-extract", "agent-fix-lru")


def load_workload(ids=None) -> dict:
    w = json.loads(WORKLOAD.read_text())
    if ids:
        w["prompts"] = [p for p in w["prompts"] if p["id"] in ids]
    return w


def render(snap, workload) -> list[tuple[str, str, list[int]]]:
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer
    template, tok = ChatTemplate(snap), SnapshotTokenizer(snap)
    out = []
    for p in workload["prompts"]:
        text = template.render(p["messages"], enable_thinking=False, tools=workload["tools"] if p["cat"] == "agent" else None)
        out.append((p["id"], p["cat"], tok.encode(text)))
    return out


def params(workload, profile: str, mode: str, max_new: int, seed: int, trace: bool = True):
    from sharar.engine.sampler import SamplingParams
    s = workload["sampling"][mode]
    kw = {"temperature": s["temperature"], "max_new_tokens": max_new, "trace": trace, **PROFILES[profile]}
    if mode != "greedy":
        kw.update(top_p=s["top_p"], top_k=s["top_k"], seed=seed)
    return SamplingParams(**kw)


def serve_one(ex, sched, ids, sampling, routes: bool = True):
    import torch
    ex.flush_cache()  # every prompt cold: prefix-cache hits change continuations (docs/flashnext.md)
    torch.cuda.synchronize()
    req = sched.submit(list(ids), sampling)
    if not routes:  # traced without the router recorder: isolates the router traces' host cost
        req.trace_routes = None
    sched.run_until_idle()
    torch.cuda.synchronize()
    decode = req.t_finish - req.t_first_token
    return req, {"output_tokens": len(req.output_ids), "finish": req.finish_reason, "decode_s": round(decode, 4),
                 "tok_s": round((len(req.output_ids) - 1) / decode, 2) if decode > 0 else None,
                 "rounds": req.spec_rounds, "emitted": req.spec_emitted, "accepted": req.spec_accepted,
                 "round_ms": round(1000 * decode / req.spec_rounds, 3) if req.spec_rounds else None}


def drain(sink, expected: int) -> None:
    deadline = time.time() + 120
    while sink.records + sink.dropped < expected and time.time() < deadline:
        time.sleep(0.05)
    sink.close(timeout=60)


def capture(args) -> None:
    import torch
    from sharar import target as targets
    from sharar.engine.config import current
    from sharar.engine.run import _load
    from sharar.engine.scheduler import Scheduler, SchedulerConfig
    from sharar.engine.traces import TraceSink
    cfg = current()
    if not cfg.trace_router:
        raise SystemExit("set SHARAR_TRACE_ROUTER=1 (and SHARAR_TRACE_ROUTER_SAMPLE=1 to keep every row)")
    workload = load_workload(args.prompts.split(",") if args.prompts else None)
    out = Path(args.out)
    model, ex, _ = _load(graphs=True, spec=args.k)
    snap = targets.get(cfg.target).snapshot_dir()
    prompts = render(snap, workload)
    meta = {"arm": args.arm, "profile": args.profile, "k": args.k, "modes": args.modes, "max_new_tokens": args.max_new,
            "workload": workload["version"], "config": cfg.describe(), "acceptance": PROFILES[args.profile],
            "draft_head": type(ex.spec.proposer.head.draft_head).__name__, "trunk_w3": getattr(model, "w3_identity", None),
            "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for mode in args.modes.split(","):
        run = out / f"{args.arm}-{mode}"
        (run / "traces").mkdir(parents=True, exist_ok=True)
        sched = Scheduler(ex, SchedulerConfig(max_running=1))
        sink = sched.sink = TraceSink(run / "traces", max_bytes=1 << 40, rotate_bytes=1 << 30)
        rows = []
        for i, (pid, cat, ids) in enumerate(prompts):
            req, row = serve_one(ex, sched, ids, params(workload, args.profile, mode, args.max_new, seed=1000 + i))
            row = {"trace_id": req.request_id, "prompt_id": pid, "cat": cat, "mode": mode, "prompt_tokens": len(ids), **row}
            rows.append(row)
            print(json.dumps(row), flush=True)
        drain(sink, len(prompts))
        stats = {"tokens": sum(r["output_tokens"] for r in rows), "rounds": sum(r["rounds"] for r in rows),
                 "sink": sink.stats(), "spec": ex.spec.stats()}
        (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "meta.json").write_text(json.dumps({**meta, "mode": mode, **stats}, indent=1))
        print(json.dumps({"arm": args.arm, "mode": mode, **{k: v for k, v in stats.items() if k != "spec"}}), flush=True)


def ab(args) -> None:
    """Outputs and per-round wall time of fixed greedy prompts, repeated and interleaved (the gate's A/B cell)."""
    import torch
    from sharar import target as targets
    from sharar.engine.config import current
    from sharar.engine.run import _load
    from sharar.engine.scheduler import Scheduler, SchedulerConfig
    cfg = current()
    workload = load_workload(AB_PROMPTS)
    model, ex, _ = _load(graphs=True, spec=args.k)
    prompts = render(targets.get(cfg.target).snapshot_dir(), workload)
    sched = Scheduler(ex, SchedulerConfig(max_running=1))
    sink = None
    router = bool(getattr(cfg, "trace_router", False))
    if args.traces:
        from sharar.engine.traces import TraceSink
        sink = sched.sink = TraceSink(args.traces, max_bytes=1 << 40, rotate_bytes=1 << 30)
    runs, outputs = [], {}
    arms = (("traced", "traced-noroutes", "untraced") if router else ("traced", "untraced")) if sink is not None else ("plain",)
    for rep in range(args.repeats):
        for pid, cat, ids in prompts:
            for arm in arms:
                req, row = serve_one(ex, sched, ids, params(workload, args.profile, "greedy", args.max_new, 0, trace=arm != "untraced"),
                                     routes=arm != "traced-noroutes")
                key = f"{pid}"
                if key in outputs and outputs[key] != req.output_ids:
                    row["differs_from_first"] = True
                outputs.setdefault(key, list(req.output_ids))
                runs.append({"rep": rep, "prompt_id": pid, "arm": arm, **row})
                print(json.dumps(runs[-1]), flush=True)
    if sink is not None:
        drain(sink, args.repeats * len(prompts) * (len(arms) - 1))
    summary = {}
    for arm in arms:
        sel = [r for r in runs if r["arm"] == arm]
        summary[arm] = {"decode_s": round(sum(r["decode_s"] for r in sel), 4), "rounds": sum(r["rounds"] for r in sel),
                        "ms_per_round": round(1000 * sum(r["decode_s"] for r in sel) / max(1, sum(r["rounds"] for r in sel)), 3),
                        "per_rep_ms_per_round": [round(1000 * sum(r["decode_s"] for r in sel if r["rep"] == i)
                                                       / max(1, sum(r["rounds"] for r in sel if r["rep"] == i)), 3) for i in range(args.repeats)]}
    doc = {"label": args.label, "k": args.k, "profile": args.profile, "router_traces": router, "config": cfg.describe(),
           "summary": summary, "outputs": outputs, "runs": runs}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps({"label": args.label, "summary": summary}), flush=True)


def dataset(args) -> None:
    """Traces of every `<arm>-<mode>` run under --run -> one dataset directory per run (README in bench/route_stats.py)."""
    import numpy as np
    from sharar.engine.routes import load_routes
    from sharar.engine.traces import iter_records
    out = Path(args.out)
    for run in sorted(p for p in Path(args.run).iterdir() if (p / "requests.jsonl").is_file()):
        meta = json.loads((run / "meta.json").read_text())
        rows_by_trace = {r["trace_id"]: r for r in map(json.loads, (run / "requests.jsonl").read_text().splitlines())}
        k = meta["k"]
        requests, R, N = [], [], []
        row_base = round_base = 0
        for doc in iter_records(run / "traces"):
            info = rows_by_trace[doc["id"]]
            routes = load_routes(doc["routes"]["path"]) if doc.get("routes") else None
            if routes is None or "ids" not in routes:
                raise SystemExit(f"{run.name}: request {doc['id']} has no full route rows (capture with SHARAR_TRACE_ROUTER_SAMPLE=1)")
            rounds = doc["spec"]["rounds"] if doc["spec"] else []
            index = len(requests)
            requests.append({"request": index, "prompt_id": info["prompt_id"], "cat": info["cat"], "mode": info["mode"],
                             "params": doc["params"], "prompt_ids": doc["prompt_ids"], "output_ids": doc["output_ids"],
                             "finish": doc["reason"], "tok_s": info["tok_s"], "rounds": len(rounds),
                             "first_round": round_base, "first_row": row_base, "rows": int(routes["ids"].shape[0])})
            n = routes["ids"].shape[0]
            N.append({"request": np.full(n, index, np.int32), **{name: routes[name] for name in ("ids", "probs", "kind", "round", "offset", "position")}})
            verify_rows = np.nonzero(routes["kind"] == 1)[0]
            first_rows = {int(routes["round"][i]): int(i) for i in verify_rows[::-1]}  # first row of each round's block
            draft_by_round = {int(r): j for j, r in enumerate(routes.get("draft_round", []))}
            for j, rec in enumerate(rounds):
                kk = len(rec["y"])
                z = rec["z"] if rec.get("z") is not None else [-1] * (kk + 1)
                dj = draft_by_round.get(j)
                R.append({"request": index, "round": j, "k": kk, "accepted": rec["a"], "emitted": min(rec["a"] + 1, kk + 1),
                          "drafted": rec["y"], "target": z, "correction": rec["c"],
                          "relaxed": [int(o in rec.get("rx", [])) for o in range(kk)],
                          "p1": rec.get("p1", [np.nan] * kk), "margin": rec.get("m", [np.nan] * kk),
                          "row0": row_base + first_rows[j] if j in first_rows else -1,
                          "position": int(routes["position"][first_rows[j]]) if j in first_rows else -1,
                          "draft_ids": routes["draft_ids"][dj, :kk] if dj is not None else np.zeros((kk, 10), np.uint16),
                          "draft_probs": routes["draft_probs"][dj, :kk] if dj is not None else np.zeros((kk, 10), np.float16)})
            row_base += n
            round_base += len(rounds)
        target = out / run.name
        target.mkdir(parents=True, exist_ok=True)
        cat = lambda name, dtype=None: np.concatenate([p[name] for p in N]).astype(dtype or N[0][name].dtype)  # noqa: E731
        np.savez(target / "rows.npz", request=cat("request"), ids=cat("ids"), probs=cat("probs"), kind=cat("kind"),
                 round=cat("round"), offset=cat("offset"), position=cat("position"))
        stack = lambda name, dtype: np.array([r[name] for r in R], dtype=dtype)  # noqa: E731
        np.savez(target / "rounds.npz", request=stack("request", np.int32), round=stack("round", np.int32), k=stack("k", np.int16),
                 accepted=stack("accepted", np.int16), emitted=stack("emitted", np.int16), drafted=stack("drafted", np.int32),
                 target=stack("target", np.int32), correction=stack("correction", np.int32), relaxed=stack("relaxed", np.int8),
                 p1=stack("p1", np.float32), margin=stack("margin", np.float32), row0=stack("row0", np.int64),
                 position=stack("position", np.int32), draft_ids=np.stack([r["draft_ids"] for r in R]).astype(np.uint16),
                 draft_probs=np.stack([r["draft_probs"] for r in R]).astype(np.float16))
        (target / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
        (target / "meta.json").write_text(json.dumps({**meta, "requests": len(requests), "rounds": len(R), "rows": row_base}, indent=1))
        print(json.dumps({"run": run.name, "requests": len(requests), "rounds": len(R), "rows": row_base}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--arm", required=True)
    c.add_argument("--profile", choices=sorted(PROFILES), required=True)
    c.add_argument("--k", type=int, required=True)
    c.add_argument("--modes", default="greedy")
    c.add_argument("--max-new", type=int, default=1536)
    c.add_argument("--prompts", default=None, help="comma list of prompt ids (default: all)")
    c.add_argument("--out", required=True)
    a = sub.add_parser("ab")
    a.add_argument("--label", required=True)
    a.add_argument("--k", type=int, default=4)
    a.add_argument("--profile", choices=sorted(PROFILES), default="exact")
    a.add_argument("--max-new", type=int, default=256)
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--traces", default=None, help="record traces here (router traces when SHARAR_TRACE_ROUTER=1)")
    a.add_argument("--out", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--run", required=True)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    {"capture": capture, "ab": ab, "dataset": dataset}[args.cmd](args)


if __name__ == "__main__":
    main()
