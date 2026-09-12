"""Replay a trace directory's held-out prompts against a running server and
report decode TPS and tokens per round, for A/B of draft checkpoints.

The held-out set is the trainer's: the newest fraction of the traces. Prompts
are sent as exact token IDs through ``/generate`` with the recorded output
budget, greedy, one at a time (BS1), after one excluded warmup. Server-side
speculation counters from ``/get_server_info`` give tokens per round. Run once
per server configuration (pinned draft, trained draft) in separate windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def server_info(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/get_server_info", timeout=10) as r:
        return json.loads(r.read())


def spec_counters(info: dict) -> dict:
    spec = info.get("spec_stats") or {}
    return {k: spec.get(k) for k in ("rounds", "emitted", "accepted", "proposed")}


def generate(url: str, ids: list[int], max_new_tokens: int, timeout: float) -> dict:
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0, "ignore_eos": True}, "trace": False}
    req = urllib.request.Request(f"{url}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    meta = payload.get("meta_info") or payload.get("meta") or payload
    return {"seconds": elapsed, "completion_tokens": meta.get("completion_tokens"), "e2e_latency": meta.get("e2e_latency"),
            "rounds": meta.get("spec_verify_ct"), "tokens_per_round": meta.get("spec_tokens_per_round"), "output_ids": payload.get("output_ids")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--holdout-tokens", type=int, default=20000, help="the trainer's held-out budget")
    ap.add_argument("--count", type=int, default=24, help="newest held-out requests to replay")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="run")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from sharar.train.data import load_samples, split
    samples = load_samples(args.traces, max_length=args.max_length, require_features=True)
    _, holdout = split(samples, args.holdout_tokens)
    holdout = holdout[-args.count:]
    info = server_info(args.url)
    rows = []
    warm = holdout[0]
    generate(args.url, list(warm.ids[:warm.prompt_len]), warm.output_len, args.timeout)  # excluded warmup
    before = spec_counters(server_info(args.url))
    started = time.perf_counter()
    for sample in holdout:
        row = generate(args.url, list(sample.ids[:sample.prompt_len]), sample.output_len, args.timeout)
        output = row.pop("output_ids", None) or []
        row.update({"id": sample.request_id, "prompt_tokens": sample.prompt_len, "budget": sample.output_len,
                    "matches_trace": output == list(sample.ids[sample.prompt_len:]),
                    "output_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(), "output_ids": output})
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("id", "completion_tokens", "seconds", "tokens_per_round", "matches_trace")}), flush=True)
    wall = time.perf_counter() - started
    after = spec_counters(server_info(args.url))
    tokens = sum(r["completion_tokens"] or 0 for r in rows)
    delta = {k: (after[k] - before[k]) if after.get(k) is not None and before.get(k) is not None else None for k in after}
    summary = {
        "label": args.label, "requests": len(rows), "completion_tokens": tokens, "wall_seconds": wall,
        "aggregate_tps": tokens / wall, "per_request_tps_median": statistics.median((r["completion_tokens"] or 0) / r["seconds"] for r in rows),
        "tokens_per_round_median": statistics.median(r["tokens_per_round"] for r in rows if r.get("tokens_per_round")),
        "server_delta": delta, "server_tokens_per_round": (delta["emitted"] / delta["rounds"]) if delta.get("rounds") else None,
        "matches_trace": sum(bool(r["matches_trace"]) for r in rows), "speculative": (info.get("speculative")),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
