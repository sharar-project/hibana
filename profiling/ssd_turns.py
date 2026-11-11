"""Concurrent multi-turn conversations against a server, the workload an SSD-tier restore stalls.

    HIBANA_KV_POOL_TOKENS=49152 uv run sharar serve 27b --fast --port 30000      # small pool: conversations get evicted
    uv run python profiling/ssd_turns.py --url http://127.0.0.1:30000 --label async

`--convs` conversations (the frozen 10K-token prompts of `profiling/inputs.py`) run at once, each in its own thread.
Phase 1, not measured: every conversation's first turn (its 10K-token prefill). Phase 2, measured: `--turns` - 1 more
turns each; a turn appends the previous answer and a new question and generates `--new` tokens greedily, then the
conversation "thinks" `--think` seconds (a tool call) before the next. With a small KV pool the others push a
thinking conversation's cached prefix out of the GPU cache, so its next turn restores it from the SSD (or re-prefills
without the tier).

Prints one JSON line: phase 2's wall time and output tokens/s, time to first token of the later turns, and the gaps
between streamed events of every request after its first token (a restore or a prefill chunk that pauses the batch
is a long gap in everyone else's stream), with the server's restore counters over phase 2.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.inputs import tasks  # noqa: E402

QUESTIONS = ["Summarize the specification in three sentences.", "Which records failed, and what do they share?",
             "What would you change first, and why?", "List the three riskiest assumptions.",
             "Estimate the effort in days, with a short justification.", "Write the acceptance criteria."]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--convs", type=int, default=6)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--think", type=float, default=1.0, help="seconds between a conversation's turns")
    ap.add_argument("--new", type=int, default=192, help="tokens generated per turn")
    args = ap.parse_args()

    def post(path, body):
        req = urllib.request.Request(args.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=60) as r:
            return json.loads(r.read())

    def encode(text):
        return post("/tokenize", {"text": text})["tokens"]

    def stream(ids):
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": args.new, "temperature": 0, "ignore_eos": True},
                "stream": True}
        req = urllib.request.Request(args.url + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0, times, out = time.perf_counter(), [], []
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                got = json.loads(line[6:]).get("output_ids") or []
                if len(got) > len(out):
                    times.append(time.perf_counter())
                    out = got
        return times[0] - t0, times, out

    prompts = [t["input_ids"] for t in tasks(encode)][:args.convs]
    if len(prompts) < args.convs:
        raise SystemExit(f"only {len(prompts)} frozen prompts")
    questions = [encode(f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                 for q in QUESTIONS]
    convs, lock = list(prompts), threading.Lock()
    later, gaps, out_tokens = [], [], [0]

    def turn(c, k, measured):
        ids = convs[c] + (questions[(k - 1) % len(questions)] if k else [])
        ttft, times, out = stream(ids)
        if measured:
            with lock:
                later.append(ttft)
                gaps.extend(b - a for a, b in zip(times, times[1:]))
                out_tokens[0] += len(out)
        convs[c] = ids + out

    def phase1(c):
        time.sleep(0.5 * c)  # stagger the first prefills
        turn(c, 0, False)

    def phase2(c):
        time.sleep(0.3 * c)
        for k in range(1, args.turns):
            time.sleep(args.think)  # a tool call: meanwhile the others push this conversation out of the GPU cache
            turn(c, k, True)

    def run(target):
        threads = [threading.Thread(target=target, args=(c,)) for c in range(args.convs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    run(phase1)
    sched = lambda info: info.get("scheduler") or {}  # noqa: E731
    ssd = lambda info: (sched(info).get("cache") or {}).get("ssd") or {}  # noqa: E731
    info0, t0 = get("/get_server_info"), time.perf_counter()
    run(phase2)
    total, info1 = time.perf_counter() - t0, get("/get_server_info")
    before, after = ssd(info0), ssd(info1)
    g = sorted(gaps)
    q = lambda p: g[min(len(g) - 1, int(p * len(g)))]  # noqa: E731
    row = {"label": args.label, "convs": args.convs, "turns": args.turns, "total_s": round(total, 1),
           "out_tok_s": round(out_tokens[0] / total, 1), "later_ttft_p50": round(statistics.median(later), 3),
           "later_ttft_p90": round(sorted(later)[int(0.9 * len(later))], 3), "later_ttft_max": round(max(later), 3),
           "gap_p50_ms": round(1000 * q(0.5), 1), "gap_p99_ms": round(1000 * q(0.99), 1), "gap_max_ms": round(1000 * g[-1], 1)}
    row["prefill_tokens"] = sched(info1).get("prefill_tokens", 0) - sched(info0).get("prefill_tokens", 0)  # new tokens and re-prefills
    for k in ("restores", "restore_fails", "async_restores", "restore_blocked_ms_total", "evict_dropped"):
        if k in after:
            row[k] = round(after[k] - before.get(k, 0), 1)
    if "restore_ms_avg" in after:
        row["restore_ms_avg"] = after["restore_ms_avg"]
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
