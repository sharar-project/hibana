"""Served speed of the 27B on one Spark or two (docs/tensor-parallel.md): what a client sees over HTTP.

Start the server first (`sharar serve 27b [--tp 2]`; `SHARAR_TP_ALLREDUCE=nccl` for NCCL's all-reduce
everywhere), then `python3 profiling/tp_serve_bench.py --label NAME --out results/tp-bench/NAME.json`. Requests
go to `/generate` with greedy sampling and the server's acceptance default (the profile's). Four phases:

- `single`: seven short prompts (chat, code, math, multilingual, prose, structured, a 12K-character code prompt), one
  at a time, up to 512 tokens each: decode tok/s from the first streamed token to the last, and a hash of the
  outputs (equal hashes: the same greedy tokens);
- `batch4`, `batch16`: 4 and 16 concurrent requests of 384 tokens each (`ignore_eos`) over the family prompts'
  questions: aggregate tok/s from the first submission to the last completion;
- `long`: the three 10K-token benchmark prompts (`hibana/benchmark_workloads.py`: code, design, reasoning), one at
  a time, 1,000 tokens each (`ignore_eos`): time to first token and decode tok/s.

Each phase also records the server's speculative rounds over it (tokens per round, ms per round).

`--api openai` measures an OpenAI-compatible server instead (vLLM: `/v1/completions`, `/tokenize`, its `/metrics`
for tokens per round) with the same prompts, token ids for the long prompts, greedy sampling and `ignore_eos`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hibana.benchmark_workloads import WORKLOADS, frozen_source, materialize  # noqa: E402

CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SINGLE = {
    "chat": "What are three practical ways to sleep better when working late shifts? Be concise but specific.",
    "code": "Write a Python function that merges overlapping intervals, with type hints, a docstring and pytest tests.",
    "math": "A train leaves at 9:40 at 72 km/h; a second leaves the same station at 10:05 at 90 km/h on the same track. "
            "When and where does the second catch up? Show every step.",
    "multilingual": "Explique en français, puis en espagnol, la différence entre un processus et un thread, avec un "
                    "exemple concret dans chaque langue.",
    "prose": "Write a short story (about 300 words) about a lighthouse keeper who receives a letter from the future.",
    "structured": "Return a JSON object describing three fictional employees with fields name, role, start_date "
                  "(ISO 8601), skills (list) and manager (name or null). Only JSON.",
    "long_code": "Here is a source file:\n\n" + frozen_source()[:12000] + "\n\nSummarize what this code does, list "
                 "its main classes and their responsibilities, and point out two possible bugs.",
}


def call(url: str, path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


OPENAI_MODEL = None  # the served model's id with --api openai


def stream(url: str, body: dict) -> dict:
    """One streamed request: start, first-token and end times, the number of output tokens and the output (token ids,
    or text from an OpenAI server)."""
    if OPENAI_MODEL:
        path = "/v1/completions"
        body = {"model": OPENAI_MODEL, "prompt": body.get("input_ids") or body["text"], "max_tokens": body["max_tokens"],
                "temperature": 0, "ignore_eos": body.get("ignore_eos", False), "stream_options": {"include_usage": True}}
    else:
        path = "/generate"
    req = urllib.request.Request(url + path, data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, out, n = None, [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if OPENAI_MODEL:
                text = "".join(c.get("text", "") for c in event.get("choices", []))
                if text:
                    first = first or time.perf_counter()
                    out.append(text)
                if event.get("usage"):
                    n = event["usage"]["completion_tokens"]
            else:
                if event["output_ids"] and first is None:
                    first = time.perf_counter()
                out = event["output_ids"]
                n = len(out)
    end = time.perf_counter()
    return {"start": t0, "first": first, "end": end, "n": n, "out": "".join(out) if OPENAI_MODEL else out}


def decode_rate(r: dict) -> float:
    return (r["n"] - 1) / (r["end"] - r["first"])


def spec(url: str) -> dict:
    if not OPENAI_MODEL:
        return call(url, "/get_server_info").get("spec_stats") or {}
    with urllib.request.urlopen(url + "/metrics", timeout=60) as r:
        metrics = {}
        for line in r.read().decode().splitlines():
            name = line.split("{")[0].split(" ")[0]
            if name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
                metrics[name] = metrics.get(name, 0) + float(line.rsplit(" ", 1)[1])
    drafts = metrics.get("vllm:spec_decode_num_drafts_total", 0)
    # each draft (a verification round) emits its accepted tokens and one from the target
    return {"rounds": drafts, "emitted": metrics.get("vllm:spec_decode_num_accepted_tokens_total", 0) + drafts}


def spec_delta(before: dict, after: dict) -> dict:
    rounds = after.get("rounds", 0) - before.get("rounds", 0)
    if rounds <= 0:
        return {}
    out = {"rounds": rounds, "tokens_per_round": (after["emitted"] - before.get("emitted", 0)) / rounds}
    if "round_ms" in after:
        out["round_ms"] = (after["round_ms"] * after["rounds"] - before.get("round_ms", 0) * before.get("rounds", 0)) / rounds
    return out


def concurrent(url: str, bodies: list[dict]) -> list[dict]:
    out: list[dict | None] = [None] * len(bodies)

    def run(i):
        out[i] = stream(url, bodies[i])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(r is None for r in out):
        raise RuntimeError("a concurrent request failed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:30001")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", default="single,batch4,batch16,long")
    parser.add_argument("--api", choices=("sharar", "openai"), default="sharar")
    args = parser.parse_args()
    global OPENAI_MODEL
    url, phases = args.url, args.phases.split(",")
    if args.api == "openai":
        OPENAI_MODEL = call(url, "/v1/models")["data"][0]["id"]
        result = {"label": args.label, "engine": "openai", "model": OPENAI_MODEL, "phases": {}}
    else:
        info = call(url, "/get_server_info")
        result = {"label": args.label, "engine": "sharar", "tp_size": info.get("tp_size"), "w3": bool(info.get("w3_target_identity")),
                  "max_running": info.get("max_running_requests"),
                  "accept_default": (info.get("speculative") or {}).get("accept_default"), "phases": {}}
    stream(url, {"text": CHAT.format("Say hello."), "max_tokens": 32})  # warm-up

    if "single" in phases:
        before, rows = spec(url), {}
        for name, prompt in SINGLE.items():
            r = stream(url, {"text": CHAT.format(prompt), "max_tokens": 512})
            rows[name] = {"tokens": r["n"], "decode_tok_s": decode_rate(r),
                          "ttft_s": r["first"] - r["start"], "sha": hashlib.sha256(str(r["out"]).encode()).hexdigest()[:12]}
            print(f"  single {name:>12}: {r['n']:4d} tokens, {rows[name]['decode_tok_s']:6.1f} tok/s", flush=True)
        result["phases"]["single"] = {"prompts": rows, "median_tok_s": statistics.median(x["decode_tok_s"] for x in rows.values()),
                                      "spec": spec_delta(before, spec(url))}

    family = json.loads((ROOT / "profiling/workloads/family_v1.json").read_text())["tasks"]
    for n in (4, 16):
        if f"batch{n}" not in phases:
            continue
        bodies = [{"text": CHAT.format(family[i % len(family)]["question"]), "max_tokens": 384, "ignore_eos": True}
                  for i in range(n)]
        before = spec(url)
        rs = concurrent(url, bodies)
        wall = max(r["end"] for r in rs) - min(r["start"] for r in rs)
        tokens = sum(r["n"] for r in rs)
        result["phases"][f"batch{n}"] = {"requests": n, "tokens": tokens, "wall_s": wall, "aggregate_tok_s": tokens / wall,
                                         "spec": spec_delta(before, spec(url))}
        print(f"  batch{n}: {tokens} tokens in {wall:.1f} s, {tokens / wall:6.1f} tok/s aggregate", flush=True)

    if "long" in phases:
        encode = lambda text: call(url, "/tokenize", {"text": text, "prompt": text, "model": OPENAI_MODEL,  # noqa: E731
                                                      "add_special_tokens": False})["tokens"]
        before, rows = spec(url), {}
        for case in WORKLOADS:
            ids = materialize(encode, case, 10_000)
            r = stream(url, {"input_ids": ids, "max_tokens": 1000, "ignore_eos": True})
            rows[case] = {"input_tokens": len(ids), "ttft_s": r["first"] - r["start"], "decode_tok_s": decode_rate(r),
                          "e2e_s": r["end"] - r["start"]}
            print(f"  long {case:>9}: TTFT {rows[case]['ttft_s']:5.2f} s, {rows[case]['decode_tok_s']:6.1f} tok/s", flush=True)
        result["phases"]["long"] = {"prompts": rows, "spec": spec_delta(before, spec(url))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
