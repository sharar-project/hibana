"""ShareGPT serving benchmark for OpenAI-compatible engines on one GB10.

Same prompts, same buckets, same concurrency levels for every engine. Prompts
are the first human turn of ShareGPT conversations; the requested output
budget is the length of the recorded assistant reply (capped), with natural
stopping, so engines generate the same *kind* of work and we report what they
actually produced. Buckets by prompt length in this model's tokens:

  short   in <= 256       out cap 256
  medium  256 < in <= 1024  out cap 512
  long    1024 < in <= 4096 out cap 512

Concurrency is a closed loop: C requests in flight until the bucket's prompt
list is exhausted. Metrics per (bucket, concurrency): aggregate output
tokens/s over the makespan, TTFT p50/p95, time per output token (TPOT) mean,
per-request completion p50, generated tokens, and any failures.

Runs inside the runtime image (needs the model tokenizer) with host network:
  python3 bench/sharegpt_bench.py --url http://127.0.0.1:30001 --engine sharar --output DIR
  python3 bench/sharegpt_bench.py --freeze DIR/inputs.json   # tokenize + sample once, then reuse
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKETS = {"short": (0, 256, 256), "medium": (256, 1024, 512), "long": (1024, 4096, 512)}
DEFAULT_DATASET = Path("/results/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")


def freeze(dataset: Path, out: Path, per_bucket: int, seed: int) -> dict:
    from sharar import config
    from sharar.tokenizer import SnapshotTokenizer

    tok = SnapshotTokenizer(config.snapshot_dir())
    convs = json.loads(dataset.read_text())
    rng = random.Random(seed)
    rng.shuffle(convs)
    picked = {name: [] for name in BUCKETS}
    for conv in convs:
        turns = conv.get("conversations") or []
        if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
            continue
        prompt, reply = turns[0]["value"].strip(), turns[1]["value"].strip()
        if not prompt or not reply:
            continue
        n_in = len(tok.encode(prompt))
        n_out = len(tok.encode(reply))
        for name, (lo, hi, cap) in BUCKETS.items():
            if lo < n_in <= hi and len(picked[name]) < per_bucket and n_out >= 16:
                picked[name].append({"id": f"{name}-{len(picked[name])}", "prompt": prompt, "input_tokens": n_in,
                                     "reply_tokens": n_out, "max_tokens": min(n_out, cap)})
                break
        if all(len(v) >= per_bucket for v in picked.values()):
            break
    frozen = {"dataset": str(dataset), "seed": seed, "per_bucket": per_bucket, "buckets": picked,
              "model": config.MODEL_ID, "revision": config.MODEL_REVISION}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, indent=1))
    for name, items in picked.items():
        print(f"{name}: {len(items)} prompts, input tokens median {statistics.median(i['input_tokens'] for i in items):.0f}, "
              f"max_tokens median {statistics.median(i['max_tokens'] for i in items):.0f}")
    return frozen


def stream_chat(url: str, model: str, prompt: str, max_tokens: int, engine: str, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body["chat_template_kwargs"] = {"enable_thinking": False}
    if engine == "sharar":
        body["trace"] = False  # benchmark traffic stays out of the trace sink
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    first = None
    chunks = 0
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content") or ""
                reasoning = (choice.get("delta") or {}).get("reasoning_content") or ""
                if delta or reasoning:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
                    text.append(delta or reasoning)
    t1 = time.perf_counter()
    out_tokens = (usage or {}).get("completion_tokens")
    return {"ttft": (first or t1) - t0, "completion": t1 - t0, "chunks": chunks, "text": "".join(text),
            "completion_tokens": out_tokens, "prompt_tokens": (usage or {}).get("prompt_tokens")}


def count_tokens(tok, text: str) -> int:
    return len(tok.encode(text)) if text else 0


def run_cell(url, model, engine, items, concurrency, tok, timeout):
    results = [None] * len(items)
    lock = threading.Lock()
    index = {"next": 0}

    def worker():
        while True:
            with lock:
                i = index["next"]
                if i >= len(items):
                    return
                index["next"] = i + 1
            item = items[i]
            try:
                r = stream_chat(url, model, item["prompt"], item["max_tokens"], engine, timeout)
                if r["completion_tokens"] is None:
                    r["completion_tokens"] = count_tokens(tok, r["text"])
                r["error"] = None
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {exc}", "ttft": None, "completion": None, "completion_tokens": 0, "text": ""}
            r["id"] = item["id"]
            results[i] = r

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    makespan = time.perf_counter() - t0
    ok = [r for r in results if r["error"] is None and r["completion_tokens"]]
    gen = sum(r["completion_tokens"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok)
    tpots = [(r["completion"] - r["ttft"]) / (r["completion_tokens"] - 1) for r in ok if r["completion_tokens"] > 1]
    pct = lambda xs, p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None
    summary = {
        "requests": len(items), "failed": len(items) - len(ok), "generated_tokens": gen, "makespan_s": round(makespan, 3),
        "output_tok_s": round(gen / makespan, 2) if makespan else None,
        "ttft_p50_s": round(pct(ttfts, 0.5), 3) if ttfts else None, "ttft_p95_s": round(pct(ttfts, 0.95), 3) if ttfts else None,
        "tpot_mean_ms": round(1000 * statistics.mean(tpots), 2) if tpots else None,
        "completion_p50_s": round(statistics.median(r["completion"] for r in ok), 3) if ok else None,
        "tokens_per_request_mean": round(gen / len(ok), 1) if ok else None,
    }
    return summary, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--freeze", type=Path, help="write the frozen prompt set here and exit")
    ap.add_argument("--inputs", type=Path, help="frozen prompt set (from --freeze)")
    ap.add_argument("--per-bucket", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--url")
    ap.add_argument("--engine", choices=("sharar", "sglang", "atlas"), default="sharar")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--buckets", nargs="+", default=list(BUCKETS))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.freeze:
        freeze(args.dataset, args.freeze, args.per_bucket, args.seed)
        return
    from sharar import config
    from sharar.tokenizer import SnapshotTokenizer

    tok = SnapshotTokenizer(config.snapshot_dir())
    frozen = json.loads(args.inputs.read_text())
    model = args.model or frozen["model"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.output / "requests.jsonl").open("w") as raw:
        for name in args.buckets:
            items = frozen["buckets"][name]
            for _ in range(args.warmup):
                stream_chat(args.url, model, items[0]["prompt"], min(32, items[0]["max_tokens"]), args.engine, args.timeout)
            for c in args.concurrency:
                summary, results = run_cell(args.url, model, args.engine, items, c, tok, args.timeout)
                summary.update({"engine": args.engine, "bucket": name, "concurrency": c})
                rows.append(summary)
                print(json.dumps(summary), flush=True)
                for r in results:
                    raw.write(json.dumps({**r, "bucket": name, "concurrency": c, "engine": args.engine}) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"engine": args.engine, "url": args.url, "model": model, "rows": rows,
                                                          "inputs": str(args.inputs), "recorded": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))


if __name__ == "__main__":
    main()
