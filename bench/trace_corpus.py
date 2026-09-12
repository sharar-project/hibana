"""Generate a request-trace corpus by replaying ShareGPT prompts over HTTP.

The server must run with `--traces`; this client only sends chat requests so
that the target regenerates every answer. Greedy by default, thinking off, a
small sampled share to exercise the sampled trace path, and one request per
run sent with `trace: false` to confirm the opt-out. No GPU imports.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def conversations(path: Path, seed: int, count: int, min_chars: int = 40, max_chars: int = 6000) -> list[dict]:
    data = json.loads(path.read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    out = []
    for conv in data:
        turns = conv.get("conversations") or []
        human = [t["value"] for t in turns if t.get("from") == "human"]
        if not human or not (min_chars <= len(human[0]) <= max_chars):
            continue
        out.append({"id": conv.get("id"), "prompt": human[0]})
        if len(out) >= count:
            break
    return out


def send(url: str, item: dict, max_tokens: int, temperature: float, trace: bool, timeout: float) -> dict:
    body = {
        "model": "sharar", "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if not trace:
        body["trace"] = False
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        usage = payload.get("usage") or {}
        return {"id": item["id"], "ok": True, "seconds": time.perf_counter() - t0, "temperature": temperature,
                "trace": trace, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": payload["choices"][0].get("finish_reason")}
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
        return {"id": item["id"], "ok": False, "seconds": time.perf_counter() - t0, "error": str(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--sampled-share", type=float, default=0.1, help="fraction sent at temperature 0.8")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--log", type=Path, required=True, help="JSONL of one row per request")
    args = ap.parse_args(argv)
    items = conversations(args.dataset, args.seed, args.count)
    rng = random.Random(args.seed + 1)
    plan = []
    for i, item in enumerate(items):
        temperature = 0.8 if rng.random() < args.sampled_share else 0.0
        plan.append((item, temperature, i != 0))  # the first request opts out of tracing
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    done = ok = tokens = 0
    with args.log.open("w") as log, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(send, args.url, item, args.max_tokens, temperature, trace, args.timeout)
                   for item, temperature, trace in plan]
        for future in futures:
            row = future.result()
            log.write(json.dumps(row) + "\n")
            log.flush()
            done += 1
            ok += row["ok"]
            tokens += row.get("completion_tokens") or 0
            if done % 20 == 0 or done == len(plan):
                elapsed = time.perf_counter() - started
                print(json.dumps({"done": done, "ok": ok, "completion_tokens": tokens, "elapsed_s": round(elapsed, 1),
                                  "tok_per_s": round(tokens / max(elapsed, 1e-9), 1)}), flush=True)
    return 0 if ok == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
