#!/usr/bin/env python3
"""GSM8K accuracy through an OpenAI-compatible chat endpoint (stdlib only).

Downloads the GSM8K test split once (openai/grade-school-math on GitHub),
sends the first N problems with a fixed prompt, greedy, thinking off, and
scores the number after "Final answer:" (falling back to the last number).
The same script runs against any engine so accuracies are comparable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
PROMPT = "Solve the following math problem. Think step by step, then finish with a line of the form 'Final answer: <number>'.\n\n{q}"


def load(n: int, cache: Path) -> list[dict]:
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    rows = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    return rows[:n]


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def predicted(text: str) -> str | None:
    m = re.findall(r"Final answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    if not m:
        return None
    v = m[-1].replace(",", "")
    return v[:-2] if v.endswith(".0") else v


def ask(base_url: str, model: str, q: str, max_tokens: int, timeout: float) -> tuple[str, int, float]:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
        "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or "", int(d["usage"]["completion_tokens"]), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30001")
    ap.add_argument("--model", default="RadixArk/Qwen3.8-27B-NVFP4")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--label", default="engine")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(args.n, Path.home() / ".cache" / "sharar" / "gsm8k_test.jsonl")
    results = [None] * len(rows)
    lock = threading.Lock()
    idx = [0]

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(rows):
                return
            try:
                text, toks, secs = ask(args.base_url, args.model, rows[i]["question"], args.max_tokens, 900)
                pred = predicted(text)
            except Exception as exc:  # noqa: BLE001
                text, toks, secs, pred = f"ERROR {exc}", 0, 0.0, None
            results[i] = {"i": i, "gold": gold(rows[i]["answer"]), "pred": pred, "tokens": toks, "seconds": secs, "text": text}

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    correct = sum(1 for r in results if r["pred"] is not None and r["pred"] == r["gold"])
    toks = sum(r["tokens"] for r in results)
    errors = sum(1 for r in results if r["text"].startswith("ERROR"))
    print(f"[{args.label}] GSM8K first {len(rows)}: accuracy {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%  "
          f"({toks} completion tokens in {wall:.0f}s = {toks / wall:.1f} tok/s at concurrency {args.concurrency}; errors {errors})")
    if args.out:
        Path(args.out).write_text(json.dumps({"label": args.label, "n": len(rows), "correct": correct, "tokens": toks, "wall": wall, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
