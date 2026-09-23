"""Test-time scaling on MATH-500: one greedy answer against self-consistency
(n sampled answers from one prefill, majority vote over the boxed answers).

    python3 bench/tts_eval.py --url http://127.0.0.1:30001 --label greedy --n 1 --output DIR
    python3 bench/tts_eval.py --url ... --label sc4 --n 4 --temperature 0.7 --output DIR

Thinking disabled, "reason step by step, final answer in \\boxed{}". A seeded
subset (--limit) spans every level. n=1 runs four problems at a time; n>1 runs
one problem at a time with its n choices decoding side by side, so both fill
the engine's four sequences and wall-clock times compare. Answers are compared
after the standard MATH normalization (Hendrycks et al.); pass@n (any sample
correct) is reported as the ceiling a verifier could reach.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

DATASET = Path("/home/yke/sharar-results/datasets/math500/test.jsonl")
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} (balanced braces)."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _fix_fracs(s):
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        elif len(sub) >= 2:
            a, b = sub[0], sub[1]
            out += "{" + a + "}{" + b + "}" + sub[2:] if b != "{" else "{" + a + "}" + sub[1:]
        else:
            return s
    return out


def _fix_sqrt(s):
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def normalize(s: str | None) -> str | None:
    """The MATH benchmark's string normalization (strip_string)."""
    if s is None:
        return None
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if "=" in s and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s).replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    if re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", s) is None and re.fullmatch(r"(\d+)/(\d+)", s):
        a, b = s.split("/")
        s = "\\frac{" + a + "}{" + b + "}"
    return s.rstrip(".")


def ask(url: str, problem: str, n: int, temperature: float, max_tokens: int, seed: int) -> tuple[list[str], dict, float]:
    body = {"model": "sharar", "messages": [{"role": "user", "content": PROMPT.format(problem=problem)}],
            "max_tokens": max_tokens, "temperature": temperature, "n": n, "seed": seed, "trace": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if temperature > 0:
        body.update(top_p=0.95, top_k=20, accept_top_k=1)  # exact sampling
    t = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.load(r)
    return [c["message"]["content"] or "" for c in data["choices"]], data["usage"], time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0, help="subset and sampling seed")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    items = [json.loads(line) for line in DATASET.read_text().splitlines()]
    random.Random(args.seed).shuffle(items)
    items = items[:args.limit]
    concurrency = 4 if args.n == 1 else 1

    def run(item):
        i = items.index(item)
        texts, usage, seconds = ask(args.url, item["problem"], args.n, args.temperature, args.max_tokens, args.seed * 100_000 + i * 16)
        answers = [normalize(boxed(t)) for t in texts]
        votes = Counter(a for a in answers if a is not None)
        voted = votes.most_common(1)[0][0] if votes else None
        truth = normalize(item["answer"])
        return {"id": item["unique_id"], "level": item["level"], "truth": truth, "answers": answers, "voted": voted,
                "correct": voted == truth, "any_correct": truth in answers, "completion_tokens": usage["completion_tokens"],
                "seconds": round(seconds, 2)}

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        rows = list(pool.map(run, items))
    wall = time.time() - start
    acc = sum(r["correct"] for r in rows) / len(rows)
    result = {"label": args.label, "n": args.n, "temperature": args.temperature, "items": len(rows), "accuracy": acc,
              "pass_at_n": sum(r["any_correct"] for r in rows) / len(rows), "wall_seconds": round(wall, 1),
              "completion_tokens": sum(r["completion_tokens"] for r in rows),
              "by_level": {lvl: round(sum(r["correct"] for r in rows if r["level"] == lvl) / max(1, sum(r["level"] == lvl for r in rows)), 3)
                           for lvl in sorted({r["level"] for r in rows})},
              "rows": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
