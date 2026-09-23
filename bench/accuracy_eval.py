"""Task-accuracy benchmarks against a running server: GSM8K (exact match on the final number) and HumanEval (pass@1, tests executed).

    python3 bench/accuracy_eval.py --task gsm8k --limit 300 --label exact --accept 1:0 --output DIR
    python3 bench/accuracy_eval.py --task humaneval --label w3 --output DIR
    python3 bench/accuracy_eval.py compare DIR/exact-gsm8k.json DIR/w3-gsm8k.json

Greedy (or `--temperature T`: sampled with top_p 0.95, top_k 20 and one seed per
item shared by every arm), thinking disabled, identical prompts for every arm;
`--accept K:M` pins the acceptance rule per request (1:0 = exact, 2:1.0 = the
relaxed default). HumanEval
completions run in a subprocess with no network, a CPU and memory limit and a
timeout. Results hold per-item correctness so two arms compare paired (McNemar).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import zlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(os.environ.get("SHARAR_RESULTS", "/home/yke/sharar-results")) / "datasets" / "accuracy"


def post(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SAMPLING = {"temperature": 0.0}  # --temperature T samples with top_p 0.95, top_k 20 (the checkpoint's generation config)


def ask(url: str, content: str, max_tokens: int, accept: tuple[int, float]) -> tuple[str, dict]:
    body = {"model": "sharar", "messages": [{"role": "user", "content": content}], **SAMPLING, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "accept_top_k": accept[0], "accept_margin": accept[1], "trace": False}
    if body["temperature"] > 0:
        body["seed"] = zlib.crc32(content.encode())  # the same seed per item in every arm
    out = post(url, body)
    return out["choices"][0]["message"].get("content") or "", out.get("usage") or {}


# ---------------------------------------------------------------- GSM8K

GSM_SUFFIX = "\n\nSolve the problem step by step. On the last line write only: Answer: <number>"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_pred(text: str) -> str | None:
    tail = text.rsplit("Answer:", 1)
    src = tail[1] if len(tail) == 2 else text
    nums = NUM.findall(src)
    if not nums:
        nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return a == b


def run_gsm8k(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, it["question"] + GSM_SUFFIX, max_tokens, accept)
        pred, gold = gsm_pred(text), gsm_gold(it["answer"])
        return {"index": i, "correct": same_number(pred, gold), "pred": pred, "gold": gold, "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens"), "text_tail": text[-160:]}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- HumanEval

HE_PREFIX = ("Complete the following Python function. Reply with the complete function (including the signature and any imports it needs) "
             "in a single ```python code block and nothing else.\n\n")
BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def he_program(prompt: str, completion: str, entry_point: str) -> str:
    blocks = BLOCK.findall(completion)
    code = max(blocks, key=len) if blocks else completion
    if f"def {entry_point}" not in code:
        code = prompt + code  # the model returned a body only
    return code


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    os.setsid()


def he_check(program: str, test: str, entry_point: str) -> tuple[bool, str]:
    src = program + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "candidate.py"
        path.write_text(src)
        try:
            r = subprocess.run([sys.executable, "-I", "-S", str(path)], cwd=d, capture_output=True, text=True, timeout=15,
                               env={"PATH": "/usr/bin:/bin"}, preexec_fn=_limits)
            return r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def run_humaneval(url, items, accept, max_tokens, concurrency):
    def one(i, it):
        t0 = time.perf_counter()
        text, usage = ask(url, HE_PREFIX + it["prompt"], max_tokens, accept)
        program = he_program(it["prompt"], text, it["entry_point"])
        ok, err = he_check(program, it["test"], it["entry_point"])
        return {"index": i, "task_id": it["task_id"], "correct": ok, "error": err if not ok else "", "seconds": time.perf_counter() - t0,
                "completion_tokens": usage.get("completion_tokens"), "relaxed_tokens": usage.get("relaxed_tokens")}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda p: one(*p), enumerate(items)))


# ---------------------------------------------------------------- compare

def mcnemar(a: list[bool], b: list[bool]) -> dict:
    n01 = sum(1 for x, y in zip(a, b) if x and not y)  # a right, b wrong
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    p = min(1.0, sum(math.comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= abs(n01 - n / 2)) / 2 ** n) if n else 1.0
    return {"a_only": n01, "b_only": n10, "p": p}


def compare(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    base = runs[0]
    print(f"{'arm':<12} {'task':<10} {'n':>4} {'acc':>7} {'vs first: +/-':>14} {'p':>6} {'tok/item':>9} {'relaxed%':>9}")
    for r in runs:
        acc = sum(x["correct"] for x in r["items"]) / len(r["items"])
        m = mcnemar([x["correct"] for x in base["items"]], [x["correct"] for x in r["items"]]) if r is not base else {"a_only": 0, "b_only": 0, "p": 1.0}
        toks = sum(x["completion_tokens"] or 0 for x in r["items"]); rel = sum(x["relaxed_tokens"] or 0 for x in r["items"])
        print(f"{r['label']:<12} {r['task']:<10} {len(r['items']):>4} {100*acc:6.1f}% {m['b_only']:>6}/{m['a_only']:<7} {m['p']:6.2f} {toks/len(r['items']):9.0f} {100*rel/max(1,toks):8.2f}%")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--temperature", type=float, default=0.0, help="sample at this temperature (top_p 0.95, top_k 20, one seed per prompt shared by every arm)")
    ap.add_argument("--accept", default="1:0", help="K:M acceptance pinned on every request (1:0 exact)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.temperature > 0:
        SAMPLING.update(temperature=args.temperature, top_p=0.95, top_k=20)
    k, m = args.accept.split(":")
    accept = (int(k), float(m))
    if args.task == "gsm8k":
        items = [json.loads(l) for l in (DATA / "gsm8k_test.jsonl").read_text().splitlines() if l.strip()]
        run, max_tokens = run_gsm8k, args.max_tokens or 640
    else:
        items = [json.loads(l) for l in (DATA / "HumanEval.jsonl").read_text().splitlines() if l.strip()]
        run, max_tokens = run_humaneval, args.max_tokens or 768
    if args.limit:
        items = items[:args.limit]
    t0 = time.perf_counter()
    rows = run(args.url, items, accept, max_tokens, args.concurrency)
    wall = time.perf_counter() - t0
    acc = sum(r["correct"] for r in rows) / len(rows)
    with urllib.request.urlopen(f"{args.url}/get_server_info", timeout=10) as r:
        info = json.loads(r.read())
    result = {"label": args.label, "task": args.task, "accept": accept, "sampling": dict(SAMPLING), "n": len(rows), "accuracy": acc, "wall_seconds": wall,
              "max_tokens": max_tokens, "speculative": info.get("speculative"), "items": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}-{args.task}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"label": args.label, "task": args.task, "n": len(rows), "accuracy": round(acc, 4), "wall_s": round(wall, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
