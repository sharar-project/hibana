# SPDX-License-Identifier: Apache-2.0
"""Greedy identity of the speculative engine across two builds.

Generates the round profiler's prompts plus JSON, chat, multilingual and a needle (passkey) prompt at batch one, and
the first four together at batch four, with exact and relaxed (3:2.0) acceptance, each twice in one process (the
second pass shows whether a build repeats itself), and writes every output's token ids.

    SHARAR_TARGET=flashnext python3 -m profiling.ab_identity --out a.json      # on build A, then on build B
    python3 -m profiling.ab_identity --compare a.json b.json

The needle prompt is the first row of a passkey file (`--passkey`, a JSON list of {"ids": [...]}), when given.
"""
import argparse
import hashlib
import json


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path))["outputs"], json.load(open(b_path))["outputs"]
    same = 0
    for key in sorted(set(a) & set(b)):
        x, y = a[key][0], b[key][0]
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))
        same += first is None
        print(f"{key:32s} {'identical' if first is None else f'differs at token {first}'}"
              f"  (repeatable: {a[key][0] == a[key][1]} / {b[key][0] == b[key][1]})")
    print(f"identical: {same} of {len(set(a) & set(b))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--spec", type=int, default=4)
    ap.add_argument("--passkey", default=None)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    from sharar.engine.run import _load
    from profiling.flashnext_round import EXTRA_PROMPTS, PROMPTS, _params, chat_ids
    model, ex, tok = _load(graphs=True, spec=args.spec)
    texts = {**PROMPTS, **EXTRA_PROMPTS,
             "chat": "What are three practical ways to sleep better when working late shifts? Be concise but specific.",
             "multilingual": "Explique en français, puis en espagnol, la différence entre un processus et un thread."}
    prompts = {k: chat_ids(tok, v) for k, v in texts.items()}
    if args.passkey:
        prompts["needle"] = json.load(open(args.passkey))[0]["ids"]
    out = {}
    for accept in (None, "3:2.0"):
        params = _params(args.tokens, accept)
        for _ in range(2):
            for name, ids in prompts.items():
                ex.flush_cache()
                out.setdefault(f"b1/{accept or 'exact'}/{name}", []).append(ex.generate(ids, params).output_ids[:])
            names = list(texts)[:4]
            ex.flush_cache()
            for n, r in zip(names, ex.generate_batch([(prompts[n], params) for n in names])):
                out.setdefault(f"b4/{accept or 'exact'}/{n}", []).append(r.output_ids[:])
    summary = {k: {"repeatable": v[0] == v[1], "sha": hashlib.sha256(json.dumps(v[0]).encode()).hexdigest()[:12], "n": len(v[0])}
               for k, v in out.items()}
    for k, v in summary.items():
        print(k, v, flush=True)
    with open(args.out, "w") as f:
        json.dump({"outputs": out, "summary": summary}, f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
