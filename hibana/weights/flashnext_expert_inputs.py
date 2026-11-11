# SPDX-License-Identifier: Apache-2.0
"""Knowledge-heavy calibration samples for Flash-Next's 3-bit routed experts.

The coding-proxy set (`w3_calibration_inputs`) routes little law, business or history through the experts, and the
weight-error calibration lost exactly that recall (MMLU-Pro law -13 pp). This adds multiple-choice knowledge
questions in the chat format the evals use, answered by the model itself:

1. `prompts`: MMLU's validation split (1,531 questions, 57 subjects) and MMLU-Pro's validation split (70), at pinned
   revisions. Any question whose normalized text appears in MMLU-Pro's test split or GPQA is dropped, so no eval
   item (nor its MMLU parent) enters. Labels are never used.
2. `generate --url`: a running Flash-Next server (the NVFP4 experts, greedy, thinking off) answers each prompt.
3. `build`: each (prompt, answer) pair is rendered with the chat template, tokenized, and packed into samples of at
   most --max-tokens; the coding-proxy samples (`--base`) come first, unchanged.

    python3 -m hibana.weights.flashnext_expert_inputs prompts --output prompts.json
    python3 -m hibana.weights.flashnext_expert_inputs generate --prompts prompts.json --url http://127.0.0.1:30001 --output answers.json
    python3 -m hibana.weights.flashnext_expert_inputs build --answers answers.json --base calibration-inputs.json \\
        --tokenizer SNAPSHOT --output expert-inputs.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MMLU = ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/validation-00000-of-00001.parquet")
MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be")
LETTERS = "ABCDEFGHIJ"


def _parquet(repo: str, revision: str, path: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("the calibration prompts' parquet files need pyarrow: `uv sync --group eval`") from exc
    from huggingface_hub import hf_hub_download
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=revision)).to_pylist()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def choice_prompt(question: str, options: list[str]) -> str:
    """The prompt sharar-eval's hard_eval sends for multiple choice (format only)."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n\n{opts}\n\nThink step by step, then finish with a line containing only: Answer: <letter>"


def prompts(args) -> None:
    held_out = {_norm(r["question"]) for r in _parquet(*MMLU_PRO, "data/test-00000-of-00001.parquet")}
    if args.exclude_jsonl:
        for path in args.exclude_jsonl:
            for line in filter(None, Path(path).read_text().split("\n")):
                row = json.loads(line)
                held_out.add(_norm(row.get("question") or row.get("Question") or ""))
    held_out.discard("")
    items, dropped = [], 0
    for r in _parquet(*MMLU):
        if _norm(r["question"]) in held_out or any(_norm(r["question"]) in h for h in held_out if len(r["question"]) > 40 and len(h) < 4 * len(r["question"])):
            dropped += 1
            continue
        items.append({"source": f"mmlu-val/{r['subject']}", "prompt": choice_prompt(r["question"], r["choices"])})
    for r in _parquet(*MMLU_PRO, "data/validation-00000-of-00001.parquet"):
        items.append({"source": f"mmlupro-val/{r['category']}", "prompt": choice_prompt(r["question"], r["options"])})
    out = {"version": "flashnext-expert-prompts-v1", "mmlu": list(MMLU), "mmlu_pro": list(MMLU_PRO), "dropped_overlap": dropped, "items": items}
    args.output.write_text(json.dumps(out))
    print(f"{len(items)} prompts ({dropped} MMLU questions dropped as overlapping the eval sets) -> {args.output}")


def prompts_sharegpt(args) -> None:
    """First user turns of ShareGPT conversations (the drafter study's train split): `--english` English ones and
    `--other` in other languages (mostly non-ASCII text), 20-4000 characters, seeded; for reasoning traces."""
    import random
    rows = json.loads(args.sharegpt.read_text())
    english, other = [], []
    for r in rows:
        turns = r.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        q = turns[0]["value"].strip()
        if not 20 <= len(q) <= 4000:
            continue
        ascii_share = sum(ch.isascii() for ch in q) / len(q)
        (english if ascii_share > 0.97 else other).append({"source": f"sharegpt/{'en' if ascii_share > 0.97 else 'other'}", "id": r.get("id"), "prompt": q})
    rng = random.Random(args.seed)
    items = rng.sample(english, min(args.english, len(english))) + rng.sample(other, min(args.other, len(other)))
    args.output.write_text(json.dumps({"version": "flashnext-expert-prompts-sharegpt-v1", "sharegpt": str(args.sharegpt),
                                       "sharegpt_sha256": hashlib.sha256(args.sharegpt.read_bytes()).hexdigest(), "seed": args.seed, "items": items}))
    print(f"{len(items)} prompts ({min(args.english, len(english))} English of {len(english)}, {min(args.other, len(other))} other of {len(other)}) -> {args.output}")


def generate(args) -> None:
    items = json.loads(args.prompts.read_text())["items"]

    def ask(it):
        body = {"model": "sharar", "messages": [{"role": "user", "content": it["prompt"]}], "temperature": args.temperature,
                "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": args.thinking}, "trace": False}
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            message = json.load(r)["choices"][0]["message"]
        return dict(it, answer=message.get("content") or "", reasoning=message.get("reasoning_content") or "")

    # Answers append to OUTPUT.partial.jsonl as they finish, so an interrupted run resumes where it stopped.
    import threading
    partial = args.output.with_name(args.output.name + ".partial.jsonl")
    done = {}
    if partial.is_file():
        for line in filter(None, partial.read_text().split("\n")):
            row = json.loads(line)
            done[row["prompt"]] = row
    lock = threading.Lock()

    def ask_saved(it):
        if it["prompt"] in done:
            return done[it["prompt"]]
        row = ask(it)
        with lock, partial.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    with ThreadPoolExecutor(args.concurrency) as pool:
        answered = list(pool.map(ask_saved, items))
    args.output.write_text(json.dumps({"version": "flashnext-expert-answers-v1", "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                                       "url": args.url, "max_tokens": args.max_tokens, "items": answered}))
    print(f"{len(answered)} answers -> {args.output}")


def build(args) -> None:
    from hibana.api.chat_template import ChatTemplate
    from hibana.tokenizer import SnapshotTokenizer
    tmpl, tok = ChatTemplate(args.tokenizer), SnapshotTokenizer(args.tokenizer)
    base = json.loads(args.base.read_text())
    answers = json.loads(args.answers.read_text())["items"]
    samples, current, sources = [], [], []
    for it in answers:
        if not (it["answer"].strip() or it.get("reasoning", "").strip()):
            continue
        reply = {"role": "assistant", "content": it["answer"]}
        if it.get("reasoning"):  # a thinking trace: rendered as the model writes it (<think> ... </think> then the answer)
            reply["reasoning_content"] = it["reasoning"]
        text = tmpl.render([{"role": "user", "content": it["prompt"]}, reply], add_generation_prompt=False,
                           enable_thinking=bool(it.get("reasoning")))
        ids = tok.encode(text)[:args.max_tokens]
        if current and len(current) + len(ids) > args.max_tokens:
            samples.append({"source": ",".join(sorted(set(sources))), "input_ids": current})
            current, sources = [], []
        current = current + ids
        sources.append(it["source"].split("/")[0])
    if current:
        samples.append({"source": ",".join(sorted(set(sources))), "input_ids": current})
    out = dict(base, version="flashnext-expert-inputs-v1", base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
               answers_sha256=hashlib.sha256(args.answers.read_bytes()).hexdigest(), samples=base["samples"] + samples)
    args.output.write_text(json.dumps(out))
    print(f"{len(base['samples'])} base + {len(samples)} knowledge samples, "
          f"{sum(len(s['input_ids']) for s in out['samples'])} tokens -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-jsonl", nargs="*", help="further eval sets whose questions to drop (JSONL with 'question')")
    sg = sub.add_parser("prompts-sharegpt", help="ShareGPT first turns, English and other languages, for reasoning traces")
    sg.add_argument("--sharegpt", type=Path, required=True)
    sg.add_argument("--english", type=int, default=1500)
    sg.add_argument("--other", type=int, default=1500)
    sg.add_argument("--seed", type=int, default=0)
    sg.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", type=Path, required=True)
    g.add_argument("--url", default="http://127.0.0.1:30001")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--thinking", action="store_true", help="enable thinking and keep the trace (reasoning_content)")
    g.add_argument("--temperature", type=float, default=0.0)
    b = sub.add_parser("build")
    b.add_argument("--answers", type=Path, required=True)
    b.add_argument("--base", type=Path, required=True, help="the frozen coding-proxy samples (w3_calibration_inputs)")
    b.add_argument("--tokenizer", type=Path, required=True, help="the target snapshot (its chat template)")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    {"prompts": prompts, "prompts-sharegpt": prompts_sharegpt, "generate": generate, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
