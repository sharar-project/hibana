# SPDX-License-Identifier: Apache-2.0
"""Calibration samples for every target's 3-bit copies: chat prompts answered by the target itself, rendered with
its own chat template. Every capture reads them with `load_samples`.

1. `prompts`: first user turns of Open-PerfectBlend at a pinned revision (math, code, chat, instruction following and
   arena conversations), an equal seeded draw from each of its sources, plus MMLU's and MMLU-Pro's validation
   questions (`--knowledge`) for the routed experts' recall of law, business or history, which a math-and-code set
   routes little of (Flash-Next's experts lost MMLU-Pro law -13 pp on a coding set). A prompt that shares a 13-word
   span with an eval set (`--evals`: the sharar-eval datasets, `sharar-eval datasets fetch`), or equals a shorter
   eval question, is dropped; labels and reference answers are never used. The prompts file serves every target.
2. `generate --url`: the target's server answers each prompt with thinking on, sampled. Serve the target `--exact`
   (its own distribution, not the 3-bit copies being fitted). Answers append to OUTPUT.partial.jsonl as they finish,
   so an interrupted run resumes.
3. `build --target`: each answered turn rendered with the target's chat template (its reasoning in `<think>`, then
   the answer) and tokenized with its tokenizer; a turn longer than --max-tokens is cut, shorter ones are packed
   whole into samples of up to --max-tokens.

    python3 -m sharar.weights.calibration_inputs prompts --evals $SHARAR_RESULTS/datasets --output prompts.json
    sharar serve nemotron --exact --max-batch 16 &
    python3 -m sharar.weights.calibration_inputs generate --prompts prompts.json --url http://127.0.0.1:30005 --output answers-nemotron.json
    SHARAR_HF_CACHE=... python3 -m sharar.weights.calibration_inputs build --target nemotron --answers answers-nemotron.json \\
        --output nemotron-calibration-inputs.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PERFECTBLEND = ("mlabonne/open-perfectblend", "af60f3c18201652a83a93f46fcfee1b646ba3df7",
                tuple(f"data/train-0000{i}-of-00006.parquet" for i in range(6)))
MMLU = ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/validation-00000-of-00001.parquet")
MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/validation-00000-of-00001.parquet")
# The sharar-eval datasets (relative to $SHARAR_RESULTS/datasets) and the fields holding their questions.
EVALS = {"accuracy/gsm8k_test.jsonl": ("question",), "accuracy/HumanEval.jsonl": ("prompt",), "accuracy/mbpp.jsonl": ("text",),
         "math500/test.jsonl": ("problem",), "hard/gpqa_diamond.jsonl": ("Question",), "hard/mmlupro.jsonl": ("question",),
         "hard/ifeval.jsonl": ("prompt",), "hard/wmt24pp.jsonl": ("source",)}
MGSM = "accuracy/mgsm"  # one TSV per language, the question in the first column
SPAN = 13  # words in the shared span that marks a prompt as an eval item
LETTERS = "ABCDEFGHIJ"
VERSION = "calibration-inputs-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parquet(repo: str, revision: str, path: str, columns=None) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("the calibration prompts' parquet files need pyarrow: `uv sync --group eval`") from exc
    from huggingface_hub import hf_hub_download
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=revision), columns=columns).to_pylist()


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class EvalFilter:
    """The eval sets' questions as 13-word spans (and whole questions shorter than that)."""

    def __init__(self, root: Path):
        missing = [n for n in EVALS if not (root / n).is_file()] + ([MGSM] if not (root / MGSM).is_dir() else [])
        if missing:
            raise SystemExit(f"eval sets missing under {root} (sharar-eval datasets fetch): {', '.join(missing)}")
        self.spans, self.short, self.files = set(), set(), {}
        for name, fields in EVALS.items():
            for line in filter(None, (root / name).read_text().split("\n")):
                row = json.loads(line)
                for field in fields:
                    self.add(str(row.get(field) or ""))
            self.files[name] = _sha256(root / name)
        for path in sorted((root / MGSM).glob("*.tsv")):
            for row in csv.reader(path.open(newline=""), delimiter="\t", quoting=csv.QUOTE_NONE):
                if row:
                    self.add(row[0])
            self.files[f"{MGSM}/{path.name}"] = _sha256(path)

    def add(self, text: str) -> None:
        words = _words(text)
        if len(words) < SPAN:
            if words:
                self.short.add(" ".join(words))
            return
        self.spans.update(" ".join(words[i:i + SPAN]) for i in range(len(words) - SPAN + 1))

    def __contains__(self, text: str) -> bool:
        words = _words(text)
        if " ".join(words) in self.short:
            return True
        return any(" ".join(words[i:i + SPAN]) in self.spans for i in range(len(words) - SPAN + 1))


def choice_prompt(question: str, options: list[str]) -> str:
    """The prompt sharar-eval's hard_eval sends for multiple choice (format only)."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n\n{opts}\n\nThink step by step, then finish with a line containing only: Answer: <letter>"


def _group(source: str) -> str:
    """A prompt's dataset: the Open-PerfectBlend source, or the knowledge set without its subject."""
    return source if source.startswith("perfectblend/") else source.split("/")[0]


def first_turn(conversation: list[dict]) -> str | None:
    """The first user turn of a ShareGPT-style conversation (a leading system turn is skipped)."""
    turns = [t for t in conversation if t.get("from") != "system"]
    return turns[0]["value"].strip() if turns and turns[0].get("from") == "human" else None


def prompts(args) -> None:
    evals = EvalFilter(args.evals)
    rng = random.Random(args.seed)
    by_source: dict[str, list[str]] = {}
    for path in PERFECTBLEND[2]:
        for row in _parquet(*PERFECTBLEND[:2], path, columns=["conversations", "source"]):
            q = first_turn(row["conversations"] or [])
            if q and args.min_chars <= len(q) <= args.max_chars:
                by_source.setdefault(row["source"], []).append(q)
    items, dropped = [], {}
    for source in sorted(by_source):
        pool = sorted(set(by_source[source]))
        rng.shuffle(pool)
        taken = 0
        for q in pool:
            if taken == args.per_source:
                break
            if q in evals:
                dropped[source] = dropped.get(source, 0) + 1
                continue
            items.append({"source": f"perfectblend/{source}", "prompt": q})
            taken += 1
    if args.knowledge:
        know = [(f"mmlu-val/{r['subject']}", r["question"], r["choices"]) for r in _parquet(*MMLU)]
        know += [(f"mmlupro-val/{r['category']}", r["question"], r["options"]) for r in _parquet(*MMLU_PRO)]
        rng.shuffle(know)
        taken = 0
        for source, question, options in know:
            if taken == args.knowledge:
                break
            if question in evals:
                dropped["knowledge"] = dropped.get("knowledge", 0) + 1
                continue
            items.append({"source": source, "prompt": choice_prompt(question, options)})
            taken += 1
    rng.shuffle(items)  # sources interleave, so packing mixes them and every held-out sample is a fair draw
    out = {"version": "calibration-prompts-v2", "perfectblend": list(PERFECTBLEND[:2]), "mmlu": list(MMLU),
           "mmlu_pro": list(MMLU_PRO), "seed": args.seed, "per_source": args.per_source, "knowledge": args.knowledge,
           "evals": evals.files, "dropped_as_eval": dropped, "items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    counts = {}
    for it in items:
        counts[_group(it["source"])] = counts.get(_group(it["source"]), 0) + 1
    print(f"{len(items)} prompts -> {args.output}\n  per source: {json.dumps(counts)}\n  dropped as eval items: {json.dumps(dropped)}")


def generate(args) -> None:
    items = json.loads(args.prompts.read_text())["items"]

    def ask(it):
        body = {"model": "sharar", "messages": [{"role": "user", "content": it["prompt"]}], "temperature": args.temperature,
                "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": True}, "trace": False}
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            choice = json.load(r)["choices"][0]
        message = choice["message"]
        return dict(it, answer=message.get("content") or "", reasoning=message.get("reasoning_content") or "",
                    finish_reason=choice.get("finish_reason"))

    partial = args.output.with_name(args.output.name + ".partial.jsonl")
    done = {}
    if partial.is_file():
        for line in filter(None, partial.read_text().split("\n")):
            row = json.loads(line)
            done[row["prompt"]] = row
    lock = threading.Lock()
    finished = [len(done)]

    def ask_saved(it):
        if it["prompt"] in done:
            return done[it["prompt"]]
        row = ask(it)
        with lock:
            with partial.open("a") as f:
                f.write(json.dumps(row) + "\n")
            finished[0] += 1
            if finished[0] % 50 == 0:
                print(f"  {finished[0]}/{len(items)} answered", flush=True)
        return row

    with ThreadPoolExecutor(args.concurrency) as pool:
        answered = list(pool.map(ask_saved, items))
    args.output.write_text(json.dumps({"version": "calibration-answers-v2", "prompts_sha256": _sha256(args.prompts),
                                       "url": args.url, "temperature": args.temperature, "max_tokens": args.max_tokens,
                                       "items": answered}))
    print(f"{len(answered)} answers ({sum(a['finish_reason'] == 'length' for a in answered)} cut at --max-tokens) -> {args.output}")


def build(args) -> None:
    from sharar import target as targets
    from sharar.api.chat_template import ChatTemplate
    from sharar.tokenizer import SnapshotTokenizer
    t = targets.get(args.target)
    snap = t.snapshot_dir()
    tmpl, tok = ChatTemplate(snap, t.template_defaults), SnapshotTokenizer(snap)
    answers = [it for it in json.loads(args.answers.read_text())["items"] if (it["answer"] + it["reasoning"]).strip()]
    samples, pack, sources = [], [], []

    def flush():
        if pack:
            samples.append({"source": ",".join(sorted(set(sources))), "input_ids": list(pack)})
            pack.clear()
            sources.clear()

    for it in answers:
        reply = {"role": "assistant", "content": it["answer"], "reasoning_content": it["reasoning"]}
        text = tmpl.render([{"role": "user", "content": it["prompt"]}, reply], add_generation_prompt=False, enable_thinking=True)
        ids = tok.encode(text)[:args.max_tokens]
        if len(pack) + len(ids) > args.max_tokens:
            flush()
        pack.extend(ids)
        sources.append(_group(it["source"]))
    flush()
    out = {"version": VERSION, "target": t.name, "target_revision": t.model_revision,
           "tokenizer_sha256": _sha256(snap / "tokenizer.json"), "answers_sha256": _sha256(args.answers),
           "max_tokens": args.max_tokens, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(answers)} answered turns -> {len(samples)} samples, {sum(len(s['input_ids']) for s in samples)} tokens -> {args.output}")


def load_samples(path: Path, target, limit: int = 0, max_tokens: int = 0) -> list[list[int]]:
    """The token ids of a samples file `build` wrote for `target` (its snapshot's tokenizer), the first `limit` (0: all)."""
    data = json.loads(Path(path).read_text())
    if data.get("version") != VERSION or data.get("target") != target.name:
        raise SystemExit(f"{path} is not a {VERSION} samples file for {target.name} (calibration_inputs build --target {target.name})")
    if data["tokenizer_sha256"] != _sha256(target.snapshot_dir() / "tokenizer.json"):
        raise SystemExit(f"{path} was tokenized with another {target.name} tokenizer")
    samples = [s["input_ids"] for s in data["samples"]][:limit or None]
    if not samples or any(not ids for ids in samples) or (max_tokens and any(len(ids) > max_tokens for ids in samples)):
        raise SystemExit(f"calibration samples must have between 1 and {max_tokens or 'any'} tokens (build --max-tokens)")
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts", help="Open-PerfectBlend first turns and MMLU validation questions, eval items dropped")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=160, help="prompts drawn from each Open-PerfectBlend source")
    p.add_argument("--knowledge", type=int, default=320, help="MMLU / MMLU-Pro validation questions")
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=6000)
    p.add_argument("--evals", type=Path, required=True, help="the sharar-eval datasets directory ($SHARAR_RESULTS/datasets)")
    p.add_argument("--seed", type=int, default=20261109)
    g = sub.add_parser("generate", help="the target's server answers every prompt (thinking on)")
    g.add_argument("--prompts", type=Path, required=True)
    g.add_argument("--url", required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--concurrency", type=int, default=16)
    b = sub.add_parser("build", help="render and tokenize the answers with the target's template")
    b.add_argument("--target", required=True, help="qwen38, flashnext, nemotron or bolt")
    b.add_argument("--answers", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--max-tokens", type=int, default=4096, help="2048 for the 27B (its capture's limit)")
    args = ap.parse_args()
    {"prompts": prompts, "generate": generate, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
