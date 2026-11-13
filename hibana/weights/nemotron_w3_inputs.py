# SPDX-License-Identifier: Apache-2.0
"""Frozen calibration samples for Nemotron's 3-bit copies: the trunk's (`hibana.weights.nemotron_w3_calibrate`) and,
with `--knowledge`, the routed experts' (`hibana.weights.nemotron_w3_experts`).

Each part is tokenized with Nemotron's own tokenizer (it is not the Qwen targets', so their sample files do not carry
over):

- the coding proxy of `w3_calibration_inputs` (source files of this repository at a commit), built first with that
  module and `--tokenizer` pointing at Nemotron's snapshot;
- reasoning traces: ShareGPT first user turns (English and other languages) with a model's reasoning and answer, as
  `flashnext_expert_inputs generate --thinking` writes them, rendered in Nemotron's chat template
  (`<think>` reasoning `</think>` answer) and cut to `--max-tokens`. They make the samples look like served chat:
  prose, other languages and long thinking, which a code-only set never shows the projections;
- `--knowledge` (the experts' set): MMLU / MMLU-Pro validation questions answered with their reasoning, rendered the
  same way and packed whole into samples of up to `--max-tokens`.

A seeded draw picks `--traces-en` and `--traces-other` items (and `--knowledge-items`); the output records every
input's hash, so the same command rebuilds the same samples. Task-accuracy test sets never enter.

    python3 -m hibana.weights.w3_calibration_inputs --tokenizer SNAP/tokenizer.json --commit bc7984f --output code.json
    python3 -m hibana.weights.nemotron_w3_inputs --tokenizer SNAP/tokenizer.json --code code.json \\
        --traces answers-sharegpt.json --output nemotron-calibration-inputs.json
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def render(item: dict) -> str:
    """One answered turn as Nemotron's chat template renders a finished thinking turn (empty system prompt)."""
    return (f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            f"<think>\n{item.get('reasoning') or ''}\n</think>\n{item.get('answer') or ''}<|im_end|>\n")


def load_samples(path: Path, limit: int = 0) -> list[list[int]]:
    """The token ids of a samples file this module wrote, the first `limit` of them (0: all)."""
    samples = [s["input_ids"] for s in json.loads(Path(path).read_text())["samples"]]
    return samples[:limit] if limit else samples


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=Path, required=True, help="Nemotron's snapshot tokenizer.json")
    ap.add_argument("--code", type=Path, required=True, help="w3_calibration_inputs output built with the same tokenizer")
    ap.add_argument("--traces", type=Path, required=True, help="{'items': [{source, id, prompt, reasoning, answer}, ...]}")
    ap.add_argument("--knowledge", type=Path, help="{'items': [{source, prompt, reasoning, answer}, ...]} (MMLU validation)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--traces-en", type=int, default=48)
    ap.add_argument("--traces-other", type=int, default=32)
    ap.add_argument("--knowledge-items", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=256, help="skip traces shorter than this")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    encode = lambda item: tok.encode(render(item), add_special_tokens=False).ids  # noqa: E731
    tok_sha = _sha256(args.tokenizer)
    code = json.loads(args.code.read_text())
    if code.get("tokenizer_sha256") != tok_sha:
        raise SystemExit("the coding proxy was tokenized with another tokenizer")
    samples = [{"source": s["source"], "input_ids": s["input_ids"]} for s in code["samples"]]
    items = json.loads(args.traces.read_text())["items"]
    rng = random.Random(args.seed)
    for lang, want in (("en", args.traces_en), ("other", args.traces_other)):
        pool = [it for it in items if it["source"].endswith(f"/{lang}") and (it.get("reasoning") or it.get("answer"))]
        rng.shuffle(pool)
        taken = 0
        for it in pool:
            if taken == want:
                break
            ids = encode(it)
            if len(ids) < args.min_tokens:
                continue
            samples.append({"source": f"{it['source']}/{it.get('id')}", "input_ids": ids[:args.max_tokens]})
            taken += 1
    if args.knowledge:  # short answers: packed into samples of up to max_tokens, whole answers only
        know = [it for it in json.loads(args.knowledge.read_text())["items"] if (it.get("answer") or "").strip()]
        rng.shuffle(know)
        pack, packed = [], 0
        for it in know[:args.knowledge_items]:
            ids = encode(it)[:args.max_tokens]
            if pack and len(pack) + len(ids) > args.max_tokens:
                samples.append({"source": f"knowledge/{packed}", "input_ids": pack})
                pack, packed = [], packed + 1
            pack = pack + ids
        if pack:
            samples.append({"source": f"knowledge/{packed}", "input_ids": pack})
    # the version names the set: the trunk's samples, or with the knowledge answers the experts'
    out = {"version": "nemotron-expert-inputs-v1" if args.knowledge else "nemotron-w3-calibration-inputs-v1",
           "seed": args.seed, "max_tokens": args.max_tokens, "tokenizer_sha256": tok_sha,
           "code": {"commit": code["commit"], "sha256": _sha256(args.code)}, "traces_sha256": _sha256(args.traces),
           **({"knowledge_sha256": _sha256(args.knowledge)} if args.knowledge else {}), "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(samples)} samples, {sum(len(s['input_ids']) for s in samples)} tokens -> {args.output}")


if __name__ == "__main__":
    main()
