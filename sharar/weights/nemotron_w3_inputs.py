# SPDX-License-Identifier: Apache-2.0
"""Frozen calibration samples for Nemotron's 3-bit decode copies (`sharar.weights.nemotron_w3_calibrate`).

Two parts, each tokenized with Nemotron's own tokenizer (it is not the Qwen targets', so their sample files do not
carry over):

- the coding proxy of `w3_calibration_inputs` (source files of this repository at a commit), built first with that
  module and `--tokenizer` pointing at Nemotron's snapshot;
- reasoning traces: ShareGPT first user turns (English and other languages) with a model's reasoning and answer, as
  `flashnext_expert_inputs generate --thinking` writes them, rendered in Nemotron's chat template
  (`<think>` reasoning `</think>` answer) and cut to `--max-tokens`. They make the samples look like served chat:
  prose, other languages and long thinking, which a code-only set never shows the projections.

A seeded draw picks `--traces-en` and `--traces-other` items; the output records both inputs' hashes, so the same
command rebuilds the same samples. Task-accuracy datasets never enter.

    python3 -m sharar.weights.w3_calibration_inputs --tokenizer SNAP/tokenizer.json --commit bc7984f --output code.json
    python3 -m sharar.weights.nemotron_w3_inputs --tokenizer SNAP/tokenizer.json --code code.json \\
        --traces answers-sharegpt.json --output nemotron-calibration-inputs.json
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def render(item: dict) -> str:
    """One trace as Nemotron's chat template renders a finished thinking turn (empty system prompt)."""
    return (f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            f"<think>\n{item.get('reasoning') or ''}\n</think>\n{item.get('answer') or ''}<|im_end|>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=Path, required=True, help="Nemotron's snapshot tokenizer.json")
    ap.add_argument("--code", type=Path, required=True, help="w3_calibration_inputs output built with the same tokenizer")
    ap.add_argument("--traces", type=Path, required=True, help="{'items': [{source, prompt, reasoning, answer}, ...]}")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--traces-en", type=int, default=48)
    ap.add_argument("--traces-other", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=256, help="skip traces shorter than this")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
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
            ids = tok.encode(render(it), add_special_tokens=False).ids
            if len(ids) < args.min_tokens:
                continue
            samples.append({"source": f"{it['source']}/{it.get('id')}", "input_ids": ids[:args.max_tokens]})
            taken += 1
    out = {"version": "nemotron-w3-calibration-inputs-v1", "seed": args.seed, "max_tokens": args.max_tokens,
           "tokenizer_sha256": tok_sha, "code": {"commit": code["commit"], "sha256": hashlib.sha256(args.code.read_bytes()).hexdigest()},
           "traces_sha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(), "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(samples)} samples, {sum(len(s['input_ids']) for s in samples)} tokens -> {args.output}")


if __name__ == "__main__":
    main()
