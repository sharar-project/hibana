# SPDX-License-Identifier: Apache-2.0
"""Frozen calibration samples for the 3-bit decode copies (`hibana.weights.w3_calibrate`, `hibana.weights.flashnext_w3_calibrate`).

A synthetic coding proxy: Python source files under the package (`hibana/`, or `sharar/` at commits before the rename; a
`parity/` filter only matters when rebuilding samples at an older commit that still had it), chosen by a seeded shuffle of the files at a commit, each file's text tokenized with the
target's tokenizer and cut to --max-tokens. Task-accuracy datasets never enter. Both targets share the
tokenizer, so one file serves both calibrations. The output records the commit, the tokenizer's hash and each
source file's hash, so the same command at the same commit rebuilds the same samples.

    python3 -m hibana.weights.w3_calibration_inputs --tokenizer SNAPSHOT/tokenizer.json --output calibration-inputs.json
"""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=Path, required=True, help="the target snapshot's tokenizer.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--commit", default="HEAD", help="the repository commit whose files are sampled")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=256, help="skip files shorter than this")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", args.commit], capture_output=True, text=True, check=True).stdout.strip()
    names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", commit, "hibana/", "sharar/"],
                           capture_output=True, text=True, check=True).stdout.split()
    names = sorted(n for n in names if n.endswith(".py") and not n.startswith(("hibana/parity/", "sharar/parity/")))
    random.Random(args.seed).shuffle(names)
    samples, files = [], []
    for name in names:
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{name}"], capture_output=True, text=True, check=True).stdout
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < args.min_tokens:
            continue
        samples.append({"source": name, "input_ids": ids[:args.max_tokens]})
        files.append({"path": name, "sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(samples) == args.samples:
            break
    out = {"version": "w3-calibration-inputs-v1", "commit": commit, "seed": args.seed, "max_tokens": args.max_tokens,
           "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(), "files": files, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(samples)} samples, {sum(len(s['input_ids']) for s in samples)} tokens from {commit[:10]} -> {args.output}")


if __name__ == "__main__":
    main()
