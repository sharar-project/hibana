#!/usr/bin/env python3
"""Capture reference outputs over HTTP from the running SGLang container.

For every prompt in the parity set this records, using exact token IDs:
  - top-8 logprobs at every prompt position (prefill logits argmax check),
  - the greedy continuation (decode-path check),
  - the chat-completion text for the canaries (quality check).
Runs on the host with the standard library only. Output: <out>/http/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MODEL = "RadixArk/Qwen3.8-27B-NVFP4"


def post(url: str, payload: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # /flush_cache answers with plain text.
        return body.decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.out) / "http"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in prompts.items():
        post(f"{args.base_url}/flush_cache", {}, timeout=30)
        resp = post(
            f"{args.base_url}/generate",
            {
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": args.max_new_tokens, "temperature": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": args.top_k,
            },
        )
        meta = resp["meta_info"]
        # input_top_logprobs[i] is None for position 0 and a list of
        # [logprob, token_id, text] for the prediction made at position i-1.
        top = []
        for entry in meta["input_top_logprobs"]:
            top.append(None if entry is None else [[float(lp), int(t)] for lp, t, _ in entry])
        record = {
            "name": name,
            "prompt_len": len(ids),
            "input_top_logprobs": top,
            "input_token_logprobs": [
                None if e[0] is None else float(e[0]) for e in meta["input_token_logprobs"]
            ],
            "output_ids": [int(t) for t in resp["output_ids"]],
            "output_text": resp["text"],
            "output_top_logprobs": [
                [[float(lp), int(t)] for lp, t, _ in entry] for entry in meta["output_top_logprobs"]
            ],
        }
        (out_dir / f"{name}.json").write_text(json.dumps(record))
        print(f"{name}: {len(ids)} tokens -> {record['output_ids'][:8]}... {record['output_text'][:60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
