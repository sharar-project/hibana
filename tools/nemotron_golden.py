#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Golden outputs of the Nemotron-3.5-Super plain-torch reference (`sharar/model/nemotron/reference.py`), for checking
kernels and the engine against it.

    tools/nemotron_golden.py [--out DIR] [--dtype bf16|fp32] [--new 32]

Four fixed chat prompts (thinking off, rendered by the snapshot's chat template): a ~600-token summarisation, a
short factual question, a short code request and a short arithmetic one. For each, `prompt{k}.pt` (torch.save) holds:

- `text`, `ids` (int32 [T]): the rendered prompt and its tokens; `positions` (int64 [8]): the last 8 positions;
- `hidden` {block: bf16 [8, 4096]}: the residual stream after trunk blocks 0, 1, 7, 8, 40, 87 at those positions
  (block 0 Mamba, 1 MoE, 7 the first attention, 8 MoE after it, 40 Mamba, 87 the last, an MoE);
- `final` bf16 [8, 4096]: after norm_f; `logits` bf16 [8, 131072] from lm_head;
- `greedy` int32 [new]: the greedy continuation, decoded one token a step through the carried state (Mamba conv/SSM
  states, KV cache), and `decode_top` = (FP32 [new, 8] logits, int32 [new, 8] ids): each step's top 8;
- `mtp_next` int32 [8]: the token after each of the 8 positions (the last one is greedy[0]), `mtp_final`
  bf16 [8, 4096] (after final_layernorm) and `mtp_logits` bf16 [8, 131072]: the MTP head run over the whole prompt
  (hidden = norm_f output, next token) and sliced to those positions; they predict the token two ahead;
- `mamba0_conv` [3, 10240] (bf16, raw pre-conv xBC rows) and `mamba0_ssm` FP32 [128, 64, 128]: block 0's states after
  the prompt.

`manifest.json` records the model, revision, dtype, the reference file's SHA-256, the blocks and the shapes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from sharar.model.nemotron import reference as R

BLOCKS = (0, 1, 7, 8, 40, 87)
LAST = 8

LONG = """Summarize the following design note in five bullet points, then name its single biggest risk.

Design note: a regional library system wants to replace its aging book-tracking software. The current system was
written in-house fifteen years ago, runs on a single server in the basement of the central branch, and keeps its data
in a database whose vendor stopped issuing security patches four years ago. Each of the eleven branches connects over
a leased line; when the line to a branch drops, that branch cannot check books in or out and staff fall back to paper
slips that are typed in later, which regularly produces duplicate or lost records. Patrons can reserve books online,
but the reservation page times out during the busy hour after school, and the mobile layout was never finished.

The proposal is to move to a hosted service. Every branch would keep a small local cache, so that check-outs and
returns keep working while the network is down and are reconciled automatically once it comes back. Reconciliation
would treat the physical book as the source of truth: if two branches disagree about where a copy is, the branch
that most recently scanned the barcode wins, and the conflict is logged for a librarian to review. Patron accounts,
fines and reading histories would move to the hosted database; reading histories would be kept for ninety days by
default instead of indefinitely, with an opt-in for patrons who want a longer record.

Migration would happen branch by branch over six months, starting with the two smallest branches. Each migration
weekend would freeze the old system for that branch, export its records, run a validation script that compares item
counts, open loans and outstanding fines between the old and new systems, and only switch over when the counts
match exactly. Staff training would be two half-day sessions per branch, plus a printed quick-reference card.

The budget covers the first three years of the hosted subscription, barcode scanners for branches whose scanners are
older than seven years, and a part-time contractor for the migration scripts. It does not cover replacing the
self-service kiosks, which only speak the old system's protocol; the vendor says an adapter is "on the roadmap".
Until that adapter ships, the kiosks at the four largest branches would be switched off, which the circulation
desks estimate would add twenty to thirty minutes of queueing on Saturday mornings. Two board members have asked
whether the hosted vendor's contract allows the library to export all of its data in an open format if it later
wants to leave, and the draft contract is silent on this point. The union representing branch staff supports the
change but wants the paper-slip fallback kept as a documented procedure for at least the first year.
The board meets next month to approve or reject the plan."""

PROMPTS = (
    LONG,
    "What is the tallest mountain in Africa, and roughly how tall is it?",
    "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals and returns them "
    "sorted.",
    "If a train leaves at 14:35 and the trip takes 2 hours 50 minutes, when does it arrive?",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("/home/brian/Desktop/research/nemotron-logs/golden"))
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--expert-cache-gib", type=float, default=8.0)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    snap = R.find_snapshot()
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    t0 = time.time()
    model = R.NemotronReference(snap, dtype=dtype, expert_cache_gib=args.expert_cache_gib)
    print(f"loaded in {time.time() - t0:.0f} s")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": R.MODEL_ID, "revision": R.MODEL_REVISION, "dtype": args.dtype, "blocks": list(BLOCKS),
                "last": LAST, "new": args.new,
                "reference_sha256": hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(), "prompts": []}
    bf = torch.bfloat16
    with torch.inference_mode():
        for k, prompt in enumerate(PROMPTS):
            t0 = time.time()
            text = R.render_chat(snap, [{"role": "user", "content": prompt}])
            ids = torch.tensor(tok.encode(text, add_special_tokens=False).ids)
            t_len = ids.numel()
            state = R.State()
            out = model.forward(ids, state, capture=BLOCKS, logits=LAST)
            conv0, ssm0 = state.conv[0].clone(), state.ssm[0].clone()
            greedy, top_v, top_i = [], [], []
            logits = out["logits"][-1]
            for _ in range(args.new):
                v, i = logits.topk(8)
                top_v.append(v)
                top_i.append(i)
                greedy.append(int(i[0]))
                logits = model.forward(torch.tensor([greedy[-1]]), state, logits=1)["logits"][-1]
            nxt = torch.cat([ids[1:], torch.tensor(greedy[:1])])
            mtp = model.mtp_forward(out["final"], nxt)
            sl = slice(t_len - LAST, t_len)
            record = {
                "text": text, "ids": ids.to(torch.int32), "positions": torch.arange(t_len - LAST, t_len),
                "hidden": {b: out["hidden"][b][sl].to(bf).cpu() for b in BLOCKS},
                "final": out["final"][sl].to(bf).cpu(), "logits": out["logits"].to(bf).cpu(),
                "greedy": torch.tensor(greedy, dtype=torch.int32),
                "decode_top": (torch.stack(top_v).float().cpu(), torch.stack(top_i).to(torch.int32).cpu()),
                "mtp_next": nxt[sl].to(torch.int32), "mtp_final": mtp["final"][sl].to(bf).cpu(),
                "mtp_logits": mtp["logits"][sl].to(bf).cpu(),
                "mamba0_conv": conv0.to(bf).cpu(), "mamba0_ssm": ssm0.float().cpu(),
            }
            torch.save(record, args.out / f"prompt{k}.pt")
            cont = tok.decode(greedy)
            manifest["prompts"].append({"file": f"prompt{k}.pt", "tokens": t_len, "continuation": cont,
                                        "mtp_top1": mtp["logits"][sl].argmax(-1).tolist()})
            print(f"prompt {k}: {t_len} tokens, {time.time() - t0:.0f} s -> {cont!r}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
