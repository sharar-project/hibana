"""Draft-quality check for the MTP head: run the target over a prompt, feed
the draft (x_i, h_{i-1}) for every position and measure how often its top-1
equals the actual next token x_{i+1}. High agreement on text means the module
is wired correctly; the acceptance rate under speculation is bounded by it."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from sharar import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.mtp import MtpHead
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import read_prompts
from sharar.weights.safetensors import SnapshotWeights


def main(prompts_path: str, names: str) -> int:
    snap = config.snapshot_dir()
    model = Qwen38(snap, log=None)
    ex = Executor(model, use_cache=False)
    w = SnapshotWeights(snap)
    mtp = MtpHead(w, model.device, model.embed, model.lm_head, model.rope, ex.kv_pool.num_pages)
    w.close()
    print(f"MTP head {mtp.bytes / 2**30:.2f} GiB (incl. draft KV)")
    prompts = read_prompts(Path(prompts_path))
    for name in names.split(","):
        ids = prompts[name]
        req = ex.admit(ids, SamplingParams(max_new_tokens=1))
        n = len(ids)
        # target prefill capturing all positions' post-norm hidden states
        from sharar.engine.context import ForwardContext
        ctx = ForwardContext(mode="prefill", kv_pool=ex.kv_pool, state_pool=ex.state_pool, positions=torch.arange(n, device=model.device), slot=req.slot, pages=req.pages, seq_len=n)
        out = model.forward(torch.tensor(ids, device=model.device), ctx)
        h = out["norm"]  # [n, 5120]
        pos = torch.arange(1, n, device=model.device)
        toks = torch.tensor(ids[1:], device=model.device)
        truth = torch.tensor(ids[2:], device=model.device)
        preds = {}
        for variant, hid in (("post-norm", out["norm"]), ("pre-norm", out["pre_norm"])):
            d = mtp.forward(toks, hid[:-1], pos, req.pages[1:n], [n - 1], [req.pages[1:n]])
            pred = []
            for start in range(0, n - 2, 1024):
                pred.append(mtp.logits(d[start : min(n - 2, start + 1024)]).argmax(-1))
            preds[variant] = torch.cat(pred)
        pred = preds["post-norm"]
        acc = (pred == truth).float().mean().item()
        # target's own next-token accuracy on the same positions for reference
        tgt = []
        for start in range(1, n - 1, 1024):
            tgt.append(model.logits(h[start : min(n - 1, start + 1024)]).argmax(-1))
        tgt = torch.cat(tgt)
        tgt_acc = (tgt == truth).float().mean().item()
        agree = (pred == tgt).float().mean().item()
        agree_pre = (preds["pre-norm"] == tgt).float().mean().item()
        print(f"  [{name}] T={n}: draft(post-norm h) vs target top-1 {100 * agree:.1f}% | draft(pre-norm h) vs target top-1 {100 * agree_pre:.1f}% | target top-1 vs next token {100 * tgt_acc:.1f}%")
        ex.release(req)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "text1024,canary_needle,rand2048"))
