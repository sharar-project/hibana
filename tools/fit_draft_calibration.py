#!/usr/bin/env python3
"""Fit a DFlash2 checkpoint's acceptance calibration (`DRAFT_CALIBRATION` in the target's config): the weights w of
sigmoid(w . [logit p, gap, j / 15, 1]), a drafted position's probability of acceptance given the positions before it
were accepted (`ParallelProposer.acceptance_features`), which adaptive verification (engine/depth.RowBudget) ranks
verify seats by.

The tool drafts in process, one request at a time at the full depth (HIBANA_SPEC), greedy with --accept as the
acceptance rule, on chat prompts and the benchmark workloads. Every position a round tested (up to the first
rejection) is one sample: its features and whether it was accepted. Prompts alternate between the fit and the held-out
set; the AUC is reported on the held-out prompts.

    HIBANA_SPEC=15 python3 tools/fit_draft_calibration.py --out fit.json  # the 27B's pinned DFlash2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hibana import benchmark_workloads, target as targets  # noqa: E402
from hibana.cache.spec_budget import SpecConfig  # noqa: E402
from hibana.engine.config import current, install  # noqa: E402
from hibana.engine.depth import FixedDepth  # noqa: E402
from hibana.engine.executor import Executor  # noqa: E402
from hibana.engine.prompts import ChatTemplate  # noqa: E402
from hibana.engine.sampler import SamplingParams  # noqa: E402
from hibana.tokenizer import SnapshotTokenizer  # noqa: E402

CHATS = [
    "Write a Python class implementing an LRU cache, with unit tests.",
    "Explain how TCP congestion control works, step by step.",
    "Integrate x^2 e^x dx and show every step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Return a JSON list of 15 programming languages with their year and creator.",
    "Compare quicksort and mergesort: complexity, stability, memory, and when to use each.",
    "Write a bash script that backs up a directory with daily rotation.",
    "Describe the causes and consequences of the French revolution.",
    "Translate into French, German and Spanish: 'The meeting is moved to Thursday at noon.'",
    "A train leaves at 3pm at 60 mph and another at 4pm at 80 mph. When does the second catch up?",
    "Write a SQL schema for a library system and three example queries.",
    "Summarize the plot of Hamlet in ten bullet points.",
]


class Recorder(FixedDepth):
    """The fixed depth, recording each round's features (`ParallelProposer.acceptance`) and accepted counts."""

    def __init__(self, depth: int):
        super().__init__(depth)
        self.rows: list = []  # the current round's features per request, in batch order
        self.samples: list[tuple[int, list[float], int]] = []  # (prompt, features, accepted)
        self.prompt = 0

    def observe(self, request, depth: int, accepted: int) -> None:
        if not self.rows:
            return  # a round that drafted without scores
        logit, gap, position = self.rows.pop(0)
        for j in range(min(accepted + 1, depth)):
            self.samples.append((self.prompt, [logit[j], gap[j], position[j], 1.0], int(accepted > j)))


def fit(x: np.ndarray, y: np.ndarray, steps: int = 50, ridge: float = 1e-3) -> np.ndarray:
    """Logistic regression by Newton's method."""
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - y) + ridge * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--accept", default="1:0", help="greedy acceptance K:M while recording ('1:0' is exact)")
    ap.add_argument("--tokens", type=int, default=512, help="output tokens per prompt")
    ap.add_argument("--workload-tokens", type=int, default=4000, help="prompt length of the benchmark workloads")
    args = ap.parse_args()
    cfg = install(current().replace(spec_policy="fixed"))  # every round at the full depth, recorded by `Recorder`
    target = targets.get(cfg.target)
    snap = target.snapshot_dir()
    tok, template = SnapshotTokenizer(snap), ChatTemplate(snap)
    prompts = [tok.encode(template.render([{"role": "user", "content": c}], enable_thinking=False)) for c in CHATS]
    prompts += [benchmark_workloads.materialize(tok.encode, case, args.workload_tokens) for case in benchmark_workloads.WORKLOADS]
    spec = int(cfg.spec or 15)
    model = targets.load_model(target, snap)
    ex = Executor(model, max_batch=4, max_context=max(map(len, prompts)) + args.tokens + 64, log=print,
                  spec_config=SpecConfig("dflash2", spec, "reconstruct", os.environ.get("HIBANA_DRAFT_PRECISION", "fp8"), cfg.draft_dir))
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    recorder = ex.spec.depth_policy = Recorder(spec)
    proposer = ex.spec.proposer
    proposer.score_drafts = True

    def scored(candidates, scores, tokens, original=proposer.acceptance):
        f = proposer.acceptance_features(candidates, scores, tokens)
        recorder.rows.extend(zip(*(t.tolist() for t in f)))
        return original(candidates, scores, tokens)
    proposer.acceptance = scored
    top_k, margin = args.accept.split(":")
    params = SamplingParams(max_new_tokens=args.tokens, accept_top_k=int(top_k), accept_margin=float(margin))
    for i, ids in enumerate(prompts):
        recorder.prompt = i
        ex.flush_cache()
        ex.generate_batch([(ids, params)])
        print(f"prompt {i}: {sum(1 for s in recorder.samples if s[0] == i)} samples", flush=True)
    data = recorder.samples
    x = np.array([s[1] for s in data])
    y = np.array([s[2] for s in data], dtype=float)
    held = np.array([s[0] % 2 == 1 for s in data])
    x[:, 0] = np.clip(x[:, 0], -30, 30)
    w = fit(x[~held], y[~held])
    raw = auc(x[held, 0], y[held])  # the top probability alone
    fitted = auc(x[held] @ w, y[held])

    def loss(weights):  # held-out log loss and mean predicted probability (against the acceptance rate)
        p = np.clip(1 / (1 + np.exp(-x[held] @ np.asarray(weights))), 1e-6, 1 - 1e-6)
        return round(float(-np.mean(y[held] * np.log(p) + (1 - y[held]) * np.log(1 - p))), 4), round(float(p.mean()), 4)
    pinned = dict(target.draft_calibration).get("dflash2") if cfg.draft_dir is None else None
    out = {"target": target.name, "draft": str(cfg.draft_dir or "pinned"), "spec": spec, "accept": args.accept,
           "samples": len(data), "fit_samples": int((~held).sum()), "held_out_samples": int(held.sum()),
           "acceptance_rate": float(y.mean()), "weights": [round(float(v), 4) for v in w],
           "held_out_auc": round(fitted, 4), "held_out_auc_top_probability": round(raw, 4),
           "held_out_log_loss_mean_p": loss(w), "held_out_acceptance_rate": round(float(y[held].mean()), 4),
           "pinned": None if pinned is None else {"weights": list(pinned), "held_out_auc": round(auc(x[held] @ np.asarray(pinned), y[held]), 4),
                                                   "held_out_log_loss_mean_p": loss(pinned)}}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if math.isfinite(fitted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
