"""Sharar decode speed on Nemotron-3.5-Super under the lossy speculation knobs: relaxed acceptance and budgeted expert
selection ('MoE sharing'), swept over MTP depths in one model load.

    flock $SHARAR_RESULTS/.gpu.lock python profiling/nemotron_lossy_bench.py \\
        --depths 3,4,5,6 --accept off,2:1.0,3:2.0 --moe off,8:36:96 --batches 1,4

For every (MoE setting, depth) one executor is built and its graphs captured; every acceptance rule then decodes the
same four fixed chat prompts (code, story, arithmetic, bullet points; thinking off) greedily for --tokens tokens
(ignore_eos). Decode time is generate(N) minus generate(1) on the same prompts, the prefix cache flushed before each
(profiling/nemotron_bench.py's method), so prefill and the first token are excluded. A batch-1 row averages the four
prompts one at a time; a batch-4 row runs them together, twice. Each JSON line reports tok/s (total over the batch),
tokens per round (emitted per request and verify round) and ms per round.

--moe takes KEEP:CAP or KEEP:P:CAP (SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET as [P:]CAP), or KEEP alone (each token's top
KEEP, no budget). The setting is applied to the loaded trunk's MoE blocks before the executor captures its graphs
(`NemotronMoe.route` reads it at capture), so one load serves the whole sweep; the MTP head always routes in full.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from sharar import target as targets
from sharar.engine import config
from sharar.engine.sampler import SamplingParams
from sharar.tokenizer import SnapshotTokenizer

PROMPTS = ["Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.",
           "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
           "Explain step by step how to compute 37 * 48 mentally, then verify with long multiplication.",
           "Describe the causes of the French Revolution in five bullet points, with one sentence each."]
TEMPLATE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def parse_moe(text: str) -> tuple[int, int, int]:
    """'off' -> (0, 0, 0); 'K' -> (K, 0, 0); 'K:CAP' -> (K, CAP, 0); 'K:P:CAP' -> (K, CAP, P)."""
    if text == "off":
        return 0, 0, 0
    parts = [int(x) for x in text.split(":")]
    if len(parts) == 1:
        return parts[0], 0, 0
    if len(parts) == 2:
        return parts[0], parts[1], 0
    return parts[0], parts[2], parts[1]


def field(item: str) -> tuple[str, object]:
    """'name=value' -> (name, int, float or str value)."""
    name, value = item.split("=", 1)
    for kind in (int, float):
        try:
            return name, kind(value)
        except ValueError:
            pass
    return name, value


def set_moe(model, keep: int, budget: int, per_request: int) -> None:
    for layer in model.layers:
        if layer.kind == "moe":
            m = layer.mixer
            m.keep = keep if keep < m.topk else 0
            m.budget = budget if m.keep else 0
            m.per_request = per_request


def measure(model, ids, depth: int, rules, batches, tokens: int, row: dict, out: Path | None, max_batch: int = 4) -> None:
    """One executor at `depth` (the cap under adaptive depth), built under the installed config, every rule and batch."""
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine.executor import Executor
    kw = {"spec_config": SpecConfig("mtp", depth, "reconstruct", "bf16", None)} if depth else {}
    # fixed small pools: a planned executor takes the spare memory, and the next one's plan would find it still held
    # (the model keeps its registered pools until the next executor registers its own)
    ex = Executor(model, log=None, max_batch=max_batch, max_context=8192, num_pages=65536 * max(1, max_batch // 4),
                  num_slots=3 * max_batch, **kw)
    ex.capture_graphs(log=None)
    ex.warmup(log=None)
    for k, m in rules:
        for b in batches:
            rates, tpr, ms = [], [], []
            for rep in range(4 if b == 1 else 2):
                sel = [ids[(rep + j) % 4] for j in range(b)]  # wider batches repeat the four prompts
                sp = lambda n: SamplingParams(max_new_tokens=n, ignore_eos=True, accept_top_k=k, accept_margin=m)  # noqa: E731
                ex.flush_cache()
                t0 = time.perf_counter(); ex.generate_batch([(i, sp(1)) for i in sel]); t1 = time.perf_counter()
                ex.flush_cache()
                spec = getattr(ex, "spec", None)
                r0, e0 = (spec.rounds, spec.emitted) if spec else (0, 0)
                t2 = time.perf_counter()
                reqs = ex.generate_batch([(i, sp(tokens)) for i in sel])
                t3 = time.perf_counter()
                toks = sum(len(r.output_ids) - 1 for r in reqs)
                secs = (t3 - t2) - (t1 - t0)
                rates.append(toks / secs)
                if spec:
                    rounds = max(1, spec.rounds - r0)
                    tpr.append((spec.emitted - e0) / rounds / b)  # a round verifies every request in the batch
                    ms.append(1000 * secs / rounds)
            r = {"engine": "Sharar", "model": "nemotron", **row, "depth": depth, "accept": f"{k}:{m}", "batch": b,
                 "tok_s": round(sum(rates) / len(rates), 2), "per_rep": [round(x, 1) for x in rates],
                 "tokens_per_round": round(sum(tpr) / len(tpr), 3) if tpr else None,
                 "round_ms": round(sum(ms) / len(ms), 2) if ms else None}
            print(json.dumps(r), flush=True)
            if out:
                with out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
    del ex
    gc.collect()
    torch.cuda.empty_cache()


def rule(text: str) -> tuple[int, float]:
    return (1, 0.0) if text == "off" else (int(text.split(":")[0]), float(text.split(":")[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", default="3")
    ap.add_argument("--accept", default="off", help="comma-separated K:M rules, 'off' = exact")
    ap.add_argument("--moe", default="off", help="comma-separated KEEP[:P]:CAP settings, 'off' = full routing")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=4, help="the executor's batch limit (graph buckets up to it)")
    ap.add_argument("--device-accept", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="an EngineConfig field for every executor, e.g. spec_step_cost=0.19, spec_depths=2:4,4:4, "
                         "draft_head=w2 (repeatable; read when the executor and its proposer are built)")
    ap.add_argument("--stack", type=Path, default=None,
                    help="a JSON list of steps {label, depth, accept, moe, set: {field: value}, batches, max_batch} measured in turn on one "
                         "load, instead of --depths/--accept/--moe/--set (each knob's increment over the step before)")
    ap.add_argument("--label", default="", help="a name for the rows")
    ap.add_argument("--out", type=Path, default=None, help="append the JSON lines here too")
    args = ap.parse_args()
    t = targets.get("nemotron")
    from sharar.launch.settings import resolve
    hf = str(resolve("nemotron").hf_cache)  # as `sharar serve nemotron` finds it (environment, config file, default)
    cfg = config.current().replace(target="nemotron", hf_cache=hf, w3_decode=False, w3_checkpoint=None,
                                   device_accept=bool(args.device_accept), **dict(field(x) for x in args.set))
    batches = [int(x) for x in args.batches.split(",")]
    with config.using(cfg):
        model = targets.load_model(t, t.snapshot_dir(), log=None)
        tok = SnapshotTokenizer(t.snapshot_dir())
        ids = [tok.encode(TEMPLATE.format(p)) for p in PROMPTS]
        if args.stack:
            for step in json.loads(args.stack.read_text()):
                set_moe(model, *parse_moe(step.get("moe", "off")))
                with config.using(cfg.replace(**step.get("set", {}))):
                    measure(model, ids, step["depth"], [rule(step.get("accept", "off"))], step.get("batches", batches), args.tokens,
                            {"label": step["label"], "moe": step.get("moe", "off"), "set": step.get("set", {})}, args.out,
                            step.get("max_batch", args.max_batch))
            return
        for moe in args.moe.split(","):
            set_moe(model, *parse_moe(moe))
            for d in (int(x) for x in args.depths.split(",")):
                measure(model, ids, d, [rule(a) for a in args.accept.split(",")], batches, args.tokens,
                        {"label": args.label, "moe": moe, "device_accept": bool(args.device_accept), "set": args.set},
                        args.out, args.max_batch)


if __name__ == "__main__":
    main()
