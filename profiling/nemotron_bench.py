"""Sharar decode tok/s on Nemotron: per depth, batch 1 and 4, decode time = generate(N) - generate(1) (same prompt,
prefix cache flushed)."""
import sys, time, json, torch
from pathlib import Path
from sharar import target as targets
from sharar.engine import config
from sharar.engine.sampler import SamplingParams
from sharar.tokenizer import SnapshotTokenizer
depths = [int(d) for d in sys.argv[1].split(",")]
extra = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
t = targets.get("nemotron")
hf = str(Path.home() / ".cache/huggingface/hub")
PROMPTS = ["Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.",
           "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
           "Explain step by step how to compute 37 * 48 mentally, then verify with long multiplication.",
           "Describe the causes of the French Revolution in five bullet points, with one sentence each."]
N = 256
cfg = config.current().replace(**{"target": "nemotron", "hf_cache": hf, "w3_decode": False, "w3_checkpoint": None,
                                  **{k: (int(v) if v.isdigit() else v) for k, v in extra.items()}})
results = []
with config.using(cfg):
    model = targets.load_model(t, t.snapshot_dir(), log=None)
    tok = SnapshotTokenizer(t.snapshot_dir())
    ids = [tok.encode(f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think></think>") for p in PROMPTS]
    from sharar.engine.executor import Executor
    for d in depths:
        kw = {}
        if d:
            from sharar.cache.spec_budget import SpecConfig
            kw["spec_config"] = SpecConfig("mtp", d, "reconstruct", "bf16", None)
        ex = Executor(model, log=None, max_batch=4, max_context=8192, **kw)
        ex.capture_graphs(log=None)
        ex.warmup(log=None)
        for b in (1, 4):
            rates, tpr = [], []
            for rep in range(2 if b == 4 else 4):
                sel = [ids[(rep + j) % 4] for j in range(b)]
                ex.flush_cache()
                t0 = time.perf_counter(); ex.generate_batch([(i, SamplingParams(max_new_tokens=1)) for i in sel]); t1 = time.perf_counter()
                ex.flush_cache()
                r0 = getattr(ex.spec, "rounds", 0) if getattr(ex, "spec", None) else 0; e0 = getattr(ex.spec, "emitted", 0) if getattr(ex, "spec", None) else 0
                t2 = time.perf_counter(); reqs = ex.generate_batch([(i, SamplingParams(max_new_tokens=N, ignore_eos=True)) for i in sel]); t3 = time.perf_counter()
                toks = sum(len(r.output_ids) - 1 for r in reqs)
                rates.append(toks / ((t3 - t2) - (t1 - t0)))
                if getattr(ex, "spec", None):
                    tpr.append((ex.spec.emitted - e0) / max(1, ex.spec.rounds - r0))
            row = {"engine": "Sharar", "depth": d, "batch": b, "tok_s": sum(rates) / len(rates), "per_prompt": [round(x, 1) for x in rates],
                   "tokens_per_round": (sum(tpr) / len(tpr)) if tpr else None, **extra}
            print(json.dumps(row), flush=True); results.append(row)
        del ex; torch.cuda.empty_cache()
