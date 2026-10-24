"""Sharar engine vs the reference's golden outputs: final logits at the last 8 prompt positions (teacher-forced
prefill), and the greedy continuation through the executor (plain decode and MTP)."""
import json, sys, torch
from pathlib import Path
from sharar import target as targets
from sharar.engine import config
from sharar.engine.sampler import SamplingParams
G = Path("/home/brian/Desktop/research/nemotron-logs/golden")
t = targets.get("nemotron")
hf = str(Path.home() / ".cache/huggingface/hub")
spec = sys.argv[1] if len(sys.argv) > 1 else "none"
# KEY=VALUE overrides of the engine config after the depth (e.g. w3_decode=1 w3_checkpoint=DIR for the 3-bit experts)
extra = {k: (int(v) if v.isdigit() else v) for k, v in (a.split("=", 1) for a in sys.argv[2:] if "=" in a)}
with config.using(config.current().replace(**{"target": "nemotron", "hf_cache": hf, "w3_decode": False, "w3_checkpoint": None, **extra})):
    model = targets.load_model(t, t.snapshot_dir(), log=None)
    for i in range(4):
        g = torch.load(G / f"prompt{i}.pt")
        ids = g["ids"].long().cuda()
        out = model.prefill(ids)
        pos = g["positions"].long()
        lg = model.logits(out["norm"][pos.cuda()]).float().cpu()
        ref = g["logits"].float()
        cos = torch.nn.functional.cosine_similarity(lg, ref, dim=-1)
        top = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
        fin = torch.nn.functional.cosine_similarity(out["norm"][pos.cuda()].float().cpu(), g["final"].float(), dim=-1)
        print(f"prompt{i} T={len(ids)} logits cos min {cos.min():.5f} mean {cos.mean():.5f} top1 {top:.3f} | final-norm cos min {fin.min():.5f}")
    from sharar.engine.executor import Executor
    kw = {}
    if spec != "none":
        from sharar.cache.spec_budget import SpecConfig
        kw["spec_config"] = SpecConfig("mtp", int(spec), "reconstruct", "bf16", None)
    ex = Executor(model, log=None, max_batch=4, max_context=8192, **kw)
    ex.capture_graphs(log=None)
    for i in range(4):
        g = torch.load(G / f"prompt{i}.pt")
        r = ex.generate_batch([(g["ids"].tolist(), SamplingParams(max_new_tokens=32, stop_token_ids=(11, 2)))])[0]
        ref = g["greedy"].tolist()
        n = next((j for j, (a, b) in enumerate(zip(r.output_ids, ref)) if a != b), min(len(r.output_ids), len(ref)))
        print(f"prompt{i} greedy: first divergence at {n} of {len(ref)} ({spec})")
