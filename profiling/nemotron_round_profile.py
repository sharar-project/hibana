"""Kernel-time breakdown of Sharar's Nemotron speculative rounds (MTP depth d, batch 1, eager) by category."""
import sys, re, collections, torch
from pathlib import Path
from sharar import target as targets
from sharar.engine import config
from sharar.engine.sampler import SamplingParams
from sharar.tokenizer import SnapshotTokenizer
d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
t = targets.get("nemotron")
with config.using(config.current().replace(target="nemotron", hf_cache=str(Path.home()/".cache/huggingface/hub"), w3_decode=False, w3_checkpoint=None)):
    model = targets.load_model(t, t.snapshot_dir(), log=None)
    tok = SnapshotTokenizer(t.snapshot_dir())
    from sharar.engine.executor import Executor
    from sharar.cache.spec_budget import SpecConfig
    ex = Executor(model, log=None, max_batch=1, max_context=8192, spec_config=SpecConfig("mtp", d, "reconstruct", "bf16", None) if d else None)
    ex.capture_graphs(log=None)  # graphs off below for attribution
    ex.use_graphs = False
    ids = tok.encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWrite a Python class implementing an LRU cache.<|im_end|>\n<|im_start|>assistant\n<think></think>")
    ex.generate_batch([(ids, SamplingParams(max_new_tokens=32, ignore_eos=True))])
    ex.flush_cache()
    req = ex.admit(ids, SamplingParams(max_new_tokens=400, ignore_eos=True)); ex.prefill(req)
    for _ in range(5): ex.decode_step([req])
    torch.cuda.synchronize()
    r0 = ex.spec.rounds if ex.spec else 0
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): ex.decode_step([req])
        torch.cuda.synchronize()
    rounds = (ex.spec.rounds - r0) if ex.spec else 20
    cats = collections.Counter(); names = collections.Counter()
    for e in p.key_averages():
        if e.device_type.name != "CUDA": continue
        n = e.key; us = e.self_device_time_total
        names[n] += us
    def cat(n):
        l = n.lower()
        for k, pat in [("moe experts (cutlass)", "cutlass|moe_gemm|fused_moe|expandinput|finalize"), ("mamba core", "ssu|conv_|_chunk|replay|gated_norm|commit"),
                       ("fp8 gemm/gemv", "fp8|e4m3|gemv|scaled_mm|sm90_xmma|cublaslt"), ("bf16 gemm", "gemm|xmma|cutlass_80|ampere|sm80|wgmma|nvjet"),
                       ("attention", "prefill|decode|attention|batch_"), ("nvfp4 quant/act", "quant|fp4|relu2"), ("route", "route|topk|sigmoid")]:
            if re.search(pat, l): return k
        return "other"
    for n, us in names.items(): cats[cat(n)] += us
    total = sum(cats.values())
    print(f"depth {d}: {rounds} rounds, GPU {total/1e3/rounds:.1f} ms/round")
    for k, us in cats.most_common(): print(f"  {k:24s} {us/1e3/rounds:7.2f} ms/round")
    print("top kernels:")
    for n, us in names.most_common(18): print(f"  {us/1e3/rounds:7.2f}  {n[:110]}")
