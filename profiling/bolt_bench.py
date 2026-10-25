"""Bounded Bolt load/greedy-generation probe with explicit request pools.

Measures host MemAvailable during loading, then generation on the actual chat
template. Run on an idle GPU. This is a bring-up measurement, not an h2h gain.
"""

import argparse
import json
from pathlib import Path
import threading
import time

import torch

from sharar.api.chat_template import ChatTemplate
from sharar.engine import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.launch.guard import start_thread
from sharar.model.bolt.model import Bolt
from sharar.model.nemotron.model import drop_page_cache
from sharar.target import NEMOTRON_BOLT
from sharar.tokenizer import SnapshotTokenizer


def available():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--spec", type=int, default=0, help="LiLiCorr draft tokens per round (0: ordinary decode)")
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snap = args.snapshot or NEMOTRON_BOLT.snapshot_dir(config.current().hf_cache or Path.home() / ".cache/huggingface")
    start_thread()
    cfg = config.current().replace(target="bolt", hf_cache=str(snap.parents[2]), w3_decode=False, moe_keep=0, moe_budget=0, moe_floors=None,
                                   moe_tau=0, moe_extra=0, ssd=False, ssm_state="fp32", kv_format="fp8")
    with config.using(cfg):
        drop_page_cache(snap)
        before = available()
        samples, stop = [before], threading.Event()

        def monitor():
            while not stop.wait(.02):
                samples.append(available())

        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        started = time.perf_counter()
        model = Bolt(snap)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        stop.set()
        worker.join()
        report = {"model": NEMOTRON_BOLT.model_id, "revision": NEMOTRON_BOLT.model_revision, "spec": args.spec,
                  "load_seconds": load_seconds, "load_peak_available_drop_gib": (before - min(samples)) / 2**30,
                  "loaded_cuda_gib": torch.cuda.memory_allocated() / 2**30, "requests": []}
        tok, template = SnapshotTokenizer(snap), ChatTemplate(snap)
        from sharar.cache.spec_budget import SpecConfig
        spec = SpecConfig("dflash2", args.spec, "reconstruct", "bf16") if args.spec else None
        ex = Executor(model, num_pages=8192, num_slots=16, max_context=4096, max_batch=1, use_cache=False, spec_config=spec)
        ex.capture_graphs()
        prompts = ["What is 17 + 25? Reply with only the integer.",
                   "Write a Python function that returns the first n Fibonacci numbers. Include a short explanation."]
        for prompt in prompts:
            ids = tok.encode(template.render([{"role": "user", "content": prompt}], enable_thinking=False))
            started = time.perf_counter()
            req = ex.generate(ids, SamplingParams(max_new_tokens=args.new_tokens, temperature=0.))
            torch.cuda.synchronize()
            row = {"prompt": prompt, "input_tokens": len(ids), "output_tokens": len(req.output_ids),
                   "text": tok.decode(req.output_ids, skip_special_tokens=True), "wall_seconds": time.perf_counter() - started,
                   "decode_seconds": req.decode_seconds, "decode_steps": req.decode_steps,
                   "decode_tps": (len(req.output_ids) - 1) / req.decode_seconds if req.decode_seconds else 0,
                   "spec_rounds": req.spec_rounds, "spec_emitted": req.spec_emitted,
                   "tokens_per_round": req.spec_emitted / req.spec_rounds if req.spec_rounds else None}
            report["requests"].append(row)
            print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
