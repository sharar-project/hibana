# SPDX-License-Identifier: Apache-2.0
"""Calibrated 3-bit decode copies for the Flash-Next trunk.

GPTQ (hibana/weights/w3_calibration.py, the pass that took the 27B's 3-bit copies
from GSM8K 92.0 to 94.5%) fitted from the checkpoint's BF16 trunk. Inputs are the
activations each projection sees on the serving path: the model loads with the
BF16 trunk, an Executor prefills the frozen calibration samples (the 27B's set;
both targets share the tokenizer), and every TrunkLinear call adds x^T x to its
projection's covariance. Every projection `FlashNext.linears()` lists whose shape
the 3-bit kernel tiles (N and K multiples of 128) is fitted; the rank-320
hyper-connection mixers stay FP8. The output loads with
HIBANA_W3_DECODE=1 HIBANA_W3_CHECKPOINT=DIR over the FP8 trunk (the launcher serves
$HIBANA_RESULTS/w3-checkpoints/flashnext-current in the fast profile).

    HIBANA_TARGET=flashnext python3 -m hibana.weights.flashnext_w3_calibrate --inputs calibration-inputs.json --output DIR
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch


class Covariances:
    """x^T x per projection, accumulated from TrunkLinear calls on the serving path."""

    def __init__(self, linears):
        self.names = {id(lin): name for name, lin in linears}
        self.matrices, self.rows = {}, {}

    def __enter__(self):
        from hibana.model.flashnext.linear import TrunkLinear
        self.cls, self.original = TrunkLinear, TrunkLinear.__call__
        covariances = self

        def call(lin, x):
            name = covariances.names.get(id(lin))
            if name is not None:
                flat = x.reshape(-1, x.shape[-1]).float()
                if name not in covariances.matrices:
                    covariances.matrices[name] = torch.zeros((flat.shape[1], flat.shape[1]), device=flat.device)
                    covariances.rows[name] = 0
                covariances.matrices[name].addmm_(flat.T, flat)
                covariances.rows[name] += flat.shape[0]
            return covariances.original(lin, x)
        TrunkLinear.__call__ = call
        return self

    def __exit__(self, *args):
        self.cls.__call__ = self.original


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True, help="frozen calibration samples: {'samples': [{'input_ids': [...]}, ...]}")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--damping", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (0: all)")
    ap.add_argument("--exclude", default="", help="regex of linears() names to leave out (e.g. lm_head)")
    ap.add_argument("--unbias", choices=("weight", "output"), help="per-row gain correction through the row scales (w3_calibration.quantize)")
    args = ap.parse_args()
    from hibana.engine.config import current, install
    from hibana.engine.executor import Executor
    from hibana.engine.sampler import SamplingParams
    from hibana.model.flashnext.model import FlashNext
    from hibana.target import FLASHNEXT
    from hibana.weights.w3_calibration import quantize
    from hibana.weights.w3_checkpoint import FORMAT_VERSION, save_weight, sha256
    install(current().replace(flashnext_trunk="bf16", w3_decode=False, w3_checkpoint=None))
    if (args.output / "manifest.json").exists():
        raise SystemExit("refusing to overwrite a finished calibration")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
    model = FlashNext(FLASHNEXT.snapshot_dir(), precision="bf16")
    ex = Executor(model, num_pages=40000, num_slots=8, max_context=32768, log=print)
    skip = re.compile(args.exclude) if args.exclude else None
    linears = [(n, l) for n, l in model.linears()
               if l.out_features % 128 == 0 and l.in_features % 128 == 0 and not (skip and skip.search(n))]
    print(f"{len(linears)} projections to fit, {len(samples)} samples, {sum(map(len, samples))} tokens", flush=True)
    started = time.perf_counter()
    with Covariances(linears) as cov:
        for i, ids in enumerate(samples):
            req = ex.admit(list(ids), SamplingParams(max_new_tokens=1))
            try:
                ex.prefill(req)
            finally:
                ex.release(req)
            ex.flush_cache()
            if (i + 1) % 16 == 0:
                print(f"  covariances: {i + 1}/{len(samples)} samples ({time.perf_counter() - started:.0f}s)", flush=True)
    manifest = {"format": FORMAT_VERSION, "target": FLASHNEXT.name, "target_revision": FLASHNEXT.model_revision,
                "source": "checkpoint BF16 trunk", "inputs_sha256": sha256(args.inputs), "samples": len(samples),
                "tokens": sum(map(len, samples)), "damping": args.damping, "exclude": args.exclude, "unbias": args.unbias, "weights": {}, "reports": {}}
    for j, (name, lin) in enumerate(linears):
        if name not in cov.matrices:
            print(f"  {name}: never called on the calibration path, left out", flush=True)
            continue
        weight = lin.dequantize()  # the BF16 checkpoint weights (the model loaded with the BF16 trunk)
        w3weight, report = quantize(weight, cov.matrices.pop(name), damping=args.damping, unbias=args.unbias)
        report["rows"] = cov.rows[name]
        manifest["weights"][name] = save_weight(args.output / f"{name}.safetensors", w3weight)
        manifest["reports"][name] = report
        del weight, w3weight
        if (j + 1) % 50 == 0:
            print(f"  fitted {j + 1}/{len(linears)} ({time.perf_counter() - started:.0f}s)", flush=True)
    manifest["seconds"] = time.perf_counter() - started
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(manifest['weights'])} matrices in {manifest['seconds']:.0f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
