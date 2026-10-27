# SPDX-License-Identifier: Apache-2.0
"""Calibrated 3-bit decode copies of Bolt's trunk projections, as `nemotron_w3_calibrate` fits Super's: the 40 Mamba
blocks' NVFP4 in_proj [18560, 4096] and out_proj [4096, 8192], the 40 MoE blocks' Q8_0 fc1_latent [1024, 4096] and
fc2_latent [4096, 1024], the 8 attention blocks' Q8_0 q|k|v [4608, 4096] and o [4096, 4096], and the Q8_0 lm_head
[131072, 4096] (`Bolt.linears()` but the IQ2_XXS shared experts, already 2.06 bits a weight).

The source is the checkpoint matrix the serving path decodes (`PackedLinear.dequantize`: NVFP4 with its block and
global scales, Q8_0 with its FP16 bases); GPTQ (`w3_calibration.quantize`) runs on each projection's input covariance
as serving produces it: the BF16 activation each projection reads, with every upstream block at its served precision.

- `capture` loads the model (b12x on the checkpoint encodings, no 3-bit copies), prefills each frozen sample alone
  and adds x^T x of every projection's input to its covariance, the lm_head's from the final norm of every position.
  Every `--probe-every`-th sample is held out as probe rows (64 per call, evenly spaced).
- `fit` reads one projection at a time from the snapshot and the capture and writes the W3 file, its manifest entry
  and a report of the relative weight error and relative output error (under the covariance and on the probes), for
  GPTQ and for nearest rounding. `--resume` continues; only a complete fit writes `manifest.json`.

    SHARAR_TARGET=bolt python3 -m sharar.weights.bolt_w3_calibrate capture --inputs INPUTS.json --activations ACT
    SHARAR_TARGET=bolt python3 -m sharar.weights.bolt_w3_calibrate fit --activations ACT --output DIR

Serve with SHARAR_W3_DECODE=1 SHARAR_W3_CHECKPOINT=DIR: calls of at most 64 rows read the copies, prefill keeps b12x.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from sharar.weights.nemotron_w3_calibrate import errors
from sharar.weights.w3_checkpoint import sha256 as _sha256

PROBE_ROWS = 64  # probe rows kept per projection call of a held-out sample, as Nemotron's capture keeps

FIT_KINDS = ("in_proj", "out_proj", "qkv", "o", "fc1", "fc2")


def source_prefixes(name: str) -> list[str]:
    """The checkpoint tensors a `Bolt.linears()` name loads from (`PackedLinear.load` prefixes, fused in order)."""
    if name == "lm_head":
        return ["language_model.lm_head"]
    m = re.fullmatch(r"layers\.(\d+)\.mixer\.(\w+)", name)
    if m is None or m.group(2) not in FIT_KINDS:
        raise ValueError(f"not a fitted Bolt projection: {name}")
    base = f"language_model.backbone.layers.{m.group(1)}.mixer"
    return {"in_proj": [f"{base}.in_proj"], "out_proj": [f"{base}.out_proj"], "o": [f"{base}.o_proj"],
            "qkv": [f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj"],
            "fc1": [f"{base}.fc1_latent_proj"], "fc2": [f"{base}.fc2_latent_proj"]}[m.group(2)]


def projection_names(target) -> list[str]:
    """Every fitted name, from the block pattern alone (the fit does not load the model)."""
    kinds = {"linear_attention": ("in_proj", "out_proj"), "full_attention": ("qkv", "o"), "moe": ("fc1", "fc2")}
    return [f"layers.{i}.mixer.{k}" for i, kind in enumerate(target.layer_types) for k in kinds[kind]] + ["lm_head"]


class Capture:
    """x^T x of each fitted projection's BF16 input (and held-out probe rows), from `PackedLinear` calls."""

    def __init__(self, linears):
        self.names = {id(lin): name for name, lin in linears}
        self.h, self.rows, self.probes = {}, {}, {}
        self.holdout = False

    def add(self, name: str, x: torch.Tensor) -> None:
        if self.holdout:
            step = max(1, x.shape[0] // PROBE_ROWS)
            self.probes.setdefault(name, []).append(x[::step][:PROBE_ROWS].to(torch.bfloat16))
            return
        if name not in self.h:
            self.h[name] = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=x.device)
            self.rows[name] = 0
        self.h[name].addmm_(x.T, x)
        self.rows[name] += x.shape[0]

    def __enter__(self):
        from sharar.model.bolt.linear import PackedLinear
        self.cls, self.original = PackedLinear, PackedLinear.__call__
        capture, original = self, PackedLinear.__call__

        def call(lin, x, **kwargs):
            name = capture.names.get(id(lin))
            if name is not None:
                capture.add(name, x.reshape(-1, x.shape[-1]).float())
            return original(lin, x, **kwargs)
        PackedLinear.__call__ = call
        return self

    def __exit__(self, *args):
        self.cls.__call__ = self.original


@torch.inference_mode()
def capture(args) -> None:
    from safetensors.torch import save_file

    from sharar import target as targets
    from sharar.engine.config import current, install
    install(current().replace(target="bolt", w3_decode=False, w3_checkpoint=None))
    t = targets.get("bolt")
    act = args.activations
    if (act / "capture.json").exists():
        raise SystemExit(f"{act} holds a finished capture")
    act.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
    if not samples or any(not ids or len(ids) > 4096 for ids in samples):
        raise ValueError("calibration samples must have between 1 and 4096 tokens")
    model = targets.load_model(t, t.snapshot_dir(), log=print)
    names = set(projection_names(t))
    linears = [(n, lin) for n, lin in model.linears() if n in names]
    started = time.perf_counter()
    with Capture(linears) as cap:
        for i, ids in enumerate(samples):
            cap.holdout = args.probe_every > 0 and i % args.probe_every == args.probe_every - 1
            out = model.prefill(torch.tensor(ids, device=model.device))
            for r in range(0, len(ids), 1024):  # the lm_head's input; the logits themselves are dropped
                model.logits(out["norm"][r:r + 1024])
            del out
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} samples ({time.perf_counter() - started:.0f}s)", flush=True)
    torch.cuda.synchronize()
    for name, _ in linears:
        if name not in cap.h:
            raise RuntimeError(f"{name} was never called on the calibration path")
        probes = torch.cat(cap.probes.pop(name)) if name in cap.probes else torch.zeros((0, cap.h[name].shape[0]), dtype=torch.bfloat16)
        save_file({"h": cap.h.pop(name).cpu(), "probes": probes.cpu().contiguous(),
                   "rows": torch.tensor([cap.rows[name]], dtype=torch.int64)}, str(act / f"{name}.safetensors"))
    (act / "capture.json").write_text(json.dumps({
        "target": t.name, "target_revision": t.model_revision, "inputs": str(args.inputs), "inputs_sha256": _sha256(args.inputs),
        "samples": len(samples), "probe_every": args.probe_every,
        "tokens": sum(len(ids) for i, ids in enumerate(samples) if not (args.probe_every and i % args.probe_every == args.probe_every - 1)),
        "probe_tokens": sum(len(ids) for i, ids in enumerate(samples) if args.probe_every and i % args.probe_every == args.probe_every - 1),
        "seconds": round(time.perf_counter() - started, 1)}, indent=1) + "\n")
    print(f"captured {len(linears)} projections -> {act}", flush=True)


def fit(args) -> None:
    from safetensors.torch import load_file

    from sharar import target as targets
    from sharar.engine.config import current, install
    from sharar.kernels import w3
    from sharar.model.bolt.linear import PackedLinear
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_calibration import quantize
    from sharar.weights.w3_checkpoint import FORMAT_VERSION, save_weight
    install(current().replace(target="bolt", w3_decode=False, w3_checkpoint=None))
    t = targets.get("bolt")
    act, out = args.activations, args.output
    captured = json.loads((act / "capture.json").read_text())
    if captured["target_revision"] != t.model_revision:
        raise SystemExit("the capture is of another checkpoint revision")
    if (out / "manifest.json").exists():
        raise SystemExit("refusing to overwrite a finished calibration")
    out.mkdir(parents=True, exist_ok=True)
    skip = re.compile(args.exclude) if args.exclude else None
    grid = [float(x) for x in args.scale_grid.split(",")]
    names = [n for n in projection_names(t) if not (skip and skip.search(n)) and not (args.only and not re.search(args.only, n))]
    manifest = {"format": FORMAT_VERSION, "target": t.name, "target_revision": t.model_revision,
                "source": "checkpoint NVFP4 / Q8_0 projections (PackedLinear.dequantize)", "method": args.method,
                "inputs_sha256": captured["inputs_sha256"], "samples": captured["samples"], "tokens": captured["tokens"],
                "probe_tokens": captured["probe_tokens"], "damping": args.damping, "rows_per_chunk": args.rows_per_chunk,
                "exclude": args.exclude, "act_order": args.act_order, "scale_grid": grid, "only": args.only,
                "weights": {}, "reports": {}}
    identity = ("format", "target_revision", "method", "inputs_sha256", "damping", "rows_per_chunk", "exclude",
                "act_order", "scale_grid", "only")
    progress = out / "progress.json"
    if progress.exists():
        if not args.resume:
            raise SystemExit("an unfinished fit exists; pass --resume or use a new output directory")
        previous = json.loads(progress.read_text())
        if any(previous.get(k) != manifest[k] for k in identity):
            raise SystemExit("--resume settings do not match the unfinished fit")
        for name, entry in previous["weights"].items():
            if _sha256(out / entry["file"]) == entry["sha256"]:
                manifest["weights"][name], manifest["reports"][name] = entry, previous["reports"][name]
    reader = SnapshotWeights(t.snapshot_dir())
    device = torch.device(args.device)
    started = time.perf_counter()
    for name in names:
        if name in manifest["weights"]:
            continue
        weight = PackedLinear.load(reader, source_prefixes(name), device).dequantize().float()
        reader.close()
        a = load_file(str(act / f"{name}.safetensors"), device=str(device))
        h = a["h"] / float(a["rows"].item())
        nearest = w3.pack(weight)
        report = {"rows": int(a["rows"].item()), "probe_rows": int(a["probes"].shape[0]),
                  "nearest": errors(weight, w3.dequantize(nearest), h, a["probes"])}
        if args.method == "gptq":
            del nearest
            packed, fitted = quantize(weight, h, damping=args.damping, rows_per_chunk=args.rows_per_chunk,
                                      act_order=args.act_order, ratios=tuple(grid))
            report.update(fitted)
            report["gptq"] = errors(weight, w3.dequantize(packed), h, a["probes"])
        else:
            packed = nearest
        manifest["weights"][name] = save_weight(out / f"{name}.safetensors", packed)
        manifest["reports"][name] = report
        chosen = report.get("gptq", report["nearest"])
        print(f"  {name}: weight {report['nearest']['weight']:.4f} -> {chosen['weight']:.4f}, output (cov) "
              f"{report['nearest']['output_cov']:.4f} -> {chosen['output_cov']:.4f}, held-out "
              f"{report['nearest'].get('output_probe', float('nan')):.4f} -> {chosen.get('output_probe', float('nan')):.4f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        del weight, a, h, packed
        if device.type == "cuda":
            torch.cuda.empty_cache()
        temporary = out / "progress.tmp"
        temporary.write_text(json.dumps(manifest, indent=1) + "\n")
        temporary.replace(progress)
    manifest["seconds"] = round(time.perf_counter() - started, 1)
    temporary = out / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    temporary.replace(out / "manifest.json")
    progress.unlink(missing_ok=True)
    print(f"done: {len(manifest['weights'])} matrices -> {out}", flush=True)


@torch.inference_mode()
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="covariances and held-out probes of every fitted projection's input (loads the model)")
    c.add_argument("--inputs", type=Path, required=True, help="frozen samples: {'samples': [{'input_ids': [...]}, ...]} (nemotron_w3_inputs)")
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--probe-every", type=int, default=8, help="hold every N-th sample out of the covariances as probes (0: none)")
    c.add_argument("--limit", type=int, default=0, help="first N samples only (0: all)")
    f = sub.add_parser("fit", help="GPTQ (or nearest) copies from a capture (no model load)")
    f.add_argument("--activations", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--method", choices=("gptq", "nearest"), default="gptq")
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--rows-per-chunk", type=int, default=1024)
    f.add_argument("--exclude", default="", help="regex of names to keep on their checkpoint encoding (e.g. lm_head)")
    f.add_argument("--act-order", action="store_true", help="GPTQ in decreasing input energy, static groups (w3_calibration.quantize)")
    f.add_argument("--scale-grid", default="0.6,0.7,0.8,0.9,1.0", help="clipping ratios the group scales try")
    f.add_argument("--only", default="", help="regex: fit only these names (the manifest covers them alone)")
    f.add_argument("--device", default="cuda")
    f.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    {"capture": capture, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
