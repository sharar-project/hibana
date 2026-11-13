# SPDX-License-Identifier: Apache-2.0
"""Calibrated 3-bit decode copies of Nemotron's FP8 projections: the 40 Mamba blocks' in_proj [18560, 4096] and
out_proj [4096, 8192], the 8 attention blocks' q|k|v [4608, 4096] and o [4096, 4096], and the lm_head
[131072, 4096] (`Nemotron.linears()`).

The source is the served FP8 matrix itself (`load_fp8_linear`, including the fused q|k|v's requantization to one
scale): the checkpoint has no wider release, and the copy should approximate what `exact` serves. GPTQ
(`w3_calibration.quantize`: the column-error update with the eight-level Lloyd codebook, 32-weight groups, E4M3 group
scales) on each projection's input covariance as the serving path produces it: the FP8 activations the projection
reads (x_q times its static input scale), with every upstream block at its served precision.

Two steps, so the full-model part stays short and the fit can resume without the model:

- `capture` loads the model (FP8 projections, NVFP4 experts, no 3-bit copies), prefills each frozen sample alone
  (`Nemotron.prefill`) and adds x^T x of every projection's input to its covariance (`w3_calibrate.InputCovariances`),
  the lm_head's from the final norm of every position. Every `--probe-every`-th sample is held out of the
  covariances: its rows (64 per call, evenly spaced) become the probes the fit measures output error on. One file per projection under ACT (fp32
  covariance, row count, BF16 probes; about 15 GB for all 97).
- `fit` reads one projection at a time from the snapshot and ACT and writes the W3 file, its manifest entry and a
  report: relative weight error, relative output error under the covariance, tr(D H D^T) / tr(W H W^T) for D the
  copy's error, and on the held-out probes, each for GPTQ and for nearest rounding of the same matrix. `--resume`
  continues after the last matrix whose file checks out; only a complete fit writes `manifest.json`.

    HIBANA_TARGET=nemotron python3 -m hibana.weights.nemotron_w3_calibrate capture --inputs INPUTS.json --activations ACT
    HIBANA_TARGET=nemotron python3 -m hibana.weights.nemotron_w3_calibrate fit --activations ACT --output DIR

Serve with HIBANA_W3_DECODE=1 HIBANA_W3_CHECKPOINT=DIR: calls of at most 64 rows read the copies, prefill keeps FP8.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch


def source_prefixes(name: str) -> list[str]:
    """The checkpoint tensors a `Nemotron.linears()` name loads from (`load_fp8_linear` prefixes, fused in order)."""
    from hibana.model.nemotron.config import PREFIX
    if name == "lm_head":
        return [f"{PREFIX}.lm_head"]
    m = re.fullmatch(r"layers\.(\d+)\.mixer\.(in_proj|out_proj|qkv|o)", name)
    if m is None:
        raise ValueError(f"not a Nemotron projection: {name}")
    base = f"{PREFIX}.backbone.layers.{m.group(1)}.mixer"
    return {"in_proj": [f"{base}.in_proj"], "out_proj": [f"{base}.out_proj"], "o": [f"{base}.o_proj"],
            "qkv": [f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj"]}[m.group(2)]


def projection_names(target) -> list[str]:
    """Every `Nemotron.linears()` name, from the block pattern alone (the fit does not load the model)."""
    names = []
    for i, kind in enumerate(target.layer_types):
        if kind == "linear_attention":
            names += [f"layers.{i}.mixer.in_proj", f"layers.{i}.mixer.out_proj"]
        elif kind == "full_attention":
            names += [f"layers.{i}.mixer.qkv", f"layers.{i}.mixer.o"]
    return names + ["lm_head"]


@torch.inference_mode()
def capture(args) -> None:
    from safetensors.torch import save_file

    from hibana import target as targets
    from hibana.engine.config import current, install
    from hibana.weights.calibration_inputs import load_samples
    from hibana.weights.w3_calibrate import InputCovariances
    from hibana.weights.w3_checkpoint import sha256
    install(current().replace(target="nemotron", w3_decode=False, w3_checkpoint=None))
    t = targets.get("nemotron")
    act = args.activations
    if (act / "capture.json").exists():
        raise SystemExit(f"{act} holds a finished capture")
    act.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.inputs, t, args.limit, max_tokens=4096)
    held = [args.probe_every > 0 and i % args.probe_every == args.probe_every - 1 for i in range(len(samples))]
    model = targets.load_model(t, t.snapshot_dir(), log=print)
    linears = list(model.linears())
    started = time.perf_counter()
    with InputCovariances(linears, served_fp8=True) as cap:
        for i, ids in enumerate(samples):
            cap.holdout = held[i]
            out = model.prefill(torch.tensor(ids, device=model.device))
            for r in range(0, len(ids), 1024):  # the lm_head's input; the logits themselves are dropped
                model.logits(out["norm"][r:r + 1024])
            del out
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} samples ({time.perf_counter() - started:.0f}s)", flush=True)
    torch.cuda.synchronize()
    for name, _ in linears:
        if name not in cap.matrices:
            raise RuntimeError(f"{name} was never called on the calibration path")
        h = cap.matrices.pop(name)
        probes = torch.cat(cap.held.pop(name)) if name in cap.held else torch.zeros((0, h.shape[0]), dtype=torch.bfloat16)
        save_file({"h": h.cpu(), "probes": probes.cpu().contiguous(), "rows": torch.tensor([cap.counts[name]], dtype=torch.int64)},
                  str(act / f"{name}.safetensors"))
    (act / "capture.json").write_text(json.dumps({
        "target": t.name, "target_revision": t.model_revision, "inputs": str(args.inputs), "inputs_sha256": sha256(args.inputs),
        "samples": len(samples), "probe_every": args.probe_every,
        "tokens": sum(len(ids) for ids, h in zip(samples, held) if not h),
        "probe_tokens": sum(len(ids) for ids, h in zip(samples, held) if h),
        "seconds": round(time.perf_counter() - started, 1)}, indent=1) + "\n")
    print(f"captured {len(linears)} projections in {time.perf_counter() - started:.0f}s -> {act}", flush=True)


def errors(weight: torch.Tensor, approx: torch.Tensor, h: torch.Tensor, probes: torch.Tensor) -> dict:
    """Relative weight error, relative output error under the covariance and on the held-out probe rows."""
    d = approx - weight
    out = {"weight": float(d.norm() / weight.norm())}
    num = den = 0.0
    for r in range(0, weight.shape[0], 8192):  # tr(D H D^T) / tr(W H W^T) a slice of rows at a time
        num += float(((d[r:r + 8192] @ h) * d[r:r + 8192]).sum())
        den += float(((weight[r:r + 8192] @ h) * weight[r:r + 8192]).sum())
    out["output_cov"] = num / den
    if probes.shape[0]:
        p = probes.float()
        out["output_probe"] = float((p @ d.T).square().sum() / (p @ weight.T).square().sum())
    return out


@torch.inference_mode()
def fit(args) -> None:
    from safetensors.torch import load_file

    from hibana import target as targets
    from hibana.engine.config import current, install
    from hibana.kernels import w3
    from hibana.weights.modelopt import load_fp8_linear
    from hibana.weights.safetensors import SnapshotWeights
    from hibana.weights.w3_calibration import quantize
    from hibana.weights.w3_checkpoint import FORMAT_VERSION, save_weight, sha256
    install(current().replace(target="nemotron", w3_decode=False, w3_checkpoint=None))
    t = targets.get("nemotron")
    act, out = args.activations, args.output
    captured = json.loads((act / "capture.json").read_text())
    if captured["target_revision"] != t.model_revision:
        raise SystemExit("the capture is of another checkpoint revision")
    if (out / "manifest.json").exists():
        raise SystemExit("refusing to overwrite a finished calibration")
    out.mkdir(parents=True, exist_ok=True)
    skip = re.compile(args.exclude) if args.exclude else None
    names = [n for n in projection_names(t) if not (skip and skip.search(n))]
    manifest = {"format": FORMAT_VERSION, "target": t.name, "target_revision": t.model_revision,
                "source": "served FP8 projections (load_fp8_linear)", "method": "gptq",
                "inputs_sha256": captured["inputs_sha256"], "samples": captured["samples"], "tokens": captured["tokens"],
                "probe_tokens": captured["probe_tokens"], "damping": args.damping,
                "rows_per_chunk": args.rows_per_chunk, "exclude": args.exclude, "weights": {}, "reports": {}}
    identity = ("format", "target_revision", "method", "inputs_sha256", "damping", "rows_per_chunk", "exclude")
    progress = out / "progress.json"
    if progress.exists():
        if not args.resume:
            raise SystemExit("an unfinished fit exists; pass --resume or use a new output directory")
        previous = json.loads(progress.read_text())
        if any(previous.get(k) != manifest[k] for k in identity):
            raise SystemExit("--resume settings do not match the unfinished fit")
        for name, entry in previous["weights"].items():
            if sha256(out / entry["file"]) == entry["sha256"]:
                manifest["weights"][name], manifest["reports"][name] = entry, previous["reports"][name]
    reader = SnapshotWeights(t.snapshot_dir())
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    started = time.perf_counter()
    for name in names:
        if name in manifest["weights"]:
            continue
        lin = load_fp8_linear(reader, source_prefixes(name), device)
        reader.close()
        weight = lin.dequantize()
        del lin
        a = load_file(str(act / f"{name}.safetensors"), device=str(device))
        h = a["h"] / float(a["rows"].item())
        report = {"rows": int(a["rows"].item()), "probe_rows": int(a["probes"].shape[0]),
                  "nearest": errors(weight, w3.dequantize(w3.pack(weight)), h, a["probes"])}
        packed, fitted = quantize(weight, h, damping=args.damping, rows_per_chunk=args.rows_per_chunk)
        report.update(fitted)
        report["gptq"] = errors(weight, w3.dequantize(packed), h, a["probes"])
        manifest["weights"][name] = save_weight(out / f"{name}.safetensors", packed)
        manifest["reports"][name] = report
        nearest, gptq = report["nearest"], report["gptq"]
        print(f"  {name}: weight {nearest['weight']:.4f} -> {gptq['weight']:.4f}, output (cov) "
              f"{nearest['output_cov']:.4f} -> {gptq['output_cov']:.4f}, held-out "
              f"{nearest.get('output_probe', float('nan')):.4f} -> {gptq.get('output_probe', float('nan')):.4f} "
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="covariances and held-out probes of every projection's input (loads the model)")
    c.add_argument("--inputs", type=Path, required=True, help="calibration_inputs build --target nemotron")
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--probe-every", type=int, default=8, help="hold every N-th sample out of the covariances as probes (0: none)")
    c.add_argument("--limit", type=int, default=0, help="first N samples only (0: all)")
    f = sub.add_parser("fit", help="GPTQ copies from a capture (no model load)")
    f.add_argument("--activations", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--rows-per-chunk", type=int, default=1024)
    f.add_argument("--exclude", default="", help="regex of names to leave FP8 (e.g. lm_head)")
    f.add_argument("--device", default="cuda", help="cpu fits without the GPU (about 15 minutes for all 97 on the GB10's cores)")
    f.add_argument("--threads", type=int, default=16, help="CPU threads with --device cpu")
    f.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    {"capture": capture, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
