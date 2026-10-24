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
  (`Nemotron.prefill`) and adds x^T x of every projection's input to its covariance, the lm_head's from the final
  norm of every position. Every `--probe-every`-th sample is held out of the covariances: its rows (64 per call,
  evenly spaced) become the probes the fit measures output error on. One file per projection under ACT (fp32
  covariance, row count, BF16 probes; about 15 GB for all 97).
- `fit` reads one projection at a time from the snapshot and ACT and writes the W3 file, its manifest entry and a
  report: relative weight error, relative output error under the covariance, tr(D H D^T) / tr(W H W^T) for D the
  copy's error, and on the held-out probes, each for GPTQ and for nearest rounding of the same matrix. `--resume`
  continues after the last matrix whose file checks out; only a complete fit writes `manifest.json`.
  `--method nearest` writes nearest-rounded copies in the same format (the comparison arm).

    SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate capture --inputs INPUTS.json --activations ACT
    SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate fit --activations ACT --output DIR

Serve with SHARAR_W3_DECODE=1 SHARAR_W3_CHECKPOINT=DIR: calls of at most 64 rows read the copies, prefill keeps FP8.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

PROBE_ROWS = 64  # probe rows kept per projection call of a held-out sample


def source_prefixes(name: str) -> list[str]:
    """The checkpoint tensors a `Nemotron.linears()` name loads from (`load_fp8_linear` prefixes, fused in order)."""
    from sharar.model.nemotron.config import PREFIX
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


class Capture:
    """x^T x of each projection's served input (and held-out probe rows), from `ops.fp8_linear` calls."""

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
        from sharar.kernels import ops
        from sharar.kernels.fp8_quant import quant_fp8
        self.ops, self.original = ops, ops.fp8_linear
        capture, original = self, ops.fp8_linear

        def fp8_linear(x, lin, x_q=None):
            name = capture.names.get(id(lin))
            if name is not None:
                # what the projection (and its 3-bit copy) reads: the FP8 activation at the layer's static scale
                q = x_q if x_q is not None else quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
                capture.add(name, q.reshape(-1, q.shape[-1]).float() * lin.input_scale.float())
            return original(x, lin, x_q=x_q)
        ops.fp8_linear = fp8_linear
        return self

    def __exit__(self, *args):
        self.ops.fp8_linear = self.original


def _sha256(path: Path) -> str:
    from sharar.weights.w3_checkpoint import sha256
    return sha256(path)


@torch.inference_mode()
def capture(args) -> None:
    from safetensors.torch import save_file

    from sharar import target as targets
    from sharar.engine.config import current, install
    install(current().replace(target="nemotron", w3_decode=False, w3_checkpoint=None))
    t = targets.get("nemotron")
    act = args.activations
    if (act / "capture.json").exists():
        raise SystemExit(f"{act} holds a finished capture")
    act.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
    if not samples or any(not ids or len(ids) > 4096 for ids in samples):
        raise ValueError("calibration samples must have between 1 and 4096 tokens")
    model = targets.load_model(t, t.snapshot_dir(), log=print)
    linears = list(model.linears())
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
    print(f"captured {len(linears)} projections in {time.perf_counter() - started:.0f}s -> {act}", flush=True)


class UpstreamCapture:
    """The served inputs of each projection twice over the same tokens: `x` with every block FP8 (a whole-sample
    prefill, which never reads a 3-bit copy) and `x~` with the 3-bit copies serving (64-row chunks, the decode path),
    for the asymmetric fit: H~ = x~^T x~ and Q = x^T x~ (the copy then answers for its own error and for the error
    that reached its input). Phase "fp" stores the FP8 rows of the selected projections; phase "w3" pairs each 64-row
    call with the next stored rows of its projection."""

    def __init__(self, linears, names: set[str], probe_stride: int = 16):
        self.names = {id(lin): name for name, lin in linears if name in names}
        self.phase, self.holdout, self.stride = "fp", False, probe_stride
        self.fp, self.cursor = {}, {}
        self.h, self.q, self.rows, self.probes, self.probes_fp = {}, {}, {}, {}, {}

    def start_sample(self) -> None:
        self.phase, self.fp, self.cursor = "fp", {}, {}

    def switch(self) -> None:
        self.fp = {name: torch.cat(parts) for name, parts in self.fp.items()}
        self.cursor = dict.fromkeys(self.fp, 0)
        self.phase = "w3"

    def finish_sample(self) -> None:
        for name, rows in self.fp.items():
            if self.cursor[name] != rows.shape[0]:
                raise RuntimeError(f"{name}: {self.cursor[name]} of {rows.shape[0]} FP8 rows paired")

    def add(self, name: str, x_q: torch.Tensor, scale: torch.Tensor) -> None:
        if self.phase == "fp":
            self.fp.setdefault(name, []).append(x_q.clone())
            return
        start = self.cursor[name]
        ref = self.fp[name][start:start + x_q.shape[0]]
        self.cursor[name] = start + x_q.shape[0]
        xt, xf = x_q.float() * scale.float(), ref.float() * scale.float()
        if self.holdout:
            keep = torch.arange(x_q.shape[0], device=x_q.device)
            keep = keep[(keep + start) % self.stride == 0]
            self.probes.setdefault(name, []).append(xt[keep].to(torch.bfloat16))
            self.probes_fp.setdefault(name, []).append(xf[keep].to(torch.bfloat16))
            return
        if name not in self.h:
            k = x_q.shape[1]
            self.h[name] = torch.zeros((k, k), dtype=torch.float32, device=x_q.device)
            self.q[name] = torch.zeros((k, k), dtype=torch.float32, device=x_q.device)
            self.rows[name] = 0
        self.h[name].addmm_(xt.T, xt)
        self.q[name].addmm_(xf.T, xt)
        self.rows[name] += x_q.shape[0]

    def __enter__(self):
        from sharar.kernels import ops
        from sharar.kernels.fp8_quant import quant_fp8
        self.ops, self.original = ops, ops.fp8_linear
        capture, original = self, ops.fp8_linear

        def fp8_linear(x, lin, x_q=None):
            name = capture.names.get(id(lin))
            if name is not None:
                q = x_q if x_q is not None else quant_fp8(x.reshape(-1, x.shape[-1]), lin.input_scale)
                capture.add(name, q.reshape(-1, q.shape[-1]), lin.input_scale)
            return original(x, lin, x_q=x_q)
        ops.fp8_linear = fp8_linear
        return self

    def __exit__(self, *args):
        self.ops.fp8_linear = self.original


@torch.inference_mode()
def capture_upstream(args) -> None:
    """`capture` for the asymmetric fit: each sample prefills whole (FP8) and then in 64-row chunks through an executor
    with the 3-bit copies of --checkpoint serving (the gate's decode-path scoring); see `UpstreamCapture`."""
    from safetensors.torch import save_file

    from sharar import target as targets
    from sharar.engine.config import current, install, using
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    install(current().replace(target="nemotron", w3_decode=True, w3_checkpoint=str(args.checkpoint)))
    t = targets.get("nemotron")
    act = args.activations
    if (act / "capture.json").exists():
        raise SystemExit(f"{act} holds a finished capture")
    act.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
    model = targets.load_model(t, t.snapshot_dir(), log=print)
    linears = list(model.linears())
    names = {n for n, _ in linears if not args.only or re.search(args.only, n)}
    names -= {n for n in names if (act / f"{n}.safetensors").exists()}
    ex = Executor(model, log=None, max_batch=1, max_context=max(len(ids) for ids in samples) + 64, num_pages=65536, num_slots=3)
    started = time.perf_counter()
    with UpstreamCapture(linears, names) as cap:
        for i, ids in enumerate(samples):
            cap.holdout = args.probe_every > 0 and i % args.probe_every == args.probe_every - 1
            cap.start_sample()
            with using(current().replace(w3_decode=False)):  # every call FP8, the head's last short slice too
                out = model.prefill(torch.tensor(ids, device=model.device))
                for r in range(0, len(ids), 1024):
                    model.logits(out["norm"][r:r + 1024])
            del out
            cap.switch()
            ex.flush_cache()
            req = ex.admit(list(ids), SamplingParams(max_new_tokens=1, trace=False))
            try:
                while req.prefill_pos < req.prefill_len:
                    batch = ex._prepare_single_prefill(req, 64)
                    out = model.forward(batch.ids, batch.context)
                    model.logits(out["norm"])
                    ex._finish_single_prefill(batch, out)
            finally:
                req.cacheable = False
                ex.release(req)
            cap.finish_sample()
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(samples)} samples ({time.perf_counter() - started:.0f}s)", flush=True)
        cap.fp = {}
    torch.cuda.synchronize()
    for name in sorted(names):
        if name not in cap.h:
            raise RuntimeError(f"{name} was never called on the calibration path")
        empty = torch.zeros((0, cap.h[name].shape[0]), dtype=torch.bfloat16)
        probes = torch.cat(cap.probes.pop(name)).cpu() if name in cap.probes else empty
        probes_fp = torch.cat(cap.probes_fp.pop(name)).cpu() if name in cap.probes_fp else empty
        save_file({"h": cap.h.pop(name).cpu(), "q": cap.q.pop(name).cpu(), "probes": probes.contiguous(),
                   "probes_fp": probes_fp.contiguous(), "rows": torch.tensor([cap.rows[name]], dtype=torch.int64)},
                  str(act / f"{name}.safetensors"))
    if all((act / f"{n}.safetensors").exists() for n, _ in linears):
        (act / "capture.json").write_text(json.dumps({
            "target": t.name, "target_revision": t.model_revision, "inputs": str(args.inputs), "inputs_sha256": _sha256(args.inputs),
            "upstream": str(args.checkpoint), "upstream_identity": model.w3_identity,
            "samples": len(samples), "probe_every": args.probe_every,
            "tokens": sum(len(ids) for i, ids in enumerate(samples) if not (args.probe_every and i % args.probe_every == args.probe_every - 1)),
            "probe_tokens": sum(len(ids) for i, ids in enumerate(samples) if args.probe_every and i % args.probe_every == args.probe_every - 1),
            }, indent=1) + "\n")
    print(f"captured {len(names)} projections in {time.perf_counter() - started:.0f}s -> {act}", flush=True)


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


def asymmetric_errors(weight: torch.Tensor, approx: torch.Tensor, probes: torch.Tensor, probes_fp: torch.Tensor) -> float:
    """Held-out ||x~ W~^T - x W^T||^2 / ||x W^T||^2: the copy read by the 3-bit model's input against the FP8 model."""
    if not probes.shape[0]:
        return float("nan")
    p, pf = probes.float(), probes_fp.float()
    ref = pf @ weight.T
    return float((p @ approx.T - ref).square().sum() / ref.square().sum())


@torch.inference_mode()
def fit(args) -> None:
    from safetensors.torch import load_file

    from sharar import target as targets
    from sharar.engine.config import current, install
    from sharar.kernels import w3
    from sharar.weights.modelopt import load_fp8_linear
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_calibration import quantize
    from sharar.weights.w3_checkpoint import FORMAT_VERSION, save_weight
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
    grid = [float(x) for x in args.scale_grid.split(",")]
    names = [n for n in projection_names(t) if not (skip and skip.search(n)) and not (args.only and not re.search(args.only, n))]
    manifest = {"format": FORMAT_VERSION, "target": t.name, "target_revision": t.model_revision,
                "source": "served FP8 projections (load_fp8_linear)", "method": args.method,
                "inputs_sha256": captured["inputs_sha256"], "samples": captured["samples"], "tokens": captured["tokens"],
                "probe_tokens": captured["probe_tokens"], "damping": args.damping, "unbias": args.unbias,
                "rows_per_chunk": args.rows_per_chunk, "exclude": args.exclude, "act_order": args.act_order,
                "scale_grid": grid, "only": args.only, "asym_blend": args.asym_blend,
                "upstream": captured.get("upstream"), "weights": {}, "reports": {}}
    identity = ("format", "target_revision", "method", "inputs_sha256", "damping", "unbias", "rows_per_chunk", "exclude",
                "act_order", "scale_grid", "only", "asym_blend")
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
        source = weight
        if "q" in a:
            # an upstream capture (`UpstreamCapture`): the least-squares target of x~ -> x W^T is W Q H~^-1; written
            # W + W (Q - H~) H~^-1 with the damped H~, the ridge pulls the correction (not the weights) toward zero,
            # so identical inputs give W back
            damped = h.clone()
            damped.diagonal().add_(args.damping * h.diagonal().mean())
            delta = a["q"] / float(a["rows"].item()) - h
            correction = torch.cholesky_solve(delta.T @ weight.T, torch.linalg.cholesky(damped)).T
            del damped, delta
            source = weight + args.asym_blend * correction
            del correction
        nearest = w3.pack(weight)
        report = {"rows": int(a["rows"].item()), "probe_rows": int(a["probes"].shape[0]),
                  "nearest": errors(weight, w3.dequantize(nearest), h, a["probes"])}
        if args.method == "gptq":
            del nearest
            packed, fitted = quantize(source, h, damping=args.damping, unbias=args.unbias, rows_per_chunk=args.rows_per_chunk,
                                      act_order=args.act_order, ratios=tuple(grid))
            report.update(fitted)
            report["gptq"] = errors(weight, w3.dequantize(packed), h, a["probes"])
            if "q" in a:
                report["asymmetric"] = {
                    "upstream_only": asymmetric_errors(weight, weight, a["probes"], a["probes_fp"]),
                    "copy": asymmetric_errors(weight, w3.dequantize(packed), a["probes"], a["probes_fp"]),
                    "source_shift": float((source - weight).norm() / weight.norm())}
                print(f"    held-out vs FP8 model: upstream only {report['asymmetric']['upstream_only']:.4f}, "
                      f"with the copy {report['asymmetric']['copy']:.4f} (target moved {report['asymmetric']['source_shift']:.4f})", flush=True)
        else:
            packed = nearest
        manifest["weights"][name] = save_weight(out / f"{name}.safetensors", packed)
        manifest["reports"][name] = report
        chosen = report.get("gptq", report["nearest"])
        print(f"  {name}: weight {report['nearest']['weight']:.4f} -> {chosen['weight']:.4f}, output (cov) "
              f"{report['nearest']['output_cov']:.4f} -> {chosen['output_cov']:.4f}, held-out "
              f"{report['nearest'].get('output_probe', float('nan')):.4f} -> {chosen.get('output_probe', float('nan')):.4f} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        del weight, a, h, packed, source
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
    c.add_argument("--inputs", type=Path, required=True, help="frozen samples: {'samples': [{'input_ids': [...]}, ...]} (nemotron_w3_inputs)")
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--probe-every", type=int, default=8, help="hold every N-th sample out of the covariances as probes (0: none)")
    c.add_argument("--limit", type=int, default=0, help="first N samples only (0: all)")
    u = sub.add_parser("capture-upstream", help="FP8 and 3-bit-upstream inputs of every projection, paired (asymmetric fit; loads the model)")
    u.add_argument("--inputs", type=Path, required=True)
    u.add_argument("--activations", type=Path, required=True)
    u.add_argument("--checkpoint", type=Path, required=True, help="the 3-bit copies serving upstream")
    u.add_argument("--probe-every", type=int, default=8)
    u.add_argument("--limit", type=int, default=0)
    u.add_argument("--only", default="", help="regex: these projections only (the accumulators of all 97 take about 27 GB; "
                                               "run twice, the second skips what the first wrote)")
    f = sub.add_parser("fit", help="GPTQ (or nearest) copies from a capture (no model load)")
    f.add_argument("--activations", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--method", choices=("gptq", "nearest"), default="gptq")
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--unbias", choices=("weight", "output"), help="per-row gain correction through the row scales (w3_calibration.quantize)")
    f.add_argument("--rows-per-chunk", type=int, default=1024)
    f.add_argument("--exclude", default="", help="regex of names to leave FP8 (e.g. lm_head)")
    f.add_argument("--act-order", action="store_true", help="GPTQ in decreasing input energy, static groups (w3_calibration.quantize)")
    f.add_argument("--scale-grid", default="0.6,0.7,0.8,0.9,1.0", help="clipping ratios the group scales try")
    f.add_argument("--only", default="", help="regex: fit only these names (the manifest covers them alone; the rest stay FP8)")
    f.add_argument("--device", default="cuda", help="cpu fits without the GPU (about 15 minutes for all 97 on the GB10's cores)")
    f.add_argument("--threads", type=int, default=16, help="CPU threads with --device cpu")
    f.add_argument("--asym-blend", type=float, default=1.0, help="upstream capture: how far toward the corrected target (0: W)")
    f.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    {"capture": capture, "capture-upstream": capture_upstream, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
