# SPDX-License-Identifier: Apache-2.0
"""3-bit routed experts for Flash-Next's W3 checkpoint (#73): a new checkpoint directory with the source
checkpoint's trunk entries and every layer's experts as `layers.N.experts.w13` / `.w2`.

The source is the checkpoint's NVFP4 experts, dequantized (there is no BF16 expert source). Each expert matrix is
fitted into the W3 format: the trunk's Lloyd codebook, one E4M3 scale per 32 weights, an FP32 row multiplier. fc1 is
stored in the kernel's row order (w3_moe.fc1_order). Serving it with the 3-bit copies on (the `fast` profile,
SHARAR_W3_CHECKPOINT=DIR) replaces the NVFP4 experts.

Two methods:
- `cheap` (the 2026-09-30 route, `w3_moe.quantize`): group scales on a clipping grid minimizing the weight error,
  nearest level. Needs no activations. It lost recall (MMLU-Pro law -13 pp).
- `gptq`: GPTQ (w3_calibration's column-error update and activation-weighted scale grid, batched over experts) on
  the inputs each expert is routed. `capture` prefills the frozen calibration samples through the serving model
  (FP8 trunk, NVFP4 experts) and stores every layer's MoE input rows on disk; `fit` then works one layer at a time:
  it routes the rows with the layer's router, accumulates per expert H = sum_t g_t^2 x_t x_t^T (g: the token's
  routing weight, so the fit weighs the error as the layer's output does) for fc1, and the same over
  h = silu(gate) * up from the NVFP4 fc1 for fc2. An expert that saw fewer than --min-tokens routed rows keeps
  the cheap fit; below --prior-tokens its covariance is shrunk toward the layer's pooled one.

    SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts capture --inputs expert-inputs.json --activations DIR
    SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts fit --method gptq --activations DIR \\
        --source $SHARAR_RESULTS/w3-checkpoints/flashnext-current --output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3-gptq \\
        [--cheap-output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

PREFIX = "model.language_model"


# ------------------------------------------------------------------ capture

@torch.inference_mode()
def capture(args) -> None:
    import numpy as np
    from sharar.engine.config import current, install
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.model.flashnext.model import FlashNext
    from sharar.model.flashnext.moe import MoeLayer
    from sharar.target import FLASHNEXT
    install(current().replace(flashnext_trunk="fp8", w3_decode=False, w3_checkpoint=None, flashnext_experts="nvfp4"))
    out = args.activations
    if (out / "capture.json").exists():
        raise SystemExit("refusing to overwrite a finished capture")
    out.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
    model = FlashNext(FLASHNEXT.snapshot_dir(), precision="fp8")
    ex = Executor(model, num_pages=args.pages, num_slots=4, max_context=8192, log=print)
    index = {id(layer.moe): layer.index for layer in model.layers}
    files = {i: open(out / f"layer{i:02d}.bin", "wb") for i in index.values()}
    rows = {i: 0 for i in index.values()}
    original = MoeLayer.__call__

    def call(moe, x):
        i = index.get(id(moe))
        if i is not None:
            flat = x.reshape(-1, x.shape[-1])
            files[i].write(flat.contiguous().view(torch.int16).cpu().numpy().tobytes())
            rows[i] += flat.shape[0]
        return original(moe, x)

    started = time.perf_counter()
    MoeLayer.__call__ = call
    try:
        for n, ids in enumerate(samples):
            req = ex.admit(list(ids), SamplingParams(max_new_tokens=1))
            try:
                ex.prefill(req)
            finally:
                ex.release(req)
            ex.flush_cache()
            if (n + 1) % 64 == 0:
                print(f"  captured {n + 1}/{len(samples)} samples, {rows[0]} rows ({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        MoeLayer.__call__ = original
        for f in files.values():
            f.close()
    if len(set(rows.values())) != 1:
        raise SystemExit(f"layers saw different row counts: {sorted(set(rows.values()))}")
    meta = {"inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(), "samples": len(samples),
            "rows": rows[0], "hidden": FLASHNEXT.hidden_size, "dtype": "bfloat16", "layers": sorted(rows),
            "trunk": "fp8", "experts": "nvfp4", "seconds": round(time.perf_counter() - started, 1)}
    (out / "capture.json").write_text(json.dumps(meta, indent=1))
    print(f"captured {rows[0]} rows per layer from {len(samples)} samples -> {out}", flush=True)
    del np


def load_rows(directory: Path, layer: int, device) -> torch.Tensor:
    import numpy as np
    meta = json.loads((directory / "capture.json").read_text())
    raw = np.fromfile(directory / f"layer{layer:02d}.bin", dtype=np.int16)
    return torch.from_numpy(raw).to(device).view(torch.bfloat16).reshape(meta["rows"], meta["hidden"])


# ------------------------------------------------------------------ batched GPTQ

@torch.no_grad()
def fit_scales_batched(weight: torch.Tensor, importance: torch.Tensor, levels, ratios):
    """w3_calibration.fit_scales over a batch: weight [C, N, K], importance [C, K] -> (E4M3 [C, N, K/32], row scales [C, N])."""
    from sharar.kernels import w3
    c, n, k = weight.shape
    grouped = weight.reshape(c, n, k // w3.GROUP, w3.GROUP)
    base = grouped.abs().amax(-1) / max(abs(x) for x in levels)
    row_scale = (base.amax(-1) / w3.FP8_MAX).clamp_min(1e-12)
    lv = torch.tensor(levels, dtype=torch.float32, device=weight.device)
    mids = (lv[1:] + lv[:-1]) / 2
    imp = importance.float().reshape(c, 1, k // w3.GROUP, w3.GROUP)
    best_loss = torch.full_like(base, float("inf"))
    best = torch.zeros_like(base)
    for ratio in ratios:
        scale = (base * ratio / row_scale[..., None]).clamp(0, w3.FP8_MAX).to(torch.float8_e4m3fn).float()
        effective = (scale * row_scale[..., None])[..., None]
        q = torch.bucketize(grouped / effective.clamp_min(1e-30), mids)
        loss = ((grouped - lv[q] * effective).square() * imp).sum(-1)
        improved = loss < best_loss
        best_loss = torch.minimum(loss, best_loss)
        best = torch.where(improved, scale, best)
        del q, loss, improved
    return best.to(torch.float8_e4m3fn), row_scale


class _BatchedBlock:
    """The 128-column GPTQ update for C experts at once, captured once as a CUDA graph and replayed."""

    def __init__(self, c, rows, levels, device):
        self.work = torch.zeros((c, rows, 128), device=device)
        self.inverse = torch.eye(128, device=device).repeat(c, 1, 1)
        self.scales = torch.ones((c, rows, 4), device=device)
        self.errors = torch.empty_like(self.work)
        self.codes = torch.empty((c, rows, 128), dtype=torch.int32, device=device)
        self.levels = torch.tensor(levels, device=device)
        self.mids = (self.levels[1:] + self.levels[:-1]) / 2
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            self._run()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run()

    def _run(self):
        for col in range(128):
            scale = self.scales[:, :, col // 32]
            value = self.work[:, :, col].clone()
            code = torch.bucketize((value / scale.clamp_min(1e-30)).contiguous(), self.mids, out_int32=True)
            error = (value - self.levels[code.long()] * scale) / self.inverse[:, col, col][:, None]
            self.codes[:, :, col].copy_(code)
            self.errors[:, :, col].copy_(error)
            self.work[:, :, col:].sub_(error[:, :, None] * self.inverse[:, col, col:][:, None, :])

    def __call__(self, weight, inverse, scales):
        self.work.copy_(weight)
        self.inverse.copy_(inverse)
        self.scales.copy_(scales)
        self.graph.replay()
        return self.codes, self.errors


_BLOCKS: dict = {}


@torch.no_grad()
def gptq_batched(weight: torch.Tensor, hessian: torch.Tensor, levels, ratios, damping: float = 0.01):
    """weight float32 [C, N, K], hessian float32 [C, K, K] (consumed) -> (codes int32 [C, N, K], E4M3 scales [C, N, K/32],
    row scales [C, N]): w3_calibration.quantize's GPTQ pass for every expert of the batch."""
    c, n, k = weight.shape
    diagonal = hessian.diagonal(dim1=1, dim2=2).clone()
    hessian.diagonal(dim1=1, dim2=2).add_(damping * diagonal.mean(-1, keepdim=True).clamp_min(1e-12))
    chol = torch.linalg.cholesky(hessian)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    del chol
    scales, row_scale = fit_scales_batched(weight, diagonal, levels, ratios)
    effective = scales.float() * row_scale[..., None]
    work = weight.clone()
    codes = torch.empty((c, n, k), dtype=torch.int32, device=weight.device)
    key = (c, n, weight.device)
    if key not in _BLOCKS:
        _BLOCKS[key] = _BatchedBlock(c, n, levels, weight.device)
    block = _BLOCKS[key]
    for col in range(0, k, 128):
        end = col + 128
        q, errors = block(work[:, :, col:end], inverse[:, col:end, col:end], effective[:, :, col // 32:end // 32])
        codes[:, :, col:end].copy_(q)
        if end < k:
            work[:, :, end:].sub_(torch.bmm(errors, inverse[:, col:end, end:]))
    del work, inverse
    return codes, scales, row_scale


def dequant_codes(codes, scales, row_scale, levels):
    lv = torch.tensor(levels, device=codes.device)
    eff = (scales.float() * row_scale[..., None]).repeat_interleave(32, dim=-1)
    return lv[codes.long()] * eff


# ------------------------------------------------------------------ fit

def route(x: torch.Tensor, router: torch.Tensor, topk: int):
    """MoeLayer.route for many rows: softmax top-k, renormalized."""
    probs = torch.softmax((x @ router.T).float(), dim=-1)
    weights, ids = torch.topk(probs, topk, dim=-1)
    return ids, weights / weights.sum(-1, keepdim=True)


@torch.no_grad()
def fit(args) -> None:
    from sharar.kernels import w3, w3_moe
    from sharar.model.flashnext.moe import load_nvfp4_experts
    from sharar.target import FLASHNEXT
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_checkpoint import CalibratedWeights, save_weight, sha256
    torch.backends.cuda.matmul.allow_tf32 = False

    source = CalibratedWeights(args.source, revision=FLASHNEXT.model_revision)
    if source.expert_layers():
        raise SystemExit("the source checkpoint already carries experts")
    outputs = {"gptq" if args.method == "gptq" else "cheap": args.output}
    if args.cheap_output:
        outputs["cheap"] = args.cheap_output
    partial = {m: d.with_name(d.name + ".partial") for m, d in outputs.items()}
    weights = {m: {} for m in outputs}
    for m, p in partial.items():
        p.mkdir(parents=True, exist_ok=True)
        for name, entry in source.weights.items():  # trunk entries unchanged, verified on the way
            src, dst = args.source / entry["file"], p / entry["file"]
            if not dst.is_file() or sha256(dst) != entry["sha256"]:
                shutil.copyfile(src, dst)
                if sha256(dst) != entry["sha256"]:
                    raise SystemExit(f"checksum mismatch copying {name}")
            weights[m][name] = entry
    progress = {m: p / "progress.json" for m, p in partial.items()}
    for m, path in progress.items():  # resume: layers finished in an interrupted run (entries re-verified)
        if path.is_file():
            done = json.loads(path.read_text())
            for name, entry in done["weights"].items():
                if (partial[m] / entry["file"]).is_file() and sha256(partial[m] / entry["file"]) == entry["sha256"]:
                    weights[m][name] = entry

    dev = torch.device("cuda")
    w = SnapshotWeights(FLASHNEXT.snapshot_dir())
    layers = [int(v) for v in args.layers.split(",")] if args.layers else list(range(FLASHNEXT.num_layers))
    levels = w3._codebook(args.codebook)
    ratios = w3_moe.RATIOS
    E, I, H, K = FLASHNEXT.num_experts, FLASHNEXT.moe_intermediate_size, FLASHNEXT.hidden_size, FLASHNEXT.experts_per_token
    order = w3_moe.fc1_order(I, dev)
    stats_path = args.output.with_name(args.output.name + ".stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    started = time.perf_counter()
    for i in layers:
        if all(f"layers.{i}.experts.{key}" in weights[m] for m in outputs for key in ("w13", "w2")):
            continue
        prefix = f"{PREFIX}.layers.{i}.mlp"
        nv = load_nvfp4_experts(w, prefix, E, dev)
        router = w.get(f"{prefix}.gate.weight").to(dev)
        w.close()
        # Routed rows per expert: (row index, routing weight), from the captured MoE inputs.
        x, counts, members = None, torch.zeros(E, dtype=torch.long), None
        if args.activations:
            x = load_rows(args.activations, i, dev)
            ids, gates = [], []
            for lo in range(0, x.shape[0], 65536):
                a, b = route(x[lo:lo + 65536], router, K)
                ids.append(a)
                gates.append(b)
            ids, gates = torch.cat(ids), torch.cat(gates)
            flat = ids.reshape(-1)
            sort = torch.sort(flat, stable=True).indices
            counts = torch.bincount(flat, minlength=E).cpu()
            members = (sort // K, gates.reshape(-1)[sort], torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).tolist())
            del ids, gates, flat, sort
        layer_stats = {"tokens": x.shape[0] if x is not None else 0, "routed_min": int(counts.min()), "routed_median": int(counts.median()),
                       "routed_max": int(counts.max()), "cheap_fallback": 0}
        parts = {m: {"w13": [], "w2": []} for m in outputs}
        err = {f"{m}_{kind}_{key}": [0.0, 0.0] for m in ("cheap", "gptq") for kind in ("weight", "output") for key in ("w13", "w2")}
        chunk = args.chunk
        # Per-expert covariances for the whole layer first (fc1 13.4 GB, fc2 0.8 GB in FP32), then their pools.
        cov = None
        if members is not None:
            rows_idx, gate_w, offsets = members
            cov = {"w13": torch.zeros((E, H, H), device=dev), "w2": torch.zeros((E, I, I), device=dev)}
            for lo in range(0, E, chunk):
                hi = min(E, lo + chunk)
                w13 = w3_moe.nvfp4_dequant(nv["w13"][lo:hi], nv["s13"][lo:hi], nv["scale2_13"][lo:hi])  # [c, 2I, H] = [up; gate]
                for j in range(hi - lo):
                    a, b = offsets[lo + j], offsets[lo + j + 1]
                    for s0 in range(a, b, 32768):
                        t = min(b, s0 + 32768)
                        xe = x[rows_idx[s0:t]].float()
                        g = gate_w[s0:t, None]
                        y = xe @ w13[j].T  # the NVFP4 fc1 (dequantized), FP32
                        hh = (torch.nn.functional.silu(y[:, I:]) * y[:, :I]) * g
                        xe.mul_(g)
                        cov["w13"][lo + j].addmm_(xe.T, xe)
                        cov["w2"][lo + j].addmm_(hh.T, hh)
                        del xe, y, hh
                del w13
            pooled = {key: (m_.sum(0) / m_.sum(0).diagonal().sum().clamp_min(1e-30)) for key, m_ in cov.items()}
        for lo in range(0, E, chunk):
            hi = min(E, lo + chunk)
            c = hi - lo
            w13 = w3_moe.nvfp4_dequant(nv["w13"][lo:hi], nv["s13"][lo:hi], nv["scale2_13"][lo:hi])[:, order]  # [c, 2I, H], kernel row order
            w2 = w3_moe.nvfp4_dequant(nv["w2"][lo:hi], nv["s2"][lo:hi], nv["scale2_2"][lo:hi])  # [c, H, I]
            hess = {key: m_[lo:hi] for key, m_ in cov.items()} if cov is not None else None
            for key, weight in (("w13", w13), ("w2", w2)):
                n, k = weight.shape[1:]
                flatw = weight.reshape(c * n, k)
                codes_c, scales_c, rows_c = w3_moe.quantize(flatw, levels, ratios)  # the cheap fit
                deq_c = dequant_codes(codes_c, scales_c, rows_c, levels).reshape(c, n, k)
                result = {"cheap": (codes_c, scales_c, rows_c, deq_c)}
                if args.method == "gptq" and hess is not None:
                    hk = hess[key]
                    cnt = counts[lo:hi].to(dev).float()
                    hk_fit = hk.clone()
                    if args.prior_tokens:
                        # Shrink sparse experts' covariance toward the layer's pooled one (unit trace, scaled to theirs).
                        beta = (args.prior_tokens / (cnt + args.prior_tokens))[:, None, None]
                        tr = hk.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]
                        hk_fit += beta * tr * pooled[key][None]
                    use = (cnt >= args.min_tokens)
                    hk_fit[~use] = torch.eye(k, device=dev)  # placeholders; these experts keep the cheap fit
                    codes_g, scales_g, rows_g = gptq_batched(weight, hk_fit, levels, ratios, args.damping)
                    del hk_fit
                    keep = use[:, None, None].expand(c, n, k)
                    codes_g = torch.where(keep, codes_g, codes_c.reshape(c, n, k))
                    scales_g = torch.where(use[:, None, None], scales_g.view(torch.uint8), scales_c.reshape(c, n, -1).view(torch.uint8)).view(torch.float8_e4m3fn)
                    rows_g = torch.where(use[:, None], rows_g, rows_c.reshape(c, n))
                    if key == "w13":
                        layer_stats["cheap_fallback"] += int((~use).sum())
                    deq_g = dequant_codes(codes_g, scales_g, rows_g, levels)
                    result["gptq"] = (codes_g.reshape(c * n, k), scales_g.reshape(c * n, -1), rows_g.reshape(-1), deq_g)
                for m, (codes, scales, rows, deq) in result.items():
                    delta = deq - weight
                    err[f"{m}_weight_{key}"][0] += delta.square().sum().item()
                    err[f"{m}_weight_{key}"][1] += weight.square().sum().item()
                    if hess is not None:
                        hk = hess[key]
                        err[f"{m}_output_{key}"][0] += torch.einsum("cnk,ckl,cnl->", delta, hk, delta).item()
                        err[f"{m}_output_{key}"][1] += torch.einsum("cnk,ckl,cnl->", weight, hk, weight).item()
                    if m in parts:
                        parts[m][key].append(w3.pack_codes(codes.reshape(c * n, k), scales.reshape(c * n, -1), rows.reshape(-1), levels, rows_per_chunk=n))
                    del delta
                del result
            del w13, w2, hess
        del cov
        for m, d in partial.items():
            for key, n, k in (("w13", 2 * I, H), ("w2", H, I)):
                ms = parts[m][key]
                stacked = w3.W3Weight(*(torch.cat([getattr(p, a) for p in ms]) for a in ("planes", "scales", "row_scale")),
                                      E * n, k, levels, w3.lut_of(levels))
                name = f"layers.{i}.experts.{key}"
                weights[m][name] = dict(save_weight(d / f"{name}.safetensors", stacked), experts=E)
            progress[m].write_text(json.dumps({"weights": weights[m]}))
        layer_stats.update({k: round((v[0] / v[1]) ** 0.5, 5) if v[1] else None for k, v in err.items()})
        stats[str(i)] = layer_stats
        stats_path.write_text(json.dumps(stats, indent=1))
        del nv, x, members, parts
        torch.cuda.empty_cache()
        print(f"layer {i}: " + ", ".join(f"{k} {v}" for k, v in layer_stats.items()) + f" ({time.perf_counter() - started:.0f}s)", flush=True)

    if any(f"layers.{i}.experts.w13" not in weights[m] for m in outputs for i in range(FLASHNEXT.num_layers)):
        print("partial run: no manifest written", flush=True)
        return
    keys = [k for k in next(iter(stats.values())) if k.startswith(("cheap_", "gptq_"))]
    mean = {k: round(sum(s[k] for s in stats.values() if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in stats.values())), 5) for k in keys}
    capture_meta = json.loads((args.activations / "capture.json").read_text()) if args.activations else None
    for m, d in partial.items():
        manifest = dict(source.manifest)
        manifest["weights"] = weights[m]
        manifest["experts"] = {
            "source": "checkpoint NVFP4 experts, dequantized", "codebook": list(levels), "group_size": w3.GROUP,
            "scale_grid": list(ratios), "fc1_rows": "w3_moe.fc1_order", "trunk_from": str(args.source.resolve()),
            "trunk_manifest_sha256": source.identity,
            **({"method": "nearest level, group scales on a weight-error grid"} if m == "cheap" else
               {"method": "gptq on routed activations (routing-weighted covariances), batched", "damping": args.damping,
                "min_tokens": args.min_tokens, "prior_tokens": args.prior_tokens, "capture": capture_meta}),
            "relative_error_mean": {k: v for k, v in mean.items() if k.startswith(m + "_")},
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        progress[m].unlink(missing_ok=True)
        d.rename(outputs[m])
        print(f"{m}: {json.dumps(manifest['experts']['relative_error_mean'])} -> {outputs[m]} manifest {sha256(outputs[m] / 'manifest.json')}", flush=True)
    print("mean relative errors:", json.dumps(mean), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="store every layer's MoE input rows over the calibration samples")
    c.add_argument("--inputs", type=Path, required=True)
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--pages", type=int, default=4096)
    f = sub.add_parser("fit", help="write the checkpoint")
    f.add_argument("--method", choices=("cheap", "gptq"), default="gptq")
    f.add_argument("--source", type=Path, required=True, help="W3 checkpoint whose trunk entries are copied")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--cheap-output", type=Path, help="also write the cheap-route checkpoint from the same pass")
    f.add_argument("--activations", type=Path, help="capture directory (gptq; with cheap, only for the output-error report)")
    f.add_argument("--codebook", default="lloyd")
    f.add_argument("--layers", default="", help="comma-separated subset (a partial checkpoint does not serve)")
    f.add_argument("--chunk", type=int, default=32, help="experts fitted together")
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--min-tokens", type=int, default=16, help="fewer routed rows keep the cheap fit")
    f.add_argument("--prior-tokens", type=float, default=256, help="covariance shrinkage toward the layer's pool: weight n0 / (n + n0)")
    args = ap.parse_args()
    if args.command == "capture":
        capture(args)
    else:
        if args.method == "gptq" and not args.activations:
            raise SystemExit("gptq needs --activations (run capture first)")
        fit(args)


if __name__ == "__main__":
    main()
