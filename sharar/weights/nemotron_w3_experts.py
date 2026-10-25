# SPDX-License-Identifier: Apache-2.0
"""3-bit routed experts for Nemotron-3.5-Super's W3 checkpoint: every latent MoE block's 512 experts (the 40 trunk
blocks and the MTP head's) as `layers.N.experts.w1` / `.w2` and `mtp.experts.w1` / `.w2`.

The source is the checkpoint's NVFP4 experts, dequantized (there is no BF16 expert source), as for Flash-Next
(`flashnext_w3_experts.py`): each expert's up [2688, 1024] (fc1, the latent width in, the checkpoint's row order) and
down [1024, 2688] (fc2) into the W3 format (an eight-level E4M3 codebook, an E4M3 scale per 32 weights, an FP32 row
multiplier), served by `kernels/w3_moe.py`'s ReLU^2 geometry (`model/nemotron/moe.py`, SHARAR_W3_DECODE=1
SHARAR_W3_CHECKPOINT=DIR). The samples are `nemotron_w3_inputs` with `--knowledge`.

- `capture`: prefills the samples through the serving model (FP8 trunk, NVFP4 experts) and stores, for every MoE
  block, the expert inputs as the block's routing produced them: the BF16 latent rows [rows, 1024], the top-22 ids and
  their renormalized weights. The MTP head's block sees the head's teacher-forced input over the same samples (token
  t+1's embedding with the target's final hidden of t, through the head's attention over the sample, causal), as a
  depth-1 draft step would. About 2.2 KB a row a block; resumable after whole samples (`--resume`).
- `fit`: one block at a time. Each matrix of a block (all experts' up, all experts' down) takes the codebook of
  `--codebooks` with the least cheap-fit output error: Lloyd's for dense matrices, `zero8` (eight levels with a zero)
  for the sparse down projections of the early blocks. Then GPTQ on the routed rows (per expert H1 = sum g^2 x x^T
  over its routed latent rows for fc1, H2 the same over h = relu(x up^T)^2 from the NVFP4 up for fc2, g the routing
  weight; `flashnext_w3_experts.gptq_batched` over a chunk of experts). Experts with fewer than --min-tokens rows keep
  the cheap fit (nearest level after a weight-error scale grid), and sparse experts' covariances are shrunk toward the
  block's pooled one. Each finished block is recorded in progress.json (resumable); only a complete run publishes
  manifest.json. `--source` copies another W3 checkpoint's (trunk) entries in, verified.

    python3 -m sharar.weights.nemotron_w3_experts capture --inputs nemotron-expert-inputs.json --activations ACT
    python3 -m sharar.weights.nemotron_w3_experts fit --activations ACT --source TRUNK --output DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

LAYER_PREFIX = "language_model.backbone.layers"
MTP_PREFIX = "language_model.mtp.layers.1.mixer"
CHOICE_EVERY = 4  # the codebook choice looks at every 4th chunk of experts


# ------------------------------------------------------------------ capture

def block_names(target) -> list[str]:
    """Every MoE block's checkpoint-independent name: `layers.N` for the trunk's, `mtp` for the MTP head's."""
    from sharar.model.nemotron import config as c
    return [f"layers.{i}" for i, kind in enumerate(c.BLOCK_TYPES) if kind == "moe"] + ["mtp"]


def _files(out: Path, name: str) -> dict[str, Path]:
    return {k: out / f"{name}.{k}.bin" for k in ("lat", "ids", "w")}


@torch.inference_mode()
def capture(args) -> None:
    import numpy as np
    from sharar.engine.config import current, install
    from sharar.model.nemotron import config as c
    from sharar.model.nemotron.model import Nemotron
    from sharar.model.nemotron.moe import NemotronMoe
    from sharar.model.nemotron.mtp import NemotronMtpHead
    from sharar.target import NEMOTRON
    from sharar.weights.nemotron_w3_inputs import load_samples
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_checkpoint import sha256
    install(current().replace(target="nemotron", w3_decode=False, w3_checkpoint=None))
    out = args.activations
    if (out / "capture.json").exists():
        raise SystemExit("refusing to overwrite a finished capture")
    out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.inputs, args.limit)
    names = block_names(NEMOTRON)
    progress_path = out / "progress.json"
    done = json.loads(progress_path.read_text()) if args.resume and progress_path.is_file() else {"samples": 0, "rows": {n: 0 for n in names}}
    if args.resume:  # cut every file back to the last whole sample
        for n in names:
            for k, p in _files(out, n).items():
                width = {"lat": 2 * c.MOE_LATENT_SIZE, "ids": 2 * NEMOTRON.experts_per_token, "w": 4 * NEMOTRON.experts_per_token}[k]
                if p.exists():
                    with p.open("r+b") as f:
                        f.truncate(done["rows"][n] * width)
        print(f"resuming after {done['samples']} samples", flush=True)
    model = Nemotron(NEMOTRON.snapshot_dir())
    w = SnapshotWeights(NEMOTRON.snapshot_dir())
    head = NemotronMtpHead(w, model.device, model, num_pages=16)
    w.close()
    index = {id(layer.mixer): f"layers.{layer.index}" for layer in model.layers if layer.kind == "moe"}
    index[id(head.moe)] = "mtp"
    handles = {n: {k: p.open("ab" if args.resume else "wb") for k, p in _files(out, n).items()} for n in names}
    rows = dict(done["rows"])
    original = NemotronMoe.route

    def route(moe, fused, ctx):
        ids, weights, latent = original(moe, fused, ctx)
        n = index.get(id(moe))
        if n is not None:
            f = handles[n]
            f["lat"].write(latent.contiguous().view(torch.int16).cpu().numpy().tobytes())
            f["ids"].write(ids.to(torch.int16).cpu().numpy().tobytes())
            f["w"].write(weights.float().contiguous().cpu().numpy().tobytes())
            rows[n] += ids.shape[0]
        return ids, weights, latent

    def mtp_inputs(ids: torch.Tensor, hidden: torch.Tensor) -> None:
        """The head's MoE over token t+1 with the target's final hidden of t, attention causal over the sample."""
        u = head._fuse(ids[1:], hidden[:-1])
        t = u.shape[0]
        qkv = head.qkv(head._attention_input(u))
        q, k, v = qkv.split([head.q_size, head.kv_size, head.kv_size], dim=-1)
        shape = lambda a, h: a.reshape(t, h, head.head_dim).transpose(0, 1)[None]  # noqa: E731
        a = torch.nn.functional.scaled_dot_product_attention(shape(q, head.num_heads), shape(k, head.num_kv_heads),
                                                             shape(v, head.num_kv_heads), is_causal=True, enable_gqa=True)
        a = head.o(a[0].transpose(0, 1).reshape(t, head.q_size).contiguous())
        x, _ = head.moe_norm.fused_add(a, u)
        head.moe(x, None)

    started = time.perf_counter()
    NemotronMoe.route = route
    try:
        for n, sample in enumerate(samples):
            if n < done["samples"]:
                continue
            ids = torch.tensor(sample, dtype=torch.long, device=model.device)
            res = model.prefill(ids)
            mtp_inputs(ids, res["norm"])
            del res
            if (n + 1) % 16 == 0 or n + 1 == len(samples):
                for f in handles.values():
                    for h in f.values():
                        h.flush()
                progress_path.write_text(json.dumps({"samples": n + 1, "rows": rows}))
                print(f"  captured {n + 1}/{len(samples)} samples, {rows[names[0]]} rows ({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        NemotronMoe.route = original
        for f in handles.values():
            for h in f.values():
                h.close()
    trunk = {rows[n] for n in names if n != "mtp"}
    if len(trunk) != 1 or rows["mtp"] != rows[names[0]] - len(samples):
        raise SystemExit(f"blocks saw unexpected row counts: {rows}")
    meta = {"inputs_sha256": sha256(args.inputs), "samples": len(samples), "rows": rows,
            "latent": c.MOE_LATENT_SIZE, "topk": NEMOTRON.experts_per_token, "blocks": names, "trunk": "fp8", "experts": "nvfp4",
            "mtp": "teacher-forced depth-1 head input", "seconds": round(time.perf_counter() - started, 1)}
    (out / "capture.json").write_text(json.dumps(meta, indent=1))
    progress_path.unlink(missing_ok=True)
    print(f"captured {rows[names[0]]} rows per block from {len(samples)} samples -> {out}", flush=True)
    del np


def load_rows(directory: Path, name: str, device):
    """A block's captured rows: (latent BF16 [R, 1024], ids int64 [R, 22], weights FP32 [R, 22])."""
    import numpy as np
    meta = json.loads((directory / "capture.json").read_text())
    r, k, lat = meta["rows"][name], meta["topk"], meta["latent"]
    f = _files(directory, name)
    x = torch.from_numpy(np.fromfile(f["lat"], dtype=np.int16, count=r * lat)).to(device).view(torch.bfloat16).reshape(r, lat)
    ids = torch.from_numpy(np.fromfile(f["ids"], dtype=np.int16, count=r * k)).to(device).long().reshape(r, k)
    w = torch.from_numpy(np.fromfile(f["w"], dtype=np.float32, count=r * k)).to(device).reshape(r, k)
    for p in f.values():  # read once: leave no page cache behind (the GB10's one memory pool also serves the engine)
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return x, ids, w


# ------------------------------------------------------------------ fit

def prefix_of(name: str) -> str:
    return MTP_PREFIX if name == "mtp" else f"{LAYER_PREFIX}.{name.split('.')[1]}.mixer"


@torch.no_grad()
def fit(args) -> None:
    from sharar.kernels import w3, w3_moe
    from sharar.model.nemotron import config as c
    from sharar.model.nemotron.moe import load_experts
    from sharar.target import NEMOTRON
    from sharar.weights.flashnext_w3_experts import dequant_codes, gptq_batched
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_checkpoint import FORMAT_VERSION, CalibratedWeights, save_weight, sha256
    torch.backends.cuda.matmul.allow_tf32 = False

    source = CalibratedWeights(args.source, revision=NEMOTRON.model_revision) if args.source else None
    if source is not None and any(".experts." in n for n in source.weights):
        raise SystemExit("the source checkpoint already carries experts")
    partial = args.output.with_name(args.output.name + ".partial")
    weights = {}
    partial.mkdir(parents=True, exist_ok=True)
    for name, entry in (source.weights.items() if source else ()):  # trunk entries unchanged, verified on the way
        src, dst = args.source / entry["file"], partial / entry["file"]
        if not dst.is_file() or sha256(dst) != entry["sha256"]:
            shutil.copyfile(src, dst)
            if sha256(dst) != entry["sha256"]:
                raise SystemExit(f"checksum mismatch copying {name}")
        weights[name] = entry
    capture_meta = json.loads((args.activations / "capture.json").read_text())
    # what the fitted matrices depend on: a resume continues only the same fit
    identity = {"capture": capture_meta["inputs_sha256"], "rows": capture_meta["rows"], "codebooks": args.codebooks,
                "chunk": args.chunk, "damping": args.damping, "min_tokens": args.min_tokens,
                "prior_tokens": args.prior_tokens, "source": source.identity if source else None}
    progress = partial / "progress.json"
    if progress.is_file():  # resume: blocks finished in an interrupted run (entries re-verified)
        previous = json.loads(progress.read_text())
        if previous.get("identity") != identity:
            raise SystemExit(f"{partial} holds another fit (other settings or capture): remove it or use a new output")
        for name, entry in previous["weights"].items():
            if (partial / entry["file"]).is_file() and sha256(partial / entry["file"]) == entry["sha256"]:
                weights[name] = entry

    dev = torch.device("cuda")
    w = SnapshotWeights(NEMOTRON.snapshot_dir())
    names = block_names(NEMOTRON)
    candidates = [w3._codebook(b) for b in args.codebooks.split(",")]
    book_names = dict(zip(candidates, args.codebooks.split(",")))
    ratios = w3_moe.RATIOS
    E, I, L = NEMOTRON.num_experts, NEMOTRON.moe_intermediate_size, c.MOE_LATENT_SIZE
    stats_path = args.output.with_name(args.output.name + ".stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    started = time.perf_counter()
    for name in names:
        if all(f"{name}.experts.{key}" in weights for key in ("w1", "w2")):
            continue
        if args.deadline and time.perf_counter() - started > args.deadline:
            print(f"deadline: stopping before {name} (resume with the same command)", flush=True)
            break
        block_started = time.perf_counter()
        nv = load_experts(w, prefix_of(name), E, dev)
        w.close()
        x, ids, gates = load_rows(args.activations, name, dev)
        flat = ids.reshape(-1)
        sort = torch.sort(flat, stable=True).indices
        counts = torch.bincount(flat, minlength=E).cpu()
        members = (sort // ids.shape[1], gates.reshape(-1)[sort], torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).tolist())
        del ids, gates, flat, sort
        st = {"tokens": x.shape[0], "routed_min": int(counts.min()), "routed_median": int(counts.median()),
              "routed_max": int(counts.max()), "cheap_fallback": 0}
        parts = {"w1": [], "w2": []}
        err = {f"{m}_{kind}_{key}": [0.0, 0.0] for m in ("cheap", "gptq") for kind in ("weight", "output") for key in ("w1", "w2")}
        chunk = args.chunk

        def chunk_cov(lo: int, hi: int) -> dict[str, torch.Tensor]:
            """The routing-weighted covariances of experts lo..hi (fc1 [c, L, L], fc2 [c, I, I], FP32); one chunk at a
            time keeps the fit small (all 512 experts' fc2 covariances would be 14.8 GB)."""
            rows_idx, gate_w, offsets = members
            t0 = time.perf_counter()
            cv = {"w1": torch.zeros((hi - lo, L, L), device=dev), "w2": torch.zeros((hi - lo, I, I), device=dev)}
            up = w3_moe.nvfp4_dequant(nv["w1"][lo:hi], nv["s1"][lo:hi], nv["g1"][lo:hi])  # [c, I, L]
            for j in range(hi - lo):
                a, b = offsets[lo + j], offsets[lo + j + 1]
                for s0 in range(a, b, 16384):
                    t = min(b, s0 + 16384)
                    xe = x[rows_idx[s0:t]].float()
                    g = gate_w[s0:t, None]
                    hh = torch.relu(xe @ up[j].T).square_().mul_(g)  # the NVFP4 up (dequantized), FP32
                    xe.mul_(g)
                    cv["w1"][j].addmm_(xe.T, xe)
                    cv["w2"][j].addmm_(hh.T, hh)
                    del xe, hh
            del up
            torch.cuda.synchronize()
            st["seconds_cov"] = round(st.get("seconds_cov", 0.0) + time.perf_counter() - t0, 1)
            return cv

        # pass 1, chunk by chunk: the block's pooled covariances and each matrix's codebook, the candidate with the
        # least cheap-fit output error over every CHOICE_EVERY-th chunk of experts
        num = {key: [0.0] * len(candidates) for key in ("w1", "w2")}
        sums = {"w1": torch.zeros((L, L), device=dev), "w2": torch.zeros((I, I), device=dev)}
        for lo in range(0, E, chunk):
            hi = min(E, lo + chunk)
            cv = chunk_cov(lo, hi)
            for key in ("w1", "w2"):
                sums[key] += cv[key].sum(0)
                if len(candidates) == 1 or (lo // chunk) % CHOICE_EVERY:
                    continue
                packed, sc, g = (nv["w1"], nv["s1"], nv["g1"]) if key == "w1" else (nv["w2"], nv["s2"], nv["g2"])
                weight = w3_moe.nvfp4_dequant(packed[lo:hi], sc[lo:hi], g[lo:hi])
                cn, n, k = weight.shape
                for j, lv in enumerate(candidates):
                    delta = dequant_codes(*w3_moe.quantize(weight.reshape(cn * n, k), lv, ratios), lv).reshape(cn, n, k) - weight
                    num[key][j] += torch.einsum("cnk,ckl,cnl->", delta, cv[key], delta).item()
                    del delta
                del weight
            del cv
        pooled = {key: s_ / s_.diagonal().sum().clamp_min(1e-30) for key, s_ in sums.items()}
        del sums
        books = {key: candidates[min(range(len(candidates)), key=num[key].__getitem__)] for key in ("w1", "w2")}
        if len(candidates) > 1:  # the choice's evidence: each candidate's error relative to the best
            st["choice"] = {key: {book_names[b]: round(v / max(min(num[key]), 1e-30), 4) for b, v in zip(candidates, num[key])}
                            for key in ("w1", "w2")}
        st["codebook_w1"], st["codebook_w2"] = book_names[books["w1"]], book_names[books["w2"]]
        for lo in range(0, E, chunk):
            hi = min(E, lo + chunk)
            cn = hi - lo
            mats = {"w1": w3_moe.nvfp4_dequant(nv["w1"][lo:hi], nv["s1"][lo:hi], nv["g1"][lo:hi]),   # [c, I, L]
                    "w2": w3_moe.nvfp4_dequant(nv["w2"][lo:hi], nv["s2"][lo:hi], nv["g2"][lo:hi])}   # [c, L, I]
            hess = chunk_cov(lo, hi)  # pass 2 recomputes the chunk's covariances
            for key, weight in mats.items():
                levels = books[key]
                n, k = weight.shape[1:]
                codes_c, scales_c, rows_c = w3_moe.quantize(weight.reshape(cn * n, k), levels, ratios)  # the cheap fit
                deq_c = dequant_codes(codes_c, scales_c, rows_c, levels).reshape(cn, n, k)
                hk = hess[key]
                cnt = counts[lo:hi].to(dev).float()
                hk_fit = hk.clone()
                if args.prior_tokens:
                    beta = (args.prior_tokens / (cnt + args.prior_tokens))[:, None, None]
                    tr = hk.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]
                    hk_fit += beta * tr * pooled[key][None]
                use = cnt >= args.min_tokens
                hk_fit[~use] = torch.eye(k, device=dev)  # placeholders; these experts keep the cheap fit
                t0 = time.perf_counter()
                codes_g, scales_g, rows_g = gptq_batched(weight, hk_fit, levels, ratios, args.damping)
                torch.cuda.synchronize()
                st["seconds_gptq"] = round(st.get("seconds_gptq", 0.0) + time.perf_counter() - t0, 1)
                del hk_fit
                keep = use[:, None, None].expand(cn, n, k)
                codes_g = torch.where(keep, codes_g, codes_c.reshape(cn, n, k))
                scales_g = torch.where(use[:, None, None], scales_g.view(torch.uint8),
                                       scales_c.reshape(cn, n, -1).view(torch.uint8)).view(torch.float8_e4m3fn)
                rows_g = torch.where(use[:, None], rows_g, rows_c.reshape(cn, n))
                if key == "w1":
                    st["cheap_fallback"] += int((~use).sum())
                deq_g = dequant_codes(codes_g, scales_g, rows_g, levels)
                for m, deq in (("cheap", deq_c), ("gptq", deq_g)):
                    delta = deq - weight
                    err[f"{m}_weight_{key}"][0] += delta.square().sum().item()
                    err[f"{m}_weight_{key}"][1] += weight.square().sum().item()
                    err[f"{m}_output_{key}"][0] += torch.einsum("cnk,ckl,cnl->", delta, hk, delta).item()
                    err[f"{m}_output_{key}"][1] += torch.einsum("cnk,ckl,cnl->", weight, hk, weight).item()
                    del delta
                parts[key].append(w3.pack_codes(codes_g.reshape(cn * n, k), scales_g.reshape(cn * n, -1), rows_g.reshape(-1),
                                                levels, rows_per_chunk=n))
                del deq_c, deq_g
            del mats, hess
        for key, n, k in (("w1", I, L), ("w2", L, I)):
            levels = books[key]
            stacked = w3.W3Weight(*(torch.cat([getattr(p, a) for p in parts[key]]) for a in ("planes", "scales", "row_scale")),
                                  E * n, k, levels, w3.lut_of(levels))
            entry = f"{name}.experts.{key}"
            weights[entry] = dict(save_weight(partial / f"{entry}.safetensors", stacked), experts=E)
        progress.write_text(json.dumps({"identity": identity, "weights": weights}))
        st.update({k: round((v[0] / v[1]) ** 0.5, 5) if v[1] else None for k, v in err.items()})
        st["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        st["seconds"] = round(time.perf_counter() - block_started, 1)
        torch.cuda.reset_peak_memory_stats()
        stats[name] = st
        stats_path.write_text(json.dumps(stats, indent=1))
        del nv, x, members, parts
        torch.cuda.empty_cache()
        print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in st.items() if v is not None) + f" ({time.perf_counter() - started:.0f}s)", flush=True)

    if any(f"{n}.experts.w1" not in weights for n in names):
        print("partial run: no manifest written", flush=True)
        return
    keys = sorted({k for s in stats.values() for k in s if k.startswith(("cheap_", "gptq_")) and s[k] is not None})
    mean = {k: round(sum(s[k] for s in stats.values() if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in stats.values())), 5)
            for k in keys}
    manifest = dict(source.manifest) if source else {"format": FORMAT_VERSION, "target": NEMOTRON.name,
                                                     "target_revision": NEMOTRON.model_revision}
    manifest["weights"] = weights
    manifest["experts"] = {
        "source": "checkpoint NVFP4 experts, dequantized", "codebooks": {book_names[b]: list(b) for b in candidates},
        "codebook_choice": f"per matrix, the least cheap-fit error over every {CHOICE_EVERY}th chunk of {args.chunk} experts "
                           "(stats: codebook_w1 / codebook_w2)", "group_size": w3.GROUP,
        "scale_grid": list(ratios), "fc1": "up_proj, checkpoint row order", "blocks": names,
        "trunk_from": str(args.source.resolve()) if source else None, "trunk_manifest_sha256": source.identity if source else None,
        "method": "gptq on routed activations (routing-weighted covariances), batched", "damping": args.damping,
        "min_tokens": args.min_tokens, "prior_tokens": args.prior_tokens,
        "capture": capture_meta,
        "relative_error_mean": {k: v for k, v in mean.items() if k.startswith("gptq_")},
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=1))
    progress.unlink(missing_ok=True)
    partial.rename(args.output)
    print(f"{json.dumps(manifest['experts']['relative_error_mean'])} -> {args.output} manifest {sha256(args.output / 'manifest.json')}", flush=True)
    print("mean relative errors:", json.dumps(mean), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="store every MoE block's routed latent rows over the calibration samples")
    c.add_argument("--inputs", type=Path, required=True, help="nemotron_w3_inputs output with --knowledge")
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--resume", action="store_true", help="continue an interrupted capture after its whole samples")
    f = sub.add_parser("fit", help="write the checkpoint")
    f.add_argument("--source", type=Path, help="a W3 checkpoint whose (trunk) entries are copied in")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--activations", type=Path, required=True, help="the capture directory")
    f.add_argument("--codebooks", default="lloyd,zero8", help="candidate codebooks (w3.CODEBOOKS), one chosen per matrix")
    f.add_argument("--chunk", type=int, default=16, help="experts fitted together (and covariances held at once)")
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--min-tokens", type=int, default=16, help="fewer routed rows keep the cheap fit")
    f.add_argument("--prior-tokens", type=float, default=256, help="covariance shrinkage toward the block's pool: weight n0 / (n + n0)")
    f.add_argument("--deadline", type=float, default=0, help="seconds: start no block after this (resume later)")
    args = ap.parse_args()
    {"capture": capture, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
