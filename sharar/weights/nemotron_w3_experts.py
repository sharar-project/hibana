# SPDX-License-Identifier: Apache-2.0
"""3-bit routed experts for Nemotron-3.5-Super's W3 checkpoint: every latent MoE block's 512 experts (the 40 trunk
blocks and the MTP head's) as `layers.N.experts.w1` / `.w2` and `mtp.experts.w1` / `.w2`.

The source is the checkpoint's NVFP4 experts, dequantized (there is no BF16 expert source), as for Flash-Next
(`flashnext_w3_experts.py`): each expert's up [2688, 1024] (fc1, the latent width in, the checkpoint's row order) and
down [1024, 2688] (fc2) into the W3 format (Lloyd codebook, an E4M3 scale per 32 weights, an FP32 row multiplier),
served by `kernels/w3_moe.py`'s ReLU^2 geometry (`model/nemotron/moe.py`, SHARAR_W3_DECODE=1 SHARAR_W3_CHECKPOINT=DIR).

- `inputs`: frozen samples in Nemotron's tokenizer: the coding proxy (`w3_calibration_inputs` built with Nemotron's
  tokenizer), reasoning traces (ShareGPT first turns with a model's thinking and answer, `flashnext_expert_inputs
  generate --thinking`) and knowledge answers (MMLU / MMLU-Pro validation questions answered with their reasoning),
  rendered in Nemotron's chat template (`<think>` reasoning `</think>` answer), seeded.
- `capture`: prefills the samples through the serving model (FP8 trunk, NVFP4 experts) and stores, for every MoE
  block, the expert inputs as the block's routing produced them: the BF16 latent rows [rows, 1024], the top-22 ids and
  their renormalized weights. The MTP head's block sees the head's teacher-forced input over the same samples (token
  t+1's embedding with the target's final hidden of t, through the head's attention over the sample, causal), as a
  depth-1 draft step would. About 2.2 KB a row a block; resumable after whole samples (`--resume`).
- `fit`: one block at a time. Each matrix of a block (all experts' up, all experts' down) takes the codebook of
  `--codebooks` with the least cheap-fit error (output error when activations are given, else weight error): Lloyd's
  for dense matrices, `zero8` (eight levels with a zero) for the sparse down projections of the early blocks.
  `--method cheap`: round to nearest after a weight-error scale grid, as `w3_moe.calibrate_relu2`; needs no
  activations. `--method gptq`: GPTQ on the routed rows (per expert H1 = sum g^2 x x^T over its routed latent rows for
  fc1, H2 the same over h = relu(x up^T)^2 from the NVFP4 up for fc2, g the routing weight;
  `flashnext_w3_experts.gptq_batched` over 32 experts). Experts with fewer than --min-tokens rows keep the cheap fit,
  and sparse experts' covariances are shrunk toward the block's pooled one. Each finished block is recorded in progress.json
  (resumable); only a complete run publishes manifest.json. `--source` copies another W3 checkpoint's (trunk) entries
  in, verified.

    python3 -m sharar.weights.nemotron_w3_experts inputs --tokenizer SNAP/tokenizer.json --code code.json \\
        --traces answers-sharegpt.json --knowledge answers.json --output nemotron-expert-inputs.json
    python3 -m sharar.weights.nemotron_w3_experts capture --inputs nemotron-expert-inputs.json --activations ACT
    python3 -m sharar.weights.nemotron_w3_experts fit --method gptq --activations ACT \\
        --output $SHARAR_RESULTS/w3-checkpoints/nemotron-experts3-<date> [--cheap-output DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch

LAYER_PREFIX = "language_model.backbone.layers"
MTP_PREFIX = "language_model.mtp.layers.1.mixer"
FORMAT_VERSION = 1


# ------------------------------------------------------------------ inputs

def render(item: dict) -> str:
    """One answered turn as Nemotron's chat template renders a finished thinking turn (empty system prompt)."""
    return (f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            f"<think>\n{item.get('reasoning') or ''}\n</think>\n{item.get('answer') or ''}<|im_end|>\n")


def inputs(args) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    code = json.loads(args.code.read_text())
    if code.get("tokenizer_sha256") != tok_sha:
        raise SystemExit("the coding proxy was tokenized with another tokenizer")
    samples = [{"source": s["source"], "input_ids": s["input_ids"]} for s in code["samples"]]
    rng = random.Random(args.seed)
    traces = json.loads(args.traces.read_text())["items"]
    for lang, want in (("en", args.traces_en), ("other", args.traces_other)):
        pool = [it for it in traces if it["source"].endswith(f"/{lang}") and (it.get("reasoning") or it.get("answer"))]
        rng.shuffle(pool)
        taken = 0
        for it in pool:
            if taken == want:
                break
            ids = tok.encode(render(it), add_special_tokens=False).ids
            if len(ids) < args.min_tokens:
                continue
            samples.append({"source": f"{it['source']}/{it.get('id')}", "input_ids": ids[:args.max_tokens]})
            taken += 1
    # knowledge answers are short: packed into samples of up to max_tokens, whole answers only
    know = [it for it in json.loads(args.knowledge.read_text())["items"] if (it.get("answer") or "").strip()]
    rng.shuffle(know)
    know = know[:args.knowledge_items]
    pack, packed = [], 0
    for it in know:
        ids = tok.encode(render(it), add_special_tokens=False).ids[:args.max_tokens]
        if pack and len(pack) + len(ids) > args.max_tokens:
            samples.append({"source": f"knowledge/{packed}", "input_ids": pack})
            pack, packed = [], packed + 1
        pack = pack + ids
    if pack:
        samples.append({"source": f"knowledge/{packed}", "input_ids": pack})
    out = {"version": "nemotron-expert-inputs-v1", "seed": args.seed, "max_tokens": args.max_tokens, "tokenizer_sha256": tok_sha,
           "code": {"commit": code["commit"], "sha256": hashlib.sha256(args.code.read_bytes()).hexdigest()},
           "traces_sha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(),
           "knowledge_sha256": hashlib.sha256(args.knowledge.read_bytes()).hexdigest(), "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"{len(samples)} samples, {sum(len(s['input_ids']) for s in samples)} tokens -> {args.output}")


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
    from sharar.weights.safetensors import SnapshotWeights
    install(current().replace(target="nemotron", w3_decode=False, w3_checkpoint=None, nemotron_experts="nvfp4"))
    out = args.activations
    if (out / "capture.json").exists():
        raise SystemExit("refusing to overwrite a finished capture")
    out.mkdir(parents=True, exist_ok=True)
    samples = [s["input_ids"] for s in json.loads(args.inputs.read_text())["samples"]]
    samples = samples[:args.limit] if args.limit else samples
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
    meta = {"inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(), "samples": len(samples), "rows": rows,
            "latent": c.MOE_LATENT_SIZE, "topk": NEMOTRON.experts_per_token, "blocks": names, "trunk": "fp8", "experts": "nvfp4",
            "mtp": "teacher-forced depth-1 head input", "seconds": round(time.perf_counter() - started, 1)}
    (out / "capture.json").write_text(json.dumps(meta, indent=1))
    progress_path.unlink(missing_ok=True)
    print(f"captured {rows[names[0]]} rows per block from {len(samples)} samples -> {out}", flush=True)
    del np


def load_rows(directory: Path, name: str, device, limit: int = 0):
    """A block's captured rows: (latent BF16 [R, 1024], ids int64 [R, 22], weights FP32 [R, 22])."""
    import numpy as np
    meta = json.loads((directory / "capture.json").read_text())
    r, k, lat = meta["rows"][name], meta["topk"], meta["latent"]
    r = min(r, limit) if limit else r
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
    from sharar.weights.w3_checkpoint import CalibratedWeights, save_weight, sha256
    torch.backends.cuda.matmul.allow_tf32 = False

    source = CalibratedWeights(args.source, revision=NEMOTRON.model_revision) if args.source else None
    if source is not None and any(".experts." in n for n in source.weights):
        raise SystemExit("the source checkpoint already carries experts")
    method = args.method
    outputs = {method: args.output}
    if args.cheap_output and method == "gptq":
        outputs["cheap"] = args.cheap_output
    partial = {m: d.with_name(d.name + ".partial") for m, d in outputs.items()}
    weights = {m: {} for m in outputs}
    for m, p in partial.items():
        p.mkdir(parents=True, exist_ok=True)
        for name, entry in (source.weights.items() if source else ()):  # trunk entries unchanged, verified on the way
            src, dst = args.source / entry["file"], p / entry["file"]
            if not dst.is_file() or sha256(dst) != entry["sha256"]:
                shutil.copyfile(src, dst)
                if sha256(dst) != entry["sha256"]:
                    raise SystemExit(f"checksum mismatch copying {name}")
            weights[m][name] = entry
    progress = {m: p / "progress.json" for m, p in partial.items()}
    for m, path in progress.items():  # resume: blocks finished in an interrupted run (entries re-verified)
        if path.is_file():
            for name, entry in json.loads(path.read_text())["weights"].items():
                if (partial[m] / entry["file"]).is_file() and sha256(partial[m] / entry["file"]) == entry["sha256"]:
                    weights[m][name] = entry

    dev = torch.device("cuda")
    w = SnapshotWeights(NEMOTRON.snapshot_dir())
    names = block_names(NEMOTRON)
    todo = args.blocks.split(",") if args.blocks else names
    candidates = [w3._codebook(b) for b in args.codebooks.split(",")]
    book_names = dict(zip(candidates, args.codebooks.split(",")))
    ratios = w3_moe.RATIOS
    E, I, L = NEMOTRON.num_experts, NEMOTRON.moe_intermediate_size, c.MOE_LATENT_SIZE
    stats_path = args.output.with_name(args.output.name + ".stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    started = time.perf_counter()
    for name in todo:
        if all(f"{name}.experts.{key}" in weights[m] for m in outputs for key in ("w1", "w2")):
            continue
        if args.deadline and time.perf_counter() - started > args.deadline:
            print(f"deadline: stopping before {name} (resume with the same command)", flush=True)
            break
        block_started = time.perf_counter()
        nv = load_experts(w, prefix_of(name), E, dev)
        w.close()
        counts, members, x = torch.zeros(E, dtype=torch.long), None, None
        if args.activations:
            x, ids, gates = load_rows(args.activations, name, dev, args.max_rows)
            flat = ids.reshape(-1)
            sort = torch.sort(flat, stable=True).indices
            counts = torch.bincount(flat, minlength=E).cpu()
            members = (sort // ids.shape[1], gates.reshape(-1)[sort], torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).tolist())
            del ids, gates, flat, sort
        st = {"tokens": x.shape[0] if x is not None else 0, "routed_min": int(counts.min()), "routed_median": int(counts.median()),
              "routed_max": int(counts.max()), "cheap_fallback": 0}
        parts = {m: {"w1": [], "w2": []} for m in outputs}
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
        # least cheap-fit error over every --choice-every-th chunk of experts (output error with activations, else weight error)
        pooled = None
        num = {key: [0.0] * len(candidates) for key in ("w1", "w2")}
        if members is not None or len(candidates) > 1:
            sums = {"w1": torch.zeros((L, L), device=dev), "w2": torch.zeros((I, I), device=dev)} if members is not None else None
            for lo in range(0, E, chunk):
                hi = min(E, lo + chunk)
                cv = chunk_cov(lo, hi) if members is not None else None
                for key in ("w1", "w2"):
                    if cv is not None:
                        sums[key] += cv[key].sum(0)
                    if len(candidates) == 1 or (lo // chunk) % args.choice_every:
                        continue
                    packed, sc, g = (nv["w1"], nv["s1"], nv["g1"]) if key == "w1" else (nv["w2"], nv["s2"], nv["g2"])
                    weight = w3_moe.nvfp4_dequant(packed[lo:hi], sc[lo:hi], g[lo:hi])
                    cn, n, k = weight.shape
                    for j, lv in enumerate(candidates):
                        delta = dequant_codes(*w3_moe.quantize(weight.reshape(cn * n, k), lv, ratios), lv).reshape(cn, n, k) - weight
                        num[key][j] += (torch.einsum("cnk,ckl,cnl->", delta, cv[key], delta).item() if cv is not None
                                        else delta.square().sum().item())
                        del delta
                    del weight
                del cv
            if sums is not None:
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
            hess = chunk_cov(lo, hi) if members is not None else None  # pass 2 recomputes the chunk's covariances
            for key, weight in mats.items():
                levels = books[key]
                n, k = weight.shape[1:]
                codes_c, scales_c, rows_c = w3_moe.quantize(weight.reshape(cn * n, k), levels, ratios)  # the cheap fit
                deq_c = dequant_codes(codes_c, scales_c, rows_c, levels).reshape(cn, n, k)
                result = {"cheap": (codes_c, scales_c, rows_c, deq_c)}
                if method == "gptq" and hess is not None:
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
                    result["gptq"] = (codes_g.reshape(cn * n, k), scales_g.reshape(cn * n, -1), rows_g.reshape(-1), deq_g)
                for m, (codes, scales, rows, deq) in result.items():
                    delta = deq - weight
                    err[f"{m}_weight_{key}"][0] += delta.square().sum().item()
                    err[f"{m}_weight_{key}"][1] += weight.square().sum().item()
                    if hess is not None:
                        hk = hess[key]
                        err[f"{m}_output_{key}"][0] += torch.einsum("cnk,ckl,cnl->", delta, hk, delta).item()
                        err[f"{m}_output_{key}"][1] += torch.einsum("cnk,ckl,cnl->", weight, hk, weight).item()
                    if m in parts:
                        parts[m][key].append(w3.pack_codes(codes.reshape(cn * n, k), scales.reshape(cn * n, -1), rows.reshape(-1),
                                                           levels, rows_per_chunk=n))
                    del delta
                del result
            del mats, hess
        for m, d in partial.items():
            for key, n, k in (("w1", I, L), ("w2", L, I)):
                levels = books[key]
                ms = parts[m][key]
                stacked = w3.W3Weight(*(torch.cat([getattr(p, a) for p in ms]) for a in ("planes", "scales", "row_scale")),
                                      E * n, k, levels, w3.lut_of(levels))
                entry = f"{name}.experts.{key}"
                weights[m][entry] = dict(save_weight(d / f"{entry}.safetensors", stacked), experts=E)
            progress[m].write_text(json.dumps({"weights": weights[m]}))
        st.update({k: round((v[0] / v[1]) ** 0.5, 5) if v[1] else None for k, v in err.items()})
        st["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        st["seconds"] = round(time.perf_counter() - block_started, 1)
        torch.cuda.reset_peak_memory_stats()
        stats[name] = st
        stats_path.write_text(json.dumps(stats, indent=1))
        del nv, x, members, parts
        torch.cuda.empty_cache()
        print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in st.items() if v is not None) + f" ({time.perf_counter() - started:.0f}s)", flush=True)

    if any(f"{n}.experts.w1" not in weights[m] for m in outputs for n in names):
        print("partial run: no manifest written", flush=True)
        return
    keys = sorted({k for s in stats.values() for k in s if k.startswith(("cheap_", "gptq_")) and s[k] is not None})
    mean = {k: round(sum(s[k] for s in stats.values() if s.get(k) is not None) / max(1, sum(s.get(k) is not None for s in stats.values())), 5)
            for k in keys}
    capture_meta = json.loads((args.activations / "capture.json").read_text()) if args.activations else None
    for m, d in partial.items():
        manifest = dict(source.manifest) if source else {"format": FORMAT_VERSION, "target": NEMOTRON.name,
                                                         "target_revision": NEMOTRON.model_revision}
        manifest["weights"] = weights[m]
        manifest["experts"] = {
            "source": "checkpoint NVFP4 experts, dequantized", "codebooks": {book_names[b]: list(b) for b in candidates},
            "codebook_choice": f"per matrix, the least cheap-fit error over every {args.choice_every}th chunk of {args.chunk} experts (stats: codebook_w1 / codebook_w2)", "group_size": w3.GROUP,
            "scale_grid": list(ratios), "fc1": "up_proj, checkpoint row order", "blocks": names,
            "trunk_from": str(args.source.resolve()) if source else None, "trunk_manifest_sha256": source.identity if source else None,
            **({"method": "nearest level, group scales on a weight-error grid"} if m == "cheap" else
               {"method": "gptq on routed activations (routing-weighted covariances), batched", "damping": args.damping,
                "min_tokens": args.min_tokens, "prior_tokens": args.prior_tokens, "max_rows": args.max_rows, "capture": capture_meta}),
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
    i = sub.add_parser("inputs", help="freeze the calibration samples in Nemotron's tokenizer")
    i.add_argument("--tokenizer", type=Path, required=True)
    i.add_argument("--code", type=Path, required=True, help="w3_calibration_inputs output with Nemotron's tokenizer")
    i.add_argument("--traces", type=Path, required=True, help="{'items': [{source, id, prompt, reasoning, answer}]} (ShareGPT)")
    i.add_argument("--knowledge", type=Path, required=True, help="{'items': [{source, prompt, reasoning, answer}]} (MMLU validation)")
    i.add_argument("--output", type=Path, required=True)
    i.add_argument("--traces-en", type=int, default=150)
    i.add_argument("--traces-other", type=int, default=100)
    i.add_argument("--knowledge-items", type=int, default=600)
    i.add_argument("--max-tokens", type=int, default=2048)
    i.add_argument("--min-tokens", type=int, default=256)
    i.add_argument("--seed", type=int, default=20261024)
    c = sub.add_parser("capture", help="store every MoE block's routed latent rows over the calibration samples")
    c.add_argument("--inputs", type=Path, required=True)
    c.add_argument("--activations", type=Path, required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--resume", action="store_true", help="continue an interrupted capture after its whole samples")
    f = sub.add_parser("fit", help="write the checkpoint")
    f.add_argument("--method", choices=("cheap", "gptq"), default="gptq")
    f.add_argument("--source", type=Path, help="a W3 checkpoint whose (trunk) entries are copied in")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--cheap-output", type=Path, help="gptq: also write the cheap-route checkpoint from the same pass")
    f.add_argument("--activations", type=Path, help="capture directory (gptq; with cheap, only for the output-error report)")
    f.add_argument("--codebooks", default="lloyd,zero8", help="candidate codebooks (w3.CODEBOOKS), one chosen per matrix")
    f.add_argument("--blocks", default="", help="comma-separated subset of block names (layers.N, mtp; a partial checkpoint does not serve)")
    f.add_argument("--chunk", type=int, default=16, help="experts fitted together (and covariances held at once)")
    f.add_argument("--choice-every", type=int, default=4, help="the codebook choice looks at every n-th chunk of experts")
    f.add_argument("--damping", type=float, default=0.01)
    f.add_argument("--min-tokens", type=int, default=16, help="fewer routed rows keep the cheap fit")
    f.add_argument("--prior-tokens", type=float, default=256, help="covariance shrinkage toward the block's pool: weight n0 / (n + n0)")
    f.add_argument("--max-rows", type=int, default=0, help="use only the first rows of each block's capture (0: all)")
    f.add_argument("--deadline", type=float, default=0, help="seconds: start no block after this (resume later)")
    args = ap.parse_args()
    if args.command == "inputs":
        inputs(args)
    elif args.command == "capture":
        capture(args)
    else:
        if args.method == "gptq" and not args.activations:
            raise SystemExit("gptq needs --activations (run capture first)")
        fit(args)


if __name__ == "__main__":
    main()
