"""In-process layers of the lossy-change gate (`tools/lossy-gate.py`, #31) for one arm, one model load.

tools/lossy-gate.py runs this from the Python environment with the arm's settings in the environment (the
serving profile resolved as `sharar serve` resolves it), on the same checkout's engine as the served arm, so the
arm measures that code in the served configuration: the
executor is built as `sharar serve` builds it (proposer, depth, verify state, draft precision,
draft), and greedy requests carry the profile's acceptance rule.

    python3 -m gate.lossy_gate --prompts tools/gate/workloads/lossy_v1.json --out OUT \\
        --proposer mtp --spec 4 --draft-precision bf16 --accept 1:0 [--base BASE] [--floor-chunk 32]

Layer 2, speculation health (spec.json): every prompt generated greedily (chat template, thinking
off, cold prefix cache) `--passes` times; per prompt the emitted tokens, verify rounds, tokens per
round, decode seconds, round cost and relaxed acceptances.

Layer 1, distribution distance: a baseline arm (no --base) freezes each prompt plus its first-pass
continuation (sequences.json) and stores the top-32 log-probabilities of every continuation
position, computed `--chunk` rows per forward (base.pt); `--floor-chunk` scores the baseline
against itself at another chunk, the rounding noise floor (floor.json). A candidate arm (--base)
scores the baseline's sequences at `--chunk` (dist.json): per position the argmax agreement with
the baseline, the truncated KL(baseline || arm) over the baseline's top 32, and the arm's NLL of
the sequence's token. At 64 rows every forward takes the decode path (3-bit copies serve calls
of at most 64 rows), which is where the lossy weight options act.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

TOPK = 32
CONFIDENT = math.log(0.8)


def chat_ids(tok, prompt: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def source_hash() -> str:
    """SHA-256 over the engine package's Python files (the code this process imported)."""
    import sharar
    root = Path(sharar.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load(args):
    from sharar import target as targets
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine import config
    from sharar.engine.executor import Executor
    from sharar.tokenizer import SnapshotTokenizer

    cfg = config.EngineConfig.from_env().replace(spec=str(args.spec))
    config.install(cfg)
    served = targets.get(cfg.target)
    snap = served.snapshot_dir()
    started = time.perf_counter()
    model = targets.load_model(served, snap, log=print)
    spec = SpecConfig(args.proposer, args.spec, args.verify_state, args.draft_precision, args.draft)
    ex = Executor(model, log=print, spec_config=spec, max_context=args.max_context)
    ex.capture_graphs(log=print)
    ex.warmup(log=print)
    info = {"target": served.name, "model": f"{served.model_id}@{served.model_revision}", "config": cfg.describe(),
            "w3_identity": getattr(model, "w3_identity", None), "engine_source_sha256": source_hash(),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(), "load_s": round(time.perf_counter() - started, 1),
            "spec": {"proposer": args.proposer, "depth": args.spec, "verify_state": args.verify_state,
                     "draft_precision": args.draft_precision, "draft": args.draft}}
    return model, ex, SnapshotTokenizer(snap), info


# ---------------------------------------------------------------- layer 2

def spec_pass(ex, prompts: list[dict], accept: tuple[int, float], tokens: int) -> list[dict]:
    from sharar.engine.sampler import SamplingParams

    rows = []
    for p in prompts:
        ex.flush_cache()
        ex.spec.reset_stats()
        r = ex.generate(p["ids"], SamplingParams(max_new_tokens=tokens, accept_top_k=accept[0], accept_margin=accept[1], trace=False))
        n, rounds = len(r.output_ids), r.spec_rounds
        row = {"id": p["id"], "domain": p["domain"], "prompt_tokens": len(p["ids"]), "output_tokens": n, "rounds": rounds,
               "relaxed": r.spec_relaxed, "prefill_s": round(r.prefill_seconds, 4), "decode_s": round(r.decode_seconds, 4),
               "tok_per_round": (n - 1) / rounds if rounds else None,
               "round_ms": 1000 * r.decode_seconds / rounds if rounds else None,
               "decode_tps": (n - 1) / r.decode_seconds if n > 1 and r.decode_seconds > 0 else None,
               "output_ids": list(r.output_ids)}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "output_ids"}), flush=True)
    return rows


# ---------------------------------------------------------------- layer 1

def logprob_chunks(ex, model, ids: list[int], chunk: int):
    """(first row, log-probabilities [rows, V] fp32) of each prefill chunk of a cold prompt; row t predicts ids[t + 1]."""
    from sharar.engine.sampler import SamplingParams

    ex.flush_cache()
    req = ex.admit(list(ids), SamplingParams(max_new_tokens=1, trace=False))
    try:
        if req.prefill_pos != 0:
            raise RuntimeError(f"expected a cold prefill, {req.prefill_pos} tokens were cached")
        while req.prefill_pos < req.prefill_len:
            row = req.prefill_pos
            batch = ex._prepare_single_prefill(req, chunk)
            out = model.forward(batch.ids, batch.context)
            logp = torch.log_softmax(model.logits(out["norm"]).float(), dim=-1)
            yield row, logp
            del logp
            ex._finish_single_prefill(batch, out)
    finally:
        req.cacheable = False
        ex.release(req)


def base_rows(ex, model, seq: dict, chunk: int) -> dict:
    """Top-k log-probabilities of every continuation position (rows start-1 .. len-2)."""
    ids, lo = seq["ids"], seq["start"] - 1
    lps, ixs = [], []
    for row, logp in logprob_chunks(ex, model, ids, chunk):
        a, b = max(lo, row), min(row + logp.shape[0], len(ids) - 1)
        if a < b:
            top = logp[a - row:b - row].topk(TOPK, dim=-1)
            lps.append(top.values)
            ixs.append(top.indices)
    return {"id": seq["id"], "lp": torch.cat(lps).half().cpu(), "ix": torch.cat(ixs).int().cpu()}


def score_rows(ex, model, seq: dict, base: dict, chunk: int) -> dict:
    """Per continuation position: argmax agreement, truncated KL(base || arm), baseline-confident flag, NLL of the token."""
    ids, lo = seq["ids"], seq["start"] - 1
    b_lp, b_ix = base["lp"].cuda().float(), base["ix"].cuda().long()
    nxt = torch.tensor(ids[1:], device="cuda")
    agree, kl, nll = [], [], []
    for row, logp in logprob_chunks(ex, model, ids, chunk):
        a, b = max(lo, row), min(row + logp.shape[0], len(ids) - 1)
        if a >= b:
            continue
        part, i, j = logp[a - row:b - row], a - lo, b - lo
        q = part.gather(1, b_ix[i:j])
        agree.append(part.argmax(-1) == b_ix[i:j, 0])
        kl.append((b_lp[i:j].exp() * (b_lp[i:j] - q)).sum(-1))
        nll.append(-part.gather(1, nxt[a:b, None])[:, 0])
    agree, kl, nll = torch.cat(agree), torch.cat(kl), torch.cat(nll)
    confident = b_lp[:, 0] > CONFIDENT
    return {"id": seq["id"], "domain": seq["domain"], "n": int(agree.numel()),
            "agree": agree.int().tolist(), "confident": confident.int().tolist(),
            "kl": [round(x, 6) for x in kl.tolist()], "nll": [round(x, 5) for x in nll.tolist()]}


def summary(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    agree = sum(sum(r["agree"]) for r in rows)
    conf = [(a, c) for r in rows for a, c in zip(r["agree"], r["confident"]) if c]
    return {"n": n, "argmax": agree / n, "confident_argmax": sum(a for a, _ in conf) / max(1, len(conf)),
            "kl": sum(sum(r["kl"]) for r in rows) / n, "nll": sum(sum(r["nll"]) for r in rows) / n}


def score_all(ex, model, seqs, base, chunk, label, out: Path) -> None:
    started = time.perf_counter()
    rows = [score_rows(ex, model, s, b, chunk) for s, b in zip(seqs, base)]
    s = summary(rows)
    print(json.dumps({"layer1": label, "chunk": chunk, **{k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()},
                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    out.write_text(json.dumps({"label": label, "chunk": chunk, "summary": s, "sequences": rows}) + "\n")


# ---------------------------------------------------------------- driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="this arm's directory (steps already written are skipped)")
    ap.add_argument("--base", type=Path, help="the baseline arm's directory (candidate arm); absent: this arm is the baseline")
    ap.add_argument("--accept", default="1:0", help="K:M acceptance rule of greedy requests (the profile's; 1:0 exact)")
    ap.add_argument("--proposer", required=True)
    ap.add_argument("--spec", type=int, required=True)
    ap.add_argument("--verify-state", default="reconstruct")
    ap.add_argument("--draft-precision", default="bf16")
    ap.add_argument("--draft", default=None)
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--tokens", type=int, default=0, help="output tokens per prompt (default: the workload's)")
    ap.add_argument("--passes", type=int, default=1, help="layer-2 passes (the baseline runs two: the second is the timing floor)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--floor-chunk", type=int, default=0, help="baseline arm: also score the baseline at this chunk (the noise floor)")
    ap.add_argument("--layers", default="1,2")
    args = ap.parse_args(argv)
    layers = set(args.layers.split(","))
    k, m = args.accept.split(":")
    accept = (int(k), float(m))
    workload = json.loads(args.prompts.read_text())
    tokens = args.tokens or workload["output_tokens"]
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = args.base is None
    model, ex, tok, info = load(args)
    info.update(prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(), workload=workload["version"],
                accept=list(accept), tokens=tokens, chunk=args.chunk, floor_chunk=args.floor_chunk, baseline=baseline)
    (args.out / "arm.json").write_text(json.dumps(info, indent=1) + "\n")
    prompts = [{**p, "ids": chat_ids(tok, p["prompt"])} for p in workload["prompts"]]

    spec_file = args.out / "spec.json"
    if ("2" in layers or baseline) and not spec_file.exists():
        from sharar.engine.sampler import SamplingParams
        ex.generate(prompts[0]["ids"], SamplingParams(max_new_tokens=64, trace=False))  # first-request effects
        passes = []
        for i in range(args.passes):
            started = time.perf_counter()
            passes.append(spec_pass(ex, prompts, accept, tokens))
            print(json.dumps({"layer2_pass": i, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        spec_file.write_text(json.dumps({"accept": list(accept), "tokens": tokens, "passes": passes}) + "\n")

    if "1" in layers:
        if baseline:
            seq_file, base_file = args.out / "sequences.json", args.out / "base.pt"
            if not seq_file.exists():
                first = json.loads(spec_file.read_text())["passes"][0]
                seqs = [{"id": p["id"], "domain": p["domain"], "start": len(p["ids"]), "ids": p["ids"] + r["output_ids"]}
                        for p, r in zip(prompts, first)]
                seq_file.write_text(json.dumps(seqs) + "\n")
            seqs = json.loads(seq_file.read_text())
            if not base_file.exists():
                started = time.perf_counter()
                torch.save([base_rows(ex, model, s, args.chunk) for s in seqs], base_file)
                print(json.dumps({"layer1": "base", "chunk": args.chunk, "seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if args.floor_chunk and not (args.out / "floor.json").exists():
                score_all(ex, model, seqs, torch.load(base_file), args.floor_chunk, "floor", args.out / "floor.json")
        elif not (args.out / "dist.json").exists():
            seqs = json.loads((args.base / "sequences.json").read_text())
            score_all(ex, model, seqs, torch.load(args.base / "base.pt"), args.chunk, "candidate", args.out / "dist.json")
    print("lossy_gate arm done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
