"""Flash-Next trunk quantization error, measured against the BF16 trunk on the serving path.

SGLang disagrees with itself on ~14% of positions, which buries a trunk-precision
effect; the BF16 executor matches the reference, so it is the baseline here. `base`
runs the executor with the BF16 trunk and stores each position's top-k logprobs;
`score` runs a quantized trunk (SHARAR_TRUNK_KEEP_BF16 keeps chosen modules in
BF16, see sharar/model/flashnext/linear.py) and reports, per region (dense: rows inside the
indexer budget; sparse: past it), argmax agreement with BF16, confident agreement
(BF16 top-1 above 0.8), the truncated KL(BF16 || quantized) over BF16's top-k, and
the negative log-likelihood of the text itself.

    python3 -m profiling.flashnext_quant base --out base.pt
    python3 -m profiling.flashnext_quant score --base base.pt --label fp8 --out fp8.json
    python3 -m profiling.flashnext_quant score --base base.pt --precision bf16 --chunk 1024 --label noise --out noise.json

The last line is the control: BF16 with another prefill chunk differs from the
baseline only by rounding, which is the floor any quantized trunk is judged
against (on the frozen profile prompts (`profiling/inputs.py`), argmax 86%/91% and KL 0.36/0.27 dense/sparse;
the FP8 trunk 83%/87% and 0.49/0.36, #5).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch


TOPK = 32
LIMIT = 2051  # rows at or past this index select sparsely (budget + ratio - 1)


def snapshot() -> Path:
    """The pinned Flash-Next snapshot through the target (SHARAR_FLASHNEXT_CACHE), as the server finds it."""
    from sharar.target import FLASHNEXT
    return FLASHNEXT.snapshot_dir()


def prompts(n_prompts: int, length: int) -> list[dict]:
    from profiling.inputs import tasks
    from sharar.tokenizer import SnapshotTokenizer
    rows = tasks(SnapshotTokenizer(snapshot()).encode)[:n_prompts]
    return [{"id": f"{t['id']}[:{length}]", "ids": t["input_ids"][:length]} for t in rows]


def executor_model(precision, chunk: int = 2048):
    """The serving path: a real Executor with pools prefills in `chunk`-token chunks (the sparse
    indexer needs pages); every chunk's hidden rows give that chunk's logits."""
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.model.flashnext.model import FlashNext
    engine = FlashNext(snapshot(), precision=precision)
    ex = Executor(engine, num_pages=40000, num_slots=8, max_context=32768, log=print)

    class _Wrap:
        ablate = frozenset()

        def chunks(self, ids):
            """Each prefill chunk's logits in turn ([rows, V] fp32); full-vocab logits of long prompts are
            gigabytes, so callers that only need statistics reduce one chunk at a time."""
            req = ex.admit(list(ids), SamplingParams(max_new_tokens=1))
            try:
                while req.prefill_pos < req.prefill_len:
                    batch = ex._prepare_single_prefill(req, chunk)
                    out = engine.forward(batch.ids, batch.context)
                    yield engine.logits(out["norm"]).float()
                    ex._finish_single_prefill(batch, out)
            finally:
                ex.release(req)

        def forward(self, ids):
            return {"logits": torch.cat(list(self.chunks(ids)))}

    return _Wrap()


def reduce(model, ids, base_ix=None):
    """Per position: BF16 mode (base_ix None) the top-k logprobs; score mode the logprobs at the
    baseline's top-k ids and the argmax. Both: the NLL of the next token of the text."""
    nxt = torch.tensor(ids[1:] + [0], device="cuda")
    out, row = {"nll": [], "lp": [], "ix": [], "arg": []}, 0
    for logits in model.chunks(ids):
        logp = torch.log_softmax(logits, dim=-1)  # row t predicts ids[t + 1]
        del logits
        n = logp.shape[0]
        out["nll"].append(-logp.gather(1, nxt[row:row + n, None])[:, 0])
        if base_ix is None:
            top = logp.topk(TOPK, dim=-1)
            out["lp"].append(top.values.half()); out["ix"].append(top.indices.int())
        else:
            out["lp"].append(logp.gather(1, base_ix[row:row + n].long())); out["arg"].append(logp.argmax(-1))
        del logp
        row += n
    return {k: torch.cat(v) for k, v in out.items() if v}


def base(args):
    model = executor_model("bf16")
    out = []
    for p in prompts(args.prompts, args.length):
        t0 = time.perf_counter()
        r = reduce(model, p["ids"])
        nll = r["nll"][:-1]
        out.append({"id": p["id"], "ids": p["ids"], "top_lp": r["lp"].cpu(), "top_ix": r["ix"].cpu(), "nll": nll.cpu()})
        print(p["id"], f"{time.perf_counter() - t0:.1f}s", f"nll {nll.mean():.4f}", flush=True)
    torch.save(out, args.out)


def score(args):
    if args.watchdog:  # every thread's stack on stderr while a prompt runs longer than this
        import faulthandler
        faulthandler.dump_traceback_later(args.watchdog, repeat=True)
    rows = [r for r in torch.load(args.base) if not args.only or r["id"].startswith(args.only)]
    # Onto the device before the model loads: copies of the loaded tensors made after it (with
    # memory tight) stalled with the GPU idle, in 4 of 7 runs.
    for r in rows:
        r["top_lp"], r["top_ix"], r["nll"] = r["top_lp"].cuda().float(), r["top_ix"].cuda(), r["nll"].cuda()
    torch.cuda.synchronize()
    model = executor_model(args.precision, args.chunk)
    knobs = {k: v for k, v in os.environ.items() if k.startswith(("SHARAR_TRUNK", "SHARAR_W3"))}
    per_prompt, agg = [], {}
    for row in rows:
        b_lp, b_ix = row["top_lp"], row["top_ix"]
        r = reduce(model, row["ids"], b_ix)
        q_lp = r["lp"]
        dev = q_lp.device
        stats = {
            "agree": (r["arg"] == b_ix[:, 0]).float(),
            "confident": b_lp[:, 0] > math.log(0.8),
            "kl": (b_lp.exp() * (b_lp - q_lp)).sum(-1),
            "nll_b": torch.cat([row["nll"].to(dev), torch.zeros(1, device=dev)]),
            "nll_q": r["nll"],
        }
        n = q_lp.shape[0]
        rep = {"id": row["id"]}
        for region, sl in (("dense", slice(0, min(n, LIMIT))), ("sparse", slice(LIMIT, n - 1))):
            s = {k: v[sl] for k, v in stats.items()}
            if not s["agree"].numel():
                continue
            conf = s["confident"]
            r = {"n": int(s["agree"].numel()), "argmax": s["agree"].mean().item(),
                 "confident": s["agree"][conf].mean().item() if conf.any() else float("nan"), "confident_n": int(conf.sum()),
                 "kl": s["kl"].mean().item(), "nll_bf16": s["nll_b"].mean().item(), "nll": s["nll_q"].mean().item()}
            rep[region] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            a = agg.setdefault(region, {})
            for k in ("argmax", "kl", "nll_bf16", "nll"):
                a.setdefault(k, []).append((r[k], r["n"]))
            a.setdefault("confident", []).append((r["confident"], r["confident_n"]))
        print(json.dumps(rep), flush=True)
        per_prompt.append(rep)
    summary = {region: {k: round(sum(v * w for v, w in xs) / sum(w for _, w in xs), 4) for k, xs in a.items()} for region, a in agg.items()}
    result = {"label": args.label, "precision": args.precision, "chunk": args.chunk, "knobs": knobs, "summary": summary, "prompts": per_prompt}
    print(json.dumps({"label": args.label, "knobs": knobs, "summary": summary}), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base"); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--base", required=True); s.add_argument("--out", required=True)
    s.add_argument("--label", required=True); s.add_argument("--precision", default="fp8")
    s.add_argument("--chunk", type=int, default=2048, help="prefill chunk; with --precision bf16 and another chunk, the numerical-noise control")
    s.add_argument("--only", default=None, help="score only the prompts whose id starts with this")
    s.add_argument("--watchdog", type=float, default=0, help="dump stacks every N seconds (hang diagnosis)")
    for p in (b, s):
        p.add_argument("--prompts", type=int, default=8); p.add_argument("--length", type=int, default=8000)
    args = ap.parse_args()
    (base if args.cmd == "base" else score)(args)


if __name__ == "__main__":
    main()
