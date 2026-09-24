"""Parity of the Flash-Next reference forward against an SGLang server, position by position.

    collect: python bench/flashnext_parity.py collect --url http://127.0.0.1:30003 --out parity-sglang.json
    compare: python bench/flashnext_parity.py compare --sglang parity-sglang.json --out parity-report.json [--layers N]

collect sends each prompt's token ids through /generate with return_logprob and stores, per
position, the target token's logprob and the top-5 logprobs. compare runs the reference on the
same ids and reports next-token argmax agreement (split into the dense region, where every
block is selected, and the sparse region past the indexer budget), the logprob difference of the actual token,
and top-1/top-5 agreement at the final position. Prompts: the first tokens of three frozen
family prompts plus two short texts, all inside the dense-attention limit of the reference.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

INPUTS = Path("/home/yke/sharar-results/reviews/2026-09-15-profile/inputs/task-inputs.json")
SNAP_ROOT = Path(__import__("os").environ.get("FN_SNAP_ROOT", "/home/yke/data/huggingface-flash-next/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots"))
INPUTS = Path(__import__("os").environ.get("FN_INPUTS", str(INPUTS)))
TEXTS = ["The quick brown fox jumps over the lazy dog. Then it ran into the forest, where",
         "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return"]


def snapshot() -> Path:
    return sorted(SNAP_ROOT.iterdir())[-1]


def prompts(lengths=(64, 200, 512), long=False) -> list[dict]:
    frozen = json.loads(INPUTS.read_text())["tasks"]
    if long:  # past the indexer budget: the sparse selection decides what attention sees
        return [{"id": f"{task['id']}[:{n}]", "ids": task["input_ids"][:n]} for task, n in zip(frozen[:3], (3000, 6000, 10000))]
    from sharar.tokenizer import SnapshotTokenizer
    tok = SnapshotTokenizer(snapshot())
    out = [{"id": f"text{i}", "ids": tok.encode(t)} for i, t in enumerate(TEXTS)]
    for task, n in zip(frozen[:3], lengths):
        out.append({"id": f"{task['id']}[:{n}]", "ids": task["input_ids"][:n]})
    return out


def collect(args):
    rows = []
    for p in prompts(long=args.long):
        body = {"input_ids": p["ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 5}
        req = urllib.request.Request(f"{args.url}/generate", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        meta = d["meta_info"]
        rows.append({"id": p["id"], "ids": p["ids"],
                     "input_token_logprobs": meta["input_token_logprobs"],  # [(logprob, token_id, text)] per position (first is None)
                     "input_top_logprobs": meta["input_top_logprobs"],
                     "output_token_logprobs": meta["output_token_logprobs"], "output_top_logprobs": meta["output_top_logprobs"]})
        print(p["id"], len(p["ids"]), "tokens; next:", meta["output_token_logprobs"][0][1], flush=True)
    Path(args.out).write_text(json.dumps(rows))


def executor_model(precision, chunk: int = 2048):
    """The serving path: a real Executor with pools prefills in 2,048-token chunks (the sparse
    indexer needs pages); every chunk's hidden rows give that chunk's logits."""
    import torch
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


def compare(args):
    import math
    import torch
    rows = json.loads(Path(args.sglang).read_text())
    if args.model == "reference":
        from sharar.model.flashnext.reference import FlashNextReference
        ref = FlashNextReference(snapshot(), layers=args.layers)
    elif args.model == "executor":
        ref = executor_model(args.precision)
    else:
        from sharar.model.flashnext.model import FlashNext
        engine = FlashNext(snapshot(), precision=args.precision)

        class _Wrap:
            ablate = frozenset()

            def forward(self, ids):
                out = engine.prefill(torch.tensor(ids, dtype=torch.int64, device=engine.device))
                return {"logits": engine.logits(out["norm"]).float()}

        ref = _Wrap()
    variants = [[] if v in ("", "none") else v.split("+") for v in (args.ablate or ["none"])]
    confident = math.log(args.confident)
    report = []
    for row in rows:
        ids = row["ids"]
        n = len(ids)
        for variant in variants:
            if variant and args.model != "reference":
                raise SystemExit("ablations apply to the reference only")
            ref.ablate = frozenset(variant)
            t0 = time.perf_counter()
            out = ref.forward(ids)
            logp = torch.log_softmax(out["logits"], dim=-1)  # [T, V]; row t predicts ids[t+1]
            secs = time.perf_counter() - t0
            # SGLang: input_token_logprobs[t] is the logprob of ids[t] given ids[:t] (t >= 1), predicted by row t-1.
            positions = []
            for t in range(1, n):
                lp_s, tok, _ = row["input_token_logprobs"][t]
                if lp_s is None or not row["input_top_logprobs"][t]:
                    continue
                s_top = row["input_top_logprobs"][t]
                r_lp, r_arg = logp[t - 1].max(dim=-1)
                positions.append({"t": t, "tok": tok, "lp_s": round(lp_s, 4), "lp_r": round(logp[t - 1, tok].item(), 4),
                                  "s_top1": s_top[0][1], "s_top1_lp": round(s_top[0][0], 4), "r_top1": int(r_arg), "r_top1_lp": round(r_lp.item(), 4),
                                  "s_top5": [x[1] for x in s_top]})
            diffs = [p["lp_r"] - p["lp_s"] for p in positions]
            agree = [p["r_top1"] == p["s_top1"] for p in positions]
            conf = [a for p, a in zip(positions, agree) if p["s_top1_lp"] > confident]
            buckets = {}
            for p, a in zip(positions, agree):
                b = f"{(p['t'] // 64) * 64:04d}"
                buckets.setdefault(b, []).append(a)
            worst = sorted(positions, key=lambda p: -abs(p["lp_r"] - p["lp_s"]))[:6]
            limit = getattr(getattr(ref, "cfg", None), "max_dense_prompt", 2051)
            regions = {}
            for name, keep in (("dense", lambda t: t - 1 < limit), ("sparse", lambda t: t - 1 >= limit)):  # row t-1 predicts position t
                sel = [(p, a) for p, a in zip(positions, agree) if keep(p["t"])]
                sel_conf = [a for p, a in sel if p["s_top1_lp"] > confident]
                if sel:
                    regions[name] = {"n": len(sel), "argmax": round(sum(a for _, a in sel) / len(sel), 4),
                                     "confident": round(sum(sel_conf) / max(1, len(sel_conf)), 4), "confident_n": len(sel_conf),
                                     "logprob_diff_mean": round(sum(p["lp_r"] - p["lp_s"] for p, _ in sel) / len(sel), 4)}
            nxt_s = row["output_token_logprobs"][0][1]
            nxt_r = int(logp[n - 1].argmax())
            s_top5 = [x[1] for x in row["output_top_logprobs"][0]]
            r_top5 = logp[n - 1].topk(5).indices.tolist()
            rep = {"id": row["id"], "ablate": variant, "tokens": n, "seconds": round(secs, 1),
                   "argmax_agreement": round(sum(agree) / max(1, len(agree)), 4),
                   "top5_agreement": round(sum(p["r_top1"] in p["s_top5"] for p in positions) / max(1, len(positions)), 4),
                   "confident_agreement": round(sum(conf) / max(1, len(conf)), 4), "confident_n": len(conf), "regions": regions,
                   "agreement_by_64": {b: round(sum(v) / len(v), 3) for b, v in sorted(buckets.items())},
                   "logprob_diff_mean": round(sum(diffs) / max(1, len(diffs)), 4), "logprob_diff_max_abs": round(max(abs(d) for d in diffs) if diffs else 0, 4),
                   "next_token_match": nxt_s == nxt_r, "next_top5_sglang": s_top5, "next_top5_reference": r_top5,
                   "worst": [{k: p[k] for k in ("t", "tok", "lp_s", "lp_r", "s_top1", "r_top1", "s_top1_lp", "r_top1_lp")} for p in worst],
                   "positions": positions}
            print(json.dumps({k: v for k, v in rep.items() if k != "positions"}), flush=True)
            report.append(rep)
    Path(args.out).write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--url", default="http://127.0.0.1:30003"); c.add_argument("--out", required=True)
    c.add_argument("--long", action="store_true", help="3K/6K/10K-token prompts, past the sparse-attention budget")
    m = sub.add_parser("compare"); m.add_argument("--sglang", required=True); m.add_argument("--out", required=True); m.add_argument("--layers", type=int, default=None)
    m.add_argument("--model", choices=["reference", "engine", "executor"], default="reference")
    m.add_argument("--precision", default=None, help="engine trunk precision: bf16 or fp8 (default SHARAR_FLASHNEXT_TRUNK or fp8)")
    m.add_argument("--ablate", action="append", help="variant per flag: 'none' or '+'-joined names from {ple,fq,shared,hc,dense}; repeat for several (dense: attention ignores the indexer, the control for long prompts)")
    m.add_argument("--confident", type=float, default=0.8, help="SGLang top-1 probability above which a position counts as confident")
    args = ap.parse_args()
    (collect if args.cmd == "collect" else compare)(args)


if __name__ == "__main__":
    main()
