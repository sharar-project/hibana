# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced logits of the Sharar Nemotron engine on held-out text, to compare expert formats (NVFP4 against the
3-bit experts of a W3 checkpoint): KL(reference || candidate), top-1 agreement and the reference token's NLL, per
prompt and over all positions.

    heldout   ShareGPT traces the calibration samples did not use (`--inputs` of nemotron_w3_experts), rendered and cut
    dump      prefill each held-out prompt (one stand-alone prefill, as `Nemotron.prefill`) and store the FP32 log-softmax
              of every position in BF16; the engine's config from the environment (SHARAR_W3_DECODE, SHARAR_W3_CHECKPOINT,
              SHARAR_NEMOTRON_EXPERTS)
    compare   two dumps

    python3 -m profiling.nemotron_w3_logits heldout --traces answers-sharegpt.json --inputs nemotron-expert-inputs.json --out heldout.json
    SHARAR_NEMOTRON_EXPERTS=nvfp4 python3 -m profiling.nemotron_w3_logits dump --heldout heldout.json --out nvfp4.pt
    SHARAR_W3_DECODE=1 SHARAR_W3_CHECKPOINT=DIR python3 -m profiling.nemotron_w3_logits dump --heldout heldout.json --out w3.pt
    python3 -m profiling.nemotron_w3_logits compare nvfp4.pt w3.pt
"""
import argparse
import json
import random
from pathlib import Path

import torch


def heldout(args) -> None:
    from tokenizers import Tokenizer
    from sharar.target import NEMOTRON
    from sharar.weights.nemotron_w3_experts import render
    tok = Tokenizer.from_file(str(NEMOTRON.snapshot_dir() / "tokenizer.json") if not args.tokenizer else str(args.tokenizer))
    used = {s["source"] for s in json.loads(args.inputs.read_text())["samples"]}
    items = [it for it in json.loads(args.traces.read_text())["items"]
             if f"{it['source']}/{it.get('id')}" not in used and (it.get("reasoning") or it.get("answer"))]
    random.Random(args.seed).shuffle(items)
    out = []
    for lang, want in (("en", args.each), ("other", args.each)):
        for it in [i for i in items if i["source"].endswith("/" + lang)]:
            if sum(f"/{lang}/" in o["source"] for o in out) >= want:
                break
            ids = tok.encode(render(it), add_special_tokens=False).ids
            if len(ids) >= args.min_tokens:
                out.append({"source": f"{it['source']}/{it.get('id')}", "input_ids": ids[:args.max_tokens]})
    args.out.write_text(json.dumps({"samples": out}))
    print(f"{len(out)} held-out prompts, {sum(len(s['input_ids']) for s in out)} tokens -> {args.out}")


@torch.inference_mode()
def dump(args) -> None:
    from sharar import target as targets
    from sharar.engine.config import current, install
    install(current().replace(target="nemotron"))
    t = targets.get("nemotron")
    model = targets.load_model(t, t.snapshot_dir(), log=print)
    samples = json.loads(args.heldout.read_text())["samples"]
    res = {"identity": getattr(model, "w3_identity", None), "experts": "w3" if getattr(model, "w3_identity", None) else "nvfp4",
           "sources": [], "logprobs": [], "ids": []}
    for s in samples:
        ids = torch.tensor(s["input_ids"], dtype=torch.long, device="cuda")
        out = model.prefill(ids)
        lp = torch.log_softmax(model.logits(out["norm"]).float(), -1)
        res["sources"].append(s["source"])
        res["ids"].append(ids.cpu())
        res["logprobs"].append(lp.to(torch.bfloat16).cpu())
        del out, lp
    torch.save(res, args.out)
    print(f"{len(samples)} prompts -> {args.out} ({res['experts']}, {res['identity']})")


def compare(args) -> None:
    a, b = torch.load(args.reference), torch.load(args.candidate)
    assert a["sources"] == b["sources"]
    tot = {"kl": 0.0, "top1": 0.0, "nll_ref": 0.0, "nll_cand": 0.0, "n": 0}
    rows = []
    for src, ids, la, lb in zip(a["sources"], a["ids"], a["logprobs"], b["logprobs"]):
        la, lb = la.float().cuda(), lb.float().cuda()
        kl = (la.exp() * (la - lb)).sum(-1)
        top1 = (la.argmax(-1) == lb.argmax(-1)).float()
        nxt = ids[1:].cuda()
        nll_a, nll_b = -la[:-1].gather(1, nxt[:, None])[:, 0], -lb[:-1].gather(1, nxt[:, None])[:, 0]
        rows.append({"source": src, "tokens": len(ids), "kl_mean": round(float(kl.mean()), 5), "top1": round(float(top1.mean()), 4),
                     "nll_ref": round(float(nll_a.mean()), 4), "nll_cand": round(float(nll_b.mean()), 4)})
        tot["kl"] += float(kl.sum()); tot["top1"] += float(top1.sum()); tot["n"] += len(ids)
        tot["nll_ref"] += float(nll_a.sum()); tot["nll_cand"] += float(nll_b.sum())
    n, m = tot["n"], tot["n"] - len(rows)
    summary = {"engine": f"Sharar ({args.machine})", "reference": a["experts"], "candidate": b["experts"], "candidate_identity": b["identity"],
               "positions": n, "kl_mean": round(tot["kl"] / n, 5), "top1": round(tot["top1"] / n, 4),
               "nll_ref": round(tot["nll_ref"] / m, 4), "nll_cand": round(tot["nll_cand"] / m, 4),
               "ppl_ratio": round(float(torch.tensor((tot["nll_cand"] - tot["nll_ref"]) / m).exp()), 4)}
    for r in rows:
        print(json.dumps(r))
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "prompts": rows}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    h = sub.add_parser("heldout")
    h.add_argument("--traces", type=Path, required=True)
    h.add_argument("--inputs", type=Path, required=True, help="the calibration samples (their sources are excluded)")
    h.add_argument("--tokenizer", type=Path)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--each", type=int, default=8, help="prompts per language group (en, other)")
    h.add_argument("--min-tokens", type=int, default=512)
    h.add_argument("--max-tokens", type=int, default=1024)
    h.add_argument("--seed", type=int, default=7)
    d = sub.add_parser("dump")
    d.add_argument("--heldout", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--out", type=Path)
    c.add_argument("--machine", default="spark-9931")
    args = ap.parse_args()
    {"heldout": heldout, "dump": dump, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
