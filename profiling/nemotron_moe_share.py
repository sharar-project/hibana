# SPDX-License-Identifier: Apache-2.0
"""MoE sharing proxy for Nemotron (`model/nemotron/moe.py`, SHARAR_MOE_KEEP / SHARAR_MOE_FLOORS / SHARAR_MOE_TAU):
teacher-forced logits on held-out prompts with each MoE block's routing replaced by the verify forwards' shared
selection, against full routing, in one model load.

A verify forward routes its rows as one call: `--shape b1` groups each prompt's positions into calls of 6 consecutive
rows (one request at depth 5), `b4` into calls of four such windows a quarter of the prompt apart (four unrelated
requests, approximated by distant positions of one document). Each call's ids come from `route_budget` exactly as a
served verify call would (the block's floor, free riders, cap / extra, tau); rows outside whole calls route fully.
The engine's prefill path is otherwise unchanged (the stand-alone `Nemotron.prefill`), so the logits differ from
full routing only through the routing.

Per setting: KL(full || shared) and top-1 agreement over all positions, the next token's NLL against full routing's,
and per MoE block the mean distinct experts per call (with the full-routing count, for the bytes a setting saves).
`sensitivity` runs each block alone at each floor (the others full) plus uniform floors; `tables` runs given
settings (JSON list). Each prompt's full-routing log-probs are computed once and compared against every setting, so
memory holds one prompt's logits at a time.

    python3 -m profiling.nemotron_moe_share sensitivity --heldout heldout.json --prompts 8 --shape b1 --out sens.json
    python3 -m profiling.nemotron_moe_share tables --heldout heldout.json --settings settings.json --shape b4 --out t.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROWS = 6  # verify rows a request at depth 5


def calls(r: int, shape: str) -> list[torch.Tensor]:
    groups = r // ROWS
    if shape == "b1":
        return [torch.arange(g * ROWS, (g + 1) * ROWS) for g in range(groups)]
    q = groups // 4
    return [torch.cat([torch.arange((j + i * q) * ROWS, (j + i * q + 1) * ROWS) for i in range(4)]) for j in range(q)]


def install(shape: str, record: dict):
    """Replace NemotronMoe.route for prefill calls of blocks that carry a `sim` setting."""
    from sharar.model.nemotron import moe as M
    orig = M.NemotronMoe.route

    def route(self, fused, ctx):
        sim = getattr(self, "sim", None)
        if record.get("mass") is not None and self.block is not None and sim is None:
            s = fused[:, :self.experts].sigmoid()
            top = torch.topk(s + self.bias, self.topk, dim=-1).indices
            p = s.gather(1, top)
            m = (torch.cumsum(p, 1) / p.sum(1, keepdim=True)).mean(0)
            record["mass"].setdefault(self.block, []).append(m.cpu())
        if sim is None or ctx is None or ctx.mode != "prefill":
            return orig(self, fused, ctx)
        r = fused.shape[0]
        ids, w, lat = M.route_topk(fused, self.bias, self.topk, self.latent)
        stats = record["distinct"].setdefault(self.block, [])
        for rows in calls(r, shape):
            rows = rows.to(fused.device)
            sub = fused.index_select(0, rows)
            keep, budget, per = sim["keep"], sim.get("budget", 0), sim.get("per_request", 0)
            fb = 0
            if budget:  # NemotronMoe.fallback for this call's shape
                cap = min(budget, max(self.MIN_CAP, per * (len(rows) // ROWS))) if per else budget
                fb = max(min(self.SHARE_MIN, keep), min(keep, cap // len(rows)))
                fb = fb if fb < keep else 0
            i2, w2, _ = M.route_budget(sub, self.bias, self.topk, self.latent, keep, budget, None, ROWS,
                                       per_request=per, min_cap=self.MIN_CAP, repeat=self.w3 is None,
                                       candidates=self.topk, extra=sim.get("extra", 0), tau=sim.get("tau", 0.0),
                                       fallback=fb)
            stats.append((len(torch.unique(i2[w2 > 0])), len(torch.unique(ids.index_select(0, rows)))))
            ids.index_copy_(0, rows, i2)
            w.index_copy_(0, rows, w2)
        return ids, w, lat

    M.NemotronMoe.route = route


def apply(moes: dict, setting: dict | None):
    """setting: {"floors": {block: F}, "tau": t, "extra": X, ...} or None (full routing everywhere)."""
    for b, m in moes.items():
        f = (setting or {}).get("floors", {}).get(b, 22)
        tau = (setting or {}).get("tau", 0.0)
        if f >= 22 and not tau:
            m.sim = None
        else:
            m.sim = {"keep": min(f, 21) if f < 22 else 1, **{k: v for k, v in (setting or {}).items() if k != "floors"}}
            if f >= 22 and tau:
                m.sim["keep"] = 1


@torch.inference_mode()
def run(args, settings: list[dict]):
    from sharar import target as targets
    from sharar.engine.config import current, install as install_config
    from sharar.model.nemotron.moe import NemotronMoe
    install_config(current().replace(target="nemotron"))
    t = targets.get("nemotron")
    model = targets.load_model(t, t.snapshot_dir(), log=None)
    moes = {b.mixer.block: b.mixer for b in model.layers if isinstance(b.mixer, NemotronMoe)}
    record = {"distinct": {}, "mass": {}}
    install(args.shape, record)
    samples = json.loads(Path(args.heldout).read_text())["samples"][: args.prompts]
    acc = [{"kl": 0.0, "top1": 0.0, "nll": 0.0, "n": 0, "distinct": {}} for _ in settings]
    ref_nll, positions = 0.0, 0
    started = time.time()
    for si, smp in enumerate(samples):
        ids = torch.tensor(smp["input_ids"], dtype=torch.long, device="cuda")
        apply(moes, None)
        record["mass"] = {} if si == 0 else None  # the routing-mass curves from the first prompt's full pass
        ref = torch.log_softmax(model.logits(model.prefill(ids)["norm"]).float(), -1)[:-1]
        mass = record["mass"]
        if si == 0:
            record["mass_curves"] = {b: torch.stack(v).mean(0).tolist() for b, v in mass.items()}
        record["mass"] = None
        tgt = ids[1:]
        ref_nll += float(-ref.gather(1, tgt[:, None]).sum())
        ref_top = ref.argmax(-1)
        positions += tgt.numel()
        for k, st in enumerate(settings):
            apply(moes, st)
            record["distinct"] = {}
            lp = torch.log_softmax(model.logits(model.prefill(ids)["norm"]).float(), -1)[:-1]
            a = acc[k]
            a["kl"] += float((ref.exp() * (ref - lp)).sum())
            a["top1"] += float((lp.argmax(-1) == ref_top).sum())
            a["nll"] += float(-lp.gather(1, tgt[:, None]).sum())
            a["n"] += tgt.numel()
            for b, v in record["distinct"].items():
                d = a["distinct"].setdefault(b, [0, 0, 0])
                d[0] += sum(x for x, _ in v)
                d[1] += sum(y for _, y in v)
                d[2] += len(v)
            del lp
        del ref
        print(f"prompt {si + 1}/{len(samples)} ({time.time() - started:.0f}s)", flush=True)
        dump(args, settings, acc, record, si + 1, positions, ref_nll, final=si + 1 == len(samples))


def dump(args, settings, acc, record, prompts, positions, ref_nll, final):
    """The results so far (every prompt: a window's timeout keeps what finished)."""
    out = {"shape": args.shape, "prompts": prompts, "positions": positions, "ref_nll": ref_nll / positions,
           "mass_curves": record.get("mass_curves"), "results": []}
    for st, a in zip(settings, acc):
        dist = {b: {"distinct": d[0] / d[2], "full": d[1] / d[2]} for b, d in a["distinct"].items()}
        saved = sum(v["full"] - v["distinct"] for v in dist.values())
        out["results"].append({"setting": st, "kl": a["kl"] / a["n"], "top1": a["top1"] / a["n"], "nll": a["nll"] / a["n"],
                               "distinct": dist, "experts_saved_per_call": saved})
        if final:
            print(json.dumps({"label": st.get("label", ""), "kl": round(a["kl"] / a["n"], 5), "top1": round(a["top1"] / a["n"], 4),
                              "nll": round(a["nll"] / a["n"], 4), "saved": round(saved, 1)}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sensitivity", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--heldout", required=True)
        p.add_argument("--prompts", type=int, default=8)
        p.add_argument("--shape", choices=("b1", "b4"), default="b1")
        p.add_argument("--out", required=True)
        if name == "sensitivity":
            p.add_argument("--floors", default="8,12,16")
            p.add_argument("--uniform", default="8,10,12,14,16,18")
        else:
            p.add_argument("--settings", required=True)
    args = ap.parse_args()
    from sharar.model.nemotron import config as c
    blocks = [i for i, k in enumerate(c.BLOCK_TYPES) if k == "moe"]
    if args.cmd == "sensitivity":
        settings = [{"label": f"b{b}:F{f}", "floors": {b: f}} for b in blocks for f in map(int, args.floors.split(","))]
        settings += [{"label": f"uniform F{f}", "floors": {b: f for b in blocks}} for f in map(int, args.uniform.split(","))]
    else:
        settings = json.loads(Path(args.settings).read_text())
        for s in settings:
            s["floors"] = {int(k): v for k, v in s.get("floors", {}).items()}
    run(args, settings)


if __name__ == "__main__":
    main()
