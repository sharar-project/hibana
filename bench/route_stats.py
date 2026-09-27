# SPDX-License-Identifier: Apache-2.0
"""Router statistics over a Flash-Next route dataset (#32; the dataset comes from bench/flashnext_routes.py).

    python3 -m bench.route_stats --data DATA --out report.json [--ab-dir AB] > tables.md

Per arm directory (`<arm>-<mode>`, e.g. fast-k4-greedy):

- expert use per layer and domain over one row per output token (decode rows and verify rows whose
  input was kept): experts that carry 50% / 90% of the router mass, effective experts, top share;
- router mass by rank of the top-10 (renormalized combine weights) and the softmax mass the top-10 keep;
- distinct experts per verify block, by block prefix size (the first m rows) and by layer;
- expert bytes first used by rejected rows: in each round and layer every distinct expert belongs to
  the first row (draft order) that routes to it; rows past the accepted prefix are rejected;
- acceptance by the draft's top-1 probability, per draft position.

Row model: a verify block of k drafts has k + 1 rows; row 0 is the pending token, row j (j >= 1) the
drafted token j - 1. With `a` drafts accepted, rows 0..a carry kept tokens and rows a+1..k rejected ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPERT_BYTES = 2_764_824  # one NVFP4 expert of the checkpoint: fc1 (up, gate) and fc2 weights with block scales
DOMAINS = ("code", "math", "prose", "multilingual", "structured", "chat", "agent")
P1_BINS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001)
EQUAL_TOKENS = 1500  # size-matched skew: the fewer tokens a sample has, the fewer experts carry its mass


# ---- pure functions (tests/unit/test_route_stats.py) ----
def kept_rows(kind: np.ndarray, offset: np.ndarray, accepted_of_row: np.ndarray) -> np.ndarray:
    """Mask of rows whose input token is part of the output: decode rows, verify rows at offset <= accepted."""
    return (kind == 0) | (offset <= accepted_of_row)


def mass_per_expert(ids: np.ndarray, probs: np.ndarray, experts: int) -> tuple[np.ndarray, np.ndarray]:
    """ids/probs [N, L, k] -> (combine-weight mass [L, E], selection counts [L, E])."""
    n, layers, k = ids.shape
    p = probs.astype(np.float64)
    w = p / p.sum(-1, keepdims=True)
    flat = (np.arange(layers)[None, :, None] * experts + ids.astype(np.int64)).reshape(-1)
    mass = np.bincount(flat, weights=w.reshape(-1), minlength=layers * experts).reshape(layers, experts)
    counts = np.bincount(flat, minlength=layers * experts).reshape(layers, experts)
    return mass, counts


def experts_for_share(mass: np.ndarray, share: float) -> np.ndarray:
    """[..., E] -> [...] how many experts, largest first, carry `share` of the mass."""
    ordered = -np.sort(-mass, axis=-1)
    cum = ordered.cumsum(-1) / ordered.sum(-1, keepdims=True)
    return (cum < share - 1e-12).sum(-1) + 1


def effective_experts(mass: np.ndarray) -> np.ndarray:
    """exp(entropy) of the mass distribution over experts, per row of [..., E]."""
    p = mass / mass.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(-1)
    return np.exp(h)


def top_set_overlap(a: np.ndarray, b: np.ndarray, share: float) -> np.ndarray:
    """Mass [L, E] of two samples -> [L] Jaccard overlap of the expert sets that carry `share` of each one's mass."""
    out = np.zeros(a.shape[0])
    for layer in range(a.shape[0]):
        sets = []
        for m in (a[layer], b[layer]):
            n = int(experts_for_share(m, share))
            sets.append(set(np.argsort(-m, kind="stable")[:n].tolist()))
        out[layer] = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
    return out


def rank_shares(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """probs [N, L, k] (descending softmax) -> (mean renormalized share per rank [L, k], mean top-k softmax mass [L])."""
    p = probs.astype(np.float64)
    total = p.sum(-1, keepdims=True)
    return (p / total).mean(0), total[..., 0].mean(0)


def distinct_by_prefix(ids: np.ndarray, experts: int) -> np.ndarray:
    """ids [rows, L, k] of one block -> [rows, L] distinct experts touched by the first m = 1..rows rows."""
    rows, layers, k = ids.shape
    seen = np.zeros((layers, experts), dtype=bool)
    out = np.zeros((rows, layers), dtype=np.int32)
    for m in range(rows):
        seen[np.repeat(np.arange(layers), k), ids[m].reshape(-1)] = True
        out[m] = seen.sum(1)
    return out


def first_use_split(ids: np.ndarray, accepted: int, experts: int) -> tuple[np.ndarray, np.ndarray]:
    """ids [k+1, L, k] of one verify block -> ([L] distinct experts, [L] of them first used by a rejected row
    (row index > accepted), rows scanned in draft order)."""
    rows, layers, k = ids.shape
    first = np.full((layers, experts), rows, dtype=np.int32)
    for m in range(rows - 1, -1, -1):  # descending, so the earliest row wins
        first[np.repeat(np.arange(layers), k), ids[m].reshape(-1)] = m
    used = first < rows
    return used.sum(1), (used & (first > accepted)).sum(1)


def acceptance_by_confidence(p1: np.ndarray, accepted: np.ndarray, bins=P1_BINS) -> list[dict]:
    """p1 [R, k] draft top-1 probability, accepted [R] -> per position j and p1 bin: rounds that reached j
    (accepted >= j), and the share that accepted draft j (accepted > j)."""
    out = []
    for j in range(p1.shape[1]):
        reached = accepted >= j
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = reached & (p1[:, j] >= lo) & (p1[:, j] < hi)
            n = int(sel.sum())
            out.append({"position": j, "p1_lo": lo, "p1_hi": min(hi, 1.0), "n": n,
                        "accept": round(float((accepted[sel] > j).mean()), 4) if n else None})
    return out


# ---- dataset ----
def load_arm(path: Path) -> dict:
    rows = dict(np.load(path / "rows.npz"))
    rounds = dict(np.load(path / "rounds.npz"))
    requests = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    meta = json.loads((path / "meta.json").read_text())
    cats = np.array([r["cat"] for r in requests])
    rows["cat"] = cats[rows["request"]]
    rounds["cat"] = cats[rounds["request"]]
    # accepted count of each row's round (verify rows), for the kept-row mask
    key = rounds["request"].astype(np.int64) * 1_000_000 + rounds["round"]
    order = np.argsort(key)
    row_key = rows["request"].astype(np.int64) * 1_000_000 + np.maximum(rows["round"], 0)
    pos = np.clip(np.searchsorted(key[order], row_key), 0, len(key) - 1)
    rows["accepted_of_row"] = np.where(rows["kind"] == 1, rounds["accepted"][order][pos], 0)
    return {"rows": rows, "rounds": rounds, "requests": requests, "meta": meta}


def skew(arm: dict, experts: int) -> dict:
    rows = arm["rows"]
    kept = kept_rows(rows["kind"], rows["offset"], rows["accepted_of_row"])
    out = {}
    for name in ("all", *DOMAINS):
        sel = kept if name == "all" else kept & (rows["cat"] == name)
        if not sel.any():
            continue
        mass, counts = mass_per_expert(rows["ids"][sel], rows["probs"][sel], experts)
        n50, n90 = experts_for_share(mass, 0.5), experts_for_share(mass, 0.9)
        eff = effective_experts(mass)
        top = mass.max(1) / mass.sum(1)
        unused = (counts == 0).sum(1)
        out[name] = {"tokens": int(sel.sum()),
                     "experts_50pct": summary(n50), "experts_90pct": summary(n90), "effective_experts": summary(eff),
                     "top_expert_share": summary(top), "unused_experts": summary(unused),
                     "per_layer": {"experts_50pct": n50.tolist(), "experts_90pct": n90.tolist(), "effective": np.round(eff, 1).tolist()}}
    # Size-matched: the same number of token rows drawn from each domain (fixed seed), so domains compare fairly.
    rng = np.random.default_rng(0)
    masses, halves = {}, {}
    for name in DOMAINS:
        idx = np.nonzero(kept & (rows["cat"] == name))[0]
        if len(idx) < EQUAL_TOKENS:
            continue
        pick = np.sort(rng.choice(idx, EQUAL_TOKENS, replace=False))
        mass, _ = mass_per_expert(rows["ids"][pick], rows["probs"][pick], experts)
        masses[name] = mass
        # The overlap's noise floor: the domain's prompts split in two halves (alternate requests), one sample each.
        reqs = np.unique(rows["request"][idx])
        parts = [idx[np.isin(rows["request"][idx], reqs[h::2])] for h in (0, 1)]
        if len(reqs) >= 2 and min(len(p) for p in parts) >= EQUAL_TOKENS:
            a, b = (np.sort(rng.choice(p, EQUAL_TOKENS, replace=False)) for p in parts)
            halves[name] = (mass_per_expert(rows["ids"][a], rows["probs"][a], experts)[0],
                            mass_per_expert(rows["ids"][b], rows["probs"][b], experts)[0])
        out[name]["equal_tokens"] = {"tokens": EQUAL_TOKENS, "experts_50pct": summary(experts_for_share(mass, 0.5)),
                                     "experts_90pct": summary(experts_for_share(mass, 0.9))}
    names = sorted(masses)
    out["overlap_50pct_sets"] = {"tokens_per_domain": EQUAL_TOKENS, "domains": names,
                                 "diagonal": "the domain's prompts in two halves, one sample each (None: too few tokens per half)",
                                 "mean_jaccard_over_layers": [[round(float(top_set_overlap(*(halves[x] if x == y else (masses[x], masses[y])), 0.5).mean()), 3)
                                                               if x != y or x in halves else None for y in names] for x in names]}
    return out


def ranks(arm: dict) -> dict:
    rows = arm["rows"]
    kept = kept_rows(rows["kind"], rows["offset"], rows["accepted_of_row"])
    probs = rows["probs"][kept].astype(np.float64)
    share, topk = rank_shares(probs)
    cum = (probs / probs.sum(-1, keepdims=True)).cumsum(-1)  # [N, L, k]
    k = share.shape[1]
    return {"rank_share_mean": np.round(share.mean(0), 4).tolist(),
            "cumulative_share_by_rank": [{"rank": r + 1, "mean": round(float(cum[..., r].mean()), 4),
                                          "p10": round(float(np.percentile(cum[..., r], 10)), 4),
                                          "p50": round(float(np.percentile(cum[..., r], 50)), 4),
                                          "p90": round(float(np.percentile(cum[..., r], 90)), 4)} for r in range(k)],
            "ranks_7_10_share": summary(share[:, 6:].sum(1)), "ranks_9_10_share": summary(share[:, 8:].sum(1)),
            "ranks_7_10_per_layer": np.round(share[:, 6:].sum(1), 4).tolist(),
            "ranks_9_10_per_layer": np.round(share[:, 8:].sum(1), 4).tolist(),
            "topk_softmax_mass": summary(topk), "topk_softmax_mass_per_layer": np.round(topk, 4).tolist(),
            "rows_with_last_4_ranks_over_20pct": round(float(((1 - cum[..., k - 5]) > 0.2).mean()), 4) if k > 4 else None}


def blocks(arm: dict, experts: int) -> dict:
    rows, rounds = arm["rows"], arm["rounds"]
    k = int(rounds["k"].max())
    by_prefix = np.zeros((k + 1, rows["ids"].shape[1]), dtype=np.float64)
    split = {"distinct": np.zeros(rows["ids"].shape[1]), "rejected": np.zeros(rows["ids"].shape[1])}
    by_domain, by_accept = {}, {}
    by_depth = np.zeros((k + 1, 2))  # truncated depth d: the block's first d + 1 rows, accepted min(a, d)
    n = 0
    for i in range(len(rounds["request"])):
        r0, kk, a = int(rounds["row0"][i]), int(rounds["k"][i]), int(rounds["accepted"][i])
        if r0 < 0 or kk != k:
            continue
        ids = rows["ids"][r0:r0 + kk + 1]
        by_prefix += distinct_by_prefix(ids, experts)
        d, w = first_use_split(ids, a, experts)
        split["distinct"] += d
        split["rejected"] += w
        dom = by_domain.setdefault(str(rounds["cat"][i]), np.zeros(3))
        dom += (d.sum(), w.sum(), 1)
        acc = by_accept.setdefault(a, np.zeros(3))
        acc += (d.sum(), w.sum(), 1)
        for depth in range(1, k + 1):
            dd, ww = first_use_split(ids[:depth + 1], min(a, depth), experts)
            by_depth[depth] += (dd.sum(), ww.sum())
        n += 1
    by_prefix /= max(1, n)
    layers = by_prefix.shape[1]
    frac_layer = split["rejected"] / np.maximum(1, split["distinct"])
    return {
        "rounds": n, "k": k,
        "distinct_by_block_rows": [{"rows": m + 1, "mean_over_layers": round(float(by_prefix[m].mean()), 2),
                                    "min_layer": round(float(by_prefix[m].min()), 2), "max_layer": round(float(by_prefix[m].max()), 2)}
                                   for m in range(k + 1)],
        "distinct_full_block_per_layer": np.round(by_prefix[k], 2).tolist(),
        "rejected_first_use": {
            "fraction_of_round_expert_bytes": round(float(split["rejected"].sum() / split["distinct"].sum()), 4),
            "expert_mb_per_round": round(float(split["distinct"].sum() / n * EXPERT_BYTES / 1e6), 1),
            "rejected_mb_per_round": round(float(split["rejected"].sum() / n * EXPERT_BYTES / 1e6), 1),
            "per_layer_fraction": np.round(frac_layer, 4).tolist(),
            "per_layer_summary": summary(frac_layer),
            "by_domain": {c: {"fraction": round(float(v[1] / v[0]), 4), "rounds": int(v[2]),
                              "rejected_mb_per_round": round(float(v[1] / v[2] * EXPERT_BYTES / 1e6), 1)} for c, v in sorted(by_domain.items())},
            "by_truncated_depth": {int(dd): {"fraction": round(float(by_depth[dd, 1] / by_depth[dd, 0]), 4),
                                             "expert_mb_per_round": round(float(by_depth[dd, 0] / n * EXPERT_BYTES / 1e6), 1),
                                             "rejected_mb_per_round": round(float(by_depth[dd, 1] / n * EXPERT_BYTES / 1e6), 1)}
                                   for dd in range(1, k + 1)},
            "by_accepted": {int(a): {"fraction": round(float(v[1] / v[0]), 4), "rounds": int(v[2]),
                                     "distinct_per_layer": round(float(v[0] / v[2] / layers), 1)} for a, v in sorted(by_accept.items())},
        },
    }


def acceptance(arm: dict) -> dict:
    rounds = arm["rounds"]
    p1, acc = rounds["p1"].astype(np.float64), rounds["accepted"].astype(np.int64)
    ok = ~np.isnan(p1).any(1)
    k = p1.shape[1]
    reach = [(acc[ok] >= j) for j in range(k)]
    return {"rounds": int(ok.sum()), "tokens_per_round": round(float(rounds["emitted"].mean()), 3),
            "accept_rate_by_position": [round(float((acc[ok][reach[j]] > j).mean()), 4) if reach[j].any() else None for j in range(k)],
            "p1_mean_by_position": [round(float(p1[ok, j].mean()), 4) for j in range(k)],
            "table": acceptance_by_confidence(p1[ok], acc[ok])}


def summary(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"median": round(float(np.median(v)), 4), "mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def overhead(ab_dir: Path) -> dict:
    """The capture-cost cells of bench/flashnext_routes.py ab: per-round wall time per label and arm."""
    out = {}
    for path in sorted(ab_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        out[doc["label"]] = {"router_traces": doc["router_traces"], "summary": doc["summary"], "outputs": doc["outputs"]}
    return out


def tables(report: dict) -> str:
    """Markdown tables for SUMMARY.md."""
    lines = []
    for arm, r in report["arms"].items():
        lines += [f"### {arm}", "", f"{r['tokens']} output tokens, {r['blocks']['rounds']} verify rounds, "
                  f"{r['acceptance']['tokens_per_round']} tokens per round", ""]
        lines += ["| domain | tokens | experts for 50% (median / range over layers) | for 90% | effective experts | top expert share | unused experts |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for d, s in r["skew"].items():
            if d == "overlap_50pct_sets":
                continue
            lines.append(f"| {d} | {s['tokens']} | {s['experts_50pct']['median']:.0f} ({s['experts_50pct']['min']:.0f}-{s['experts_50pct']['max']:.0f}) "
                         f"| {s['experts_90pct']['median']:.0f} ({s['experts_90pct']['min']:.0f}-{s['experts_90pct']['max']:.0f}) "
                         f"| {s['effective_experts']['median']:.0f} | {s['top_expert_share']['median']:.3f} | {s['unused_experts']['median']:.0f} |")
        eq = {d: s["equal_tokens"] for d, s in r["skew"].items() if isinstance(s, dict) and "equal_tokens" in s}
        if eq:
            n = next(iter(eq.values()))["tokens"]
            lines += ["", f"Size-matched ({n} tokens per domain): experts for 50% / 90% of the mass, median over layers: "
                      + "; ".join(f"{d} {v['experts_50pct']['median']:.0f} / {v['experts_90pct']['median']:.0f}" for d, v in eq.items()) + "."]
            ov = r["skew"]["overlap_50pct_sets"]
            lines += ["", "| overlap of 50%-mass expert sets (Jaccard, mean over layers) | " + " | ".join(ov["domains"]) + " |",
                      "|---|" + "---:|" * len(ov["domains"])]
            for name, row in zip(ov["domains"], ov["mean_jaccard_over_layers"]):
                lines.append(f"| {name} | " + " | ".join(f"{x:.2f}" if x is not None else "-" for x in row) + " |")
            lines.append("Diagonal: the domain's prompts split in two halves, one sample of each (the noise floor for new prompts of the same domain).")
        rk = r["ranks"]
        lines += ["", "| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |", "|---|" + "---:|" * 10,
                  "| mean share | " + " | ".join(f"{x:.3f}" for x in rk["rank_share_mean"]) + " |",
                  "| cumulative p10 | " + " | ".join(f"{c['p10']:.3f}" for c in rk["cumulative_share_by_rank"]) + " |",
                  "| cumulative p50 | " + " | ".join(f"{c['p50']:.3f}" for c in rk["cumulative_share_by_rank"]) + " |",
                  "| cumulative p90 | " + " | ".join(f"{c['p90']:.3f}" for c in rk["cumulative_share_by_rank"]) + " |", "",
                  f"Ranks 7-10: {rk['ranks_7_10_share']['median']:.3f} of the combine weight (layers {rk['ranks_7_10_share']['min']:.3f}-{rk['ranks_7_10_share']['max']:.3f}); "
                  f"ranks 9-10: {rk['ranks_9_10_share']['median']:.3f} ({rk['ranks_9_10_share']['min']:.3f}-{rk['ranks_9_10_share']['max']:.3f}). "
                  f"The top 10 keep {rk['topk_softmax_mass']['median']:.3f} of the softmax (layers {rk['topk_softmax_mass']['min']:.3f}-{rk['topk_softmax_mass']['max']:.3f}).", ""]
        b = r["blocks"]
        lines += ["| block rows | " + " | ".join(str(x["rows"]) for x in b["distinct_by_block_rows"]) + " |",
                  "|---|" + "---:|" * len(b["distinct_by_block_rows"]),
                  "| distinct experts per layer (mean) | " + " | ".join(f"{x['mean_over_layers']:.1f}" for x in b["distinct_by_block_rows"]) + " |",
                  "| layer min-max | " + " | ".join(f"{x['min_layer']:.0f}-{x['max_layer']:.0f}" for x in b["distinct_by_block_rows"]) + " |", ""]
        rj = b["rejected_first_use"]
        lines += [f"Expert bytes first used by rejected rows: **{rj['fraction_of_round_expert_bytes']:.3f}** of the round's expert bytes "
                  f"({rj['rejected_mb_per_round']} of {rj['expert_mb_per_round']} MB per round; layers {rj['per_layer_summary']['min']:.3f}-{rj['per_layer_summary']['max']:.3f})", "",
                  "| domain | rounds | fraction | rejected MB per round |", "|---|---:|---:|---:|"]
        for d, v in rj["by_domain"].items():
            lines.append(f"| {d} | {v['rounds']} | {v['fraction']:.3f} | {v['rejected_mb_per_round']} |")
        td = rj["by_truncated_depth"]
        lines += ["", "The same rounds cut to depth d (rows 0..d, accepted min(a, d)):", "",
                  "| depth d | " + " | ".join(str(x) for x in td) + " |", "|---|" + "---:|" * len(td),
                  "| rejected fraction | " + " | ".join(f"{v['fraction']:.3f}" for v in td.values()) + " |",
                  "| rejected / all expert MB per round | " + " | ".join(f"{v['rejected_mb_per_round']:.0f} / {v['expert_mb_per_round']:.0f}" for v in td.values()) + " |"]
        lines += ["", "| accepted drafts | rounds | fraction | distinct experts per layer |", "|---:|---:|---:|---:|"]
        for a, v in rj["by_accepted"].items():
            lines.append(f"| {a} | {v['rounds']} | {v['fraction']:.3f} | {v['distinct_per_layer']} |")
        ac = r["acceptance"]
        k = len(ac["accept_rate_by_position"])
        lines += ["", "| draft top-1 p | " + " | ".join(f"pos {j}" for j in range(k)) + " |", "|---|" + "---:|" * k]
        for lo, hi in zip(P1_BINS[:-1], P1_BINS[1:]):
            cells = []
            for j in range(k):
                c = next(x for x in ac["table"] if x["position"] == j and x["p1_lo"] == lo)
                cells.append(f"{c['accept']:.2f} ({c['n']})" if c["n"] else "-")
            lines.append(f"| [{lo:.2f}, {min(hi, 1.0):.2f}) | " + " | ".join(cells) + " |")
        lines.append("| all | " + " | ".join(f"{x:.2f}" if x is not None else "-" for x in ac["accept_rate_by_position"]) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset directory (one sub-directory per arm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ab-dir", default=None, help="bench/flashnext_routes.py ab outputs for the overhead cell")
    ap.add_argument("--experts", type=int, default=512)
    args = ap.parse_args()
    report = {"expert_bytes": EXPERT_BYTES, "arms": {}}
    for path in sorted(p for p in Path(args.data).iterdir() if (p / "rows.npz").is_file()):
        arm = load_arm(path)
        report["arms"][path.name] = {"tokens": int(sum(len(r["output_ids"]) for r in arm["requests"])), "meta": {k: arm["meta"].get(k) for k in ("profile", "k", "mode", "acceptance")},
                                     "skew": skew(arm, args.experts), "ranks": ranks(arm), "blocks": blocks(arm, args.experts),
                                     "acceptance": acceptance(arm)}
    if args.ab_dir:
        report["overhead"] = overhead(Path(args.ab_dir))
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(tables(report))


if __name__ == "__main__":
    main()
