"""Offline screen: a prompt-lookup n-gram branch beside DFlash2 in spare verify rows.

Reads Sharar request traces (``sharar serve`` with traces on) and replays each
recorded verification round from its anchor. For every round it computes what a
second, independent proposal branch built by prompt lookup (the most recent
earlier occurrence of the context's longest suffix n-gram, followed by the next
tokens) would have matched against the continuation the target actually
produced. Two branches rooted at the same anchor accept ``max(a_dflash,
a_ngram)`` tokens plus one correction, which is the prefix-plus-correction
protocol of the tree coverage screens.

This is a coverage oracle on fixed recorded histories, not a serving
measurement: real rounds re-anchor after every union, the target's numerics
shift with row count, and the branch costs verify rows only when it fires. It
needs no GPU or model. Example::

    python3 bench/ngram_branch_screen.py $SHARAR_RESULTS/traces --widths 15 31 63 \
        --output $SHARAR_RESULTS/reviews/2026-09-14-ngram-branch/real.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

K = 15  # DFlash2 proposals per round in the recorded traces


def build_index(ctx: list[int], index: dict, start: int, end: int, min_n: int, max_n: int) -> None:
    """Index every n-gram (min_n..max_n) whose last token position is in [start, end]."""
    for e in range(max(start, min_n), end + 1):
        for n in range(min_n, max_n + 1):
            if e - n < 0:
                break
            index[(n, tuple(ctx[e - n:e]))] = e - n


def propose(ctx: list[int], index: dict, width: int, min_n: int, max_n: int) -> list[int] | None:
    """Continuation after the most recent earlier occurrence of the longest matching suffix."""
    length = len(ctx)
    for n in range(min(max_n, length - 1), min_n - 1, -1):
        pos = index.get((n, tuple(ctx[length - n:])))
        if pos is not None:
            return ctx[pos + n:pos + n + width]
    return None


def screen(paths: list[str], widths: list[int], min_n: int, max_n: int) -> dict:
    widest = max(widths)
    rounds = 0
    requests = 0
    fired = 0
    dflash = 0
    ngram_only = {w: 0 for w in widths}
    union = {w: 0 for w in widths}
    wins = {w: 0 for w in widths}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                prompt = record.get("prompt_ids")
                out = record.get("output_ids")
                rds = (record.get("spec") or {}).get("rounds")
                if not prompt or not out or not rds:
                    continue
                requests += 1
                ctx = list(prompt)
                index: dict = {}
                build_index(ctx, index, 0, len(ctx) - 1, min_n, max_n)
                pos = 0
                for rd in rds:
                    remaining = len(out) - pos
                    if remaining <= 0:
                        break
                    a_d = min(rd["a"], remaining, K)
                    proposal = propose(ctx, index, widest, min_n, max_n)
                    matched = 0
                    if proposal is not None:
                        fired += 1
                        for t, u in zip(proposal, out[pos:pos + widest]):
                            if t != u:
                                break
                            matched += 1
                    rounds += 1
                    dflash += a_d + 1
                    for w in widths:
                        a_n = min(matched, w)
                        ngram_only[w] += a_n + 1
                        union[w] += max(a_d, a_n) + 1
                        if a_n > a_d:
                            wins[w] += 1
                    emitted = rd["a"] + 1
                    start = len(ctx)
                    ctx.extend(out[pos:pos + emitted])
                    build_index(ctx, index, start, len(ctx) - 1, min_n, max_n)
                    pos += emitted
    if rounds == 0:
        raise SystemExit("no speculative rounds found")
    return {
        "files": [os.path.basename(p) for p in paths],
        "requests": requests,
        "rounds": rounds,
        "min_n": min_n,
        "max_n": max_n,
        "branch_fires_fraction": fired / rounds,
        "dflash2_tokens_per_round": dflash / rounds,
        "widths": {
            str(w): {
                "ngram_alone_tokens_per_round": ngram_only[w] / rounds,
                "union_tokens_per_round": union[w] / rounds,
                "union_gain": union[w] / dflash - 1,
                "ngram_wins_fraction": wins[w] / rounds,
                "mean_rows_if_conditional": 16 + w * fired / rounds + fired / rounds,
            }
            for w in widths
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, help="trace directory (traces-*.jsonl files)")
    ap.add_argument("--widths", type=int, nargs="+", default=[15, 31, 63], help="n-gram branch widths in rows")
    ap.add_argument("--min-n", type=int, default=3, help="shortest suffix n-gram that may match")
    ap.add_argument("--max-n", type=int, default=12, help="longest suffix n-gram tried first")
    ap.add_argument("--output", type=Path, default=None, help="write the JSON result here")
    args = ap.parse_args()
    paths = sorted(glob.glob(str(args.traces / "*.jsonl")))
    if not paths:
        raise SystemExit(f"no trace files under {args.traces}")
    result = screen(paths, args.widths, args.min_n, args.max_n)
    result["traces"] = str(args.traces)
    print(f"{args.traces}: {result['requests']} requests, {result['rounds']} rounds, "
          f"branch fires in {100 * result['branch_fires_fraction']:.1f}% of rounds")
    print(f"  DFlash2 alone {result['dflash2_tokens_per_round']:.3f} tok/round")
    for w, row in result["widths"].items():
        print(f"  {w:>3}-row n-gram branch: alone {row['ngram_alone_tokens_per_round']:.3f}, "
              f"union {row['union_tokens_per_round']:.3f} ({100 * row['union_gain']:+.1f}%), "
              f"n-gram wins {100 * row['ngram_wins_fraction']:.1f}% of rounds")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
