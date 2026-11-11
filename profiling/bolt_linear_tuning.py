"""Bolt's b12x projections at the row counts serving runs them, for `hibana/model/bolt/tuning.py`: the served plan's time
and bandwidth for each shape, and with --sweep a search of the A16 geometries (row tile, N tile 64/128, K tile
64/128/256, split-K 1/2/4/8) that prints a `CONFIGS` entry wherever the best beats b12x's heuristic by 3% or more.

Every projection of a shape runs per timing (the target's 40 Mamba projections, 40 shared experts, ...; the drafter's
five layers), so weights stream from DRAM as in a round; times are CUDA-graph replays, the median of seven. Row counts
are those serving uses with seven drafts (`--spec 7`): the target verifies 8 rows a request, the LiLiCorr drafter runs
7, and the vocabulary head serves both. A sweep compiles each geometry once per shape and row count (about 3 s).

    python -m profiling.bolt_linear_tuning                         # survey: served time and GB/s
    python -m profiling.bolt_linear_tuning --sweep lm_head draft_fc  # entries for those shapes
"""
import argparse
import itertools
import logging
import os
import sys

TARGET_ROWS = (8, 16, 32)  # 1, 2 and 4 requests verifying 7 drafts
DRAFT_ROWS = (7, 14, 28)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", nargs="*", default=[], help="shape names to sweep (see the survey)")
    args = ap.parse_args()
    from hibana.launch import settings
    os.environ.update(settings.resolve("bolt", "fast").engine_env(None))
    logging.getLogger("b12x.preparation").setLevel(logging.ERROR)  # every swept geometry is prepared at first use
    import torch
    from b12x.gemm import blockscaled
    from b12x.preparation.types import require_prepared
    from hibana import target as targets
    from hibana.model.bolt import tuning
    from hibana.model.lilicorr import LiLiCorrDraftModel
    from hibana.proposers.factory import draft_snapshot

    t = targets.NEMOTRON_BOLT
    model = targets.load_model(t, t.snapshot_dir(), log=lambda *a: None)
    draft = LiLiCorrDraftModel(draft_snapshot("dflash2", None, t), model, 4096)
    shapes: dict[str, tuple[list, tuple[int, ...]]] = {}
    for name, lin in model.linears():
        key = "lm_head" if name == "lm_head" else name.rsplit(".", 1)[-1]
        shapes.setdefault(key, ([], TARGET_ROWS))[0].append(lin)
    shapes["lm_head"] = ([model.lm_head], TARGET_ROWS + DRAFT_ROWS)
    for name, lin in draft.packed.items():
        key = f"draft_{lin.out_features}x{lin.in_features}"
        shapes.setdefault(key, ([], DRAFT_ROWS))[0].append(lin)

    def timed(fn, reps=10):
        fn()
        torch.cuda.synchronize()
        graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(7):
            a.record()
            for _ in range(reps):
                graph.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) / reps)
        return sorted(times)[3]

    def plan(lin, rows, override=None):
        query = blockscaled.BlockscaledQuery(
            recipe=lin.quant.lower(), num_tokens=rows, expected_m=rows, in_features=lin.in_features,
            padded_in_features=lin.in_features, out_features=lin.out_features, activation_mode="a16",
            **({"global_scale_kind": "multiplier"} if lin.quant == "NVFP4" else {}))
        p = blockscaled.plan(query, override=override and blockscaled.BlockscaledConfig(mode="a16", **override))
        require_prepared(p, "gemm.blockscaled_precision", lin.b12x.values.device)
        return p

    for key, (lins, rowset) in shapes.items():
        lin0, nbytes = lins[0], sum(lin.bytes for lin in lins)
        recipe, n, k = lin0.quant.lower(), lin0.out_features, lin0.in_features
        for rows in rowset:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            served = timed(lambda: [lin(x) for lin in lins])
            line = f"{key:16s} [{n}, {k}] {recipe:8s} {rows:2d} rows x{len(lins):<3d} served {served * 1e3:7.1f} us ({nbytes / served / 1e6:4.0f} GB/s)"
            if key not in args.sweep:
                print(line, flush=True)
                continue
            heuristic = plan(lin0, rows)
            base = timed(lambda: [blockscaled.mm(x, lin.b12x, plan=heuristic) for lin in lins])
            best = (base, None)
            for tm, tn, tk, sk in itertools.product((None, 8, 16, 32, 64), (64, 128), (64, 128, 256), (1, 2, 4, 8)):
                cfg = {"tile_m": tm, "tile_n": tn, "tile_k": tk, "split_k": sk}
                try:
                    p = plan(lin0, rows, cfg)
                    ms = timed(lambda: [blockscaled.mm(x, lin.b12x, plan=p) for lin in lins])
                except Exception:  # noqa: BLE001 - a geometry b12x rejects for this shape
                    continue
                best = min(best, (ms, (tm, tn, tk, sk)), key=lambda r: r[0])
            ms, cfg = best
            print(line + f"; heuristic {base * 1e3:.1f} us, best {ms * 1e3:.1f} us", flush=True)
            per = lambda v: round(v * 1e3 / len(lins))  # noqa: E731 - the table's times are per projection
            if cfg is not None and ms < 0.97 * base:
                print(f"    ('{recipe}', {n}, {k}, {rows}): {cfg},  # {per(base)} -> {per(ms)} us", flush=True)
            elif tuning.config(recipe, n, k, rows) is not None:
                print(f"    drop ('{recipe}', {n}, {k}, {rows}): the heuristic is within 3% of every geometry", flush=True)


if __name__ == "__main__":
    sys.exit(main())
