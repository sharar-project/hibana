"""Where Nemotron's 3-bit decode copies lose quality: the lossy gate's layer 1 for many coverage arms in one model load.

    flock $SHARAR_RESULTS/.gpu.lock python profiling/nemotron_w3_sensitivity.py \\
        --checkpoint W3DIR --gate $SHARAR_RESULTS/reviews/<date>-lossy-gate-<name> --arms arms.json --out OUT

The model loads with every copy of --checkpoint attached (and the copies of each --extra LABEL=DIR held beside them).
An arm then picks, per `Nemotron.linears()` name, which copy serves calls of at most 64 rows (or none: FP8) and scores
the gate baseline's frozen sequences (`--gate`/baseline: sequences.json, base.pt, floor.json) exactly as a candidate
arm of `tools/lossy-gate.py` does (64 rows per forward, tools/gate/lossy_gate.score_all), so its numbers are the
gate's layer 1: KL(exact || arm), argmax agreement, NLL, and the excess over the floor with its bootstrap interval.

arms.json is a list of {"label": ..., "w3": REGEX (names served 3-bit; default all), "fp8": REGEX (names kept FP8),
"source": LABEL (default "main"), "sources": [[REGEX, LABEL], ...] (first match wins), "fp8_rows": [[REGEX, [[lo, hi],
...]], ...] (output rows of a 3-bit projection taken from the FP8 weights instead, as a split projection would serve
them; the harness computes both)}. One JSON line per arm on stdout
and in OUT/arms.jsonl; each arm's per-position rows in OUT/<label>.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))


def gate_module():
    spec = importlib.util.spec_from_file_location("lossy_gate_tool", REPO / "tools" / "lossy-gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RowSplice:
    """`ops.fp8_linear` with some output rows of a 3-bit call replaced by the FP8 weights' (an experiment, not a kernel)."""

    def __init__(self):
        self.rows = {}  # id(lin) -> [(lo, hi), ...]

    def __enter__(self):
        from sharar.kernels import ops
        self.ops, original, rows = ops, ops.fp8_linear, self.rows

        def fp8_linear(x, lin, x_q=None):
            out = original(x, lin, x_q=x_q)
            spans = rows.get(id(lin))
            if spans and lin.w3 is not None and out.reshape(-1, out.shape[-1]).shape[0] <= 64:
                copy, lin.w3 = lin.w3, None
                try:
                    ref = original(x, lin, x_q=x_q)
                finally:
                    lin.w3 = copy
                for lo, hi in spans:
                    out[..., lo:hi] = ref[..., lo:hi]
            return out
        self.original = original
        ops.fp8_linear = fp8_linear
        return self

    def __exit__(self, *args):
        self.ops.fp8_linear = self.original


def pick(arm: dict, name: str) -> str | None:
    """The copy label serving `name` in this arm, or None (FP8)."""
    if arm.get("fp8") and re.search(arm["fp8"], name):
        return None
    if arm.get("w3") is not None and not re.search(arm["w3"], name):
        return None
    for pattern, label in arm.get("sources", []):
        if re.search(pattern, name):
            return label
    return arm.get("source", "main")


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="the copies attached at load ('main')")
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=DIR", help="more copies held beside them")
    ap.add_argument("--gate", type=Path, required=True, help="a lossy-gate run directory (its baseline/ is read)")
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from gate import lossy_gate as worker

    from sharar import target as targets
    from sharar.engine import config
    from sharar.engine.executor import Executor
    from sharar.launch.settings import resolve
    from sharar.weights.w3_checkpoint import CalibratedWeights
    gate = gate_module()
    base_dir = args.gate / "baseline"
    seqs = json.loads((base_dir / "sequences.json").read_text())
    base = torch.load(base_dir / "base.pt")
    floor = json.loads((base_dir / "floor.json").read_text())
    arms = json.loads(args.arms.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    t = targets.get("nemotron")
    cfg = config.current().replace(target="nemotron", hf_cache=str(resolve("nemotron").hf_cache), w3_decode=True,
                                   w3_checkpoint=str(args.checkpoint))
    with config.using(cfg):
        started = time.perf_counter()
        model = targets.load_model(t, t.snapshot_dir(), log=print)
        linears = dict(model.linears())
        copies = {"main": {name: lin.w3 for name, lin in linears.items() if lin.w3 is not None}}
        for item in args.extra:
            label, directory = item.split("=", 1)
            cw = CalibratedWeights(directory, revision=t.model_revision)
            loads = [(name, linears[name], linears[name].shard) for name in linears if name in cw.weights]
            copies[label] = {name: copy for (name, _, _), copy in zip(loads, cw.load_many(loads))}
            print(f"  {label}: {len(loads)} copies from {directory}", flush=True)
        ex = Executor(model, log=None, max_batch=1, max_context=16384, num_pages=65536, num_slots=3)
        print(f"loaded in {time.perf_counter() - started:.0f}s", flush=True)
        splice = RowSplice().__enter__()
        for arm in arms:
            path = args.out / f"{arm['label']}.json"
            if path.exists():
                continue
            served = {}
            splice.rows.clear()
            for pattern, spans in arm.get("fp8_rows", []):
                for name, lin in linears.items():
                    if re.search(pattern, name):
                        splice.rows[id(lin)] = [tuple(s) for s in spans]
            for name, lin in linears.items():
                label = pick(arm, name)
                lin.w3 = copies[label].get(name) if label else None
                if lin.w3 is not None:
                    served[name] = label
            w3_bytes = sum(lin.w3.bytes for lin in linears.values() if lin.w3 is not None)
            fp8_saved = sum(lin.weight.numel() - lin.w3.bytes for lin in linears.values() if lin.w3 is not None)
            t0 = time.perf_counter()
            worker.score_all(ex, model, seqs, base, 64, arm["label"], path)
            dist = json.loads(path.read_text())
            l1 = gate.layer1(dist, floor)
            row = {"engine": "Sharar", "machine": "spark-9931", "label": arm["label"], "w3_matrices": len(served),
                   "w3_gib": round(w3_bytes / 2**30, 3), "bytes_saved_gb": round(fp8_saved / 1e9, 3),
                   "kl": round(l1["all"]["kl"], 5), "kl_floor": round(l1["all"]["kl_floor"], 5),
                   "kl_excess": [round(x, 5) for x in l1["all"]["kl_excess"]],
                   "argmax": round(l1["all"]["argmax"], 5), "argmax_floor": round(l1["all"]["argmax_floor"], 5),
                   "argmax_delta": [round(x, 5) for x in l1["all"]["argmax_delta"]],
                   "nll": round(dist["summary"]["nll"], 5),
                   "within": {d: v["within_floor"] for d, v in l1.items()},
                   "domain_kl": {d: round(v["kl"], 4) for d, v in l1.items() if d != "all"},
                   "seconds": round(time.perf_counter() - t0, 1)}
            print(json.dumps(row), flush=True)
            with (args.out / "arms.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
