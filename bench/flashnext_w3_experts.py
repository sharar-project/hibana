# SPDX-License-Identifier: Apache-2.0
"""3-bit routed experts for Flash-Next's W3 checkpoint (#73): a new checkpoint directory with the source
checkpoint's trunk entries and every layer's experts as `layers.N.experts.w13` / `.w2`.

The source is the checkpoint's NVFP4 experts, dequantized (there is no BF16 expert source). Each expert matrix is
fitted into the W3 format (kernels/w3_moe.calibrate): the trunk's Lloyd codebook, per-32 group scales on a clipping
grid minimizing the weight error, nearest level. fc1 is stored in the kernel's row order (w3_moe.fc1_order).
Serving it with the 3-bit copies on (the `fast` profile, SHARAR_W3_CHECKPOINT=DIR) replaces the NVFP4 experts.

    SHARAR_TARGET=flashnext python3 -m bench.flashnext_w3_experts --source $SHARAR_RESULTS/w3-checkpoints/flashnext-current \\
        --output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="W3 checkpoint whose trunk entries are copied")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--codebook", default="lloyd")
    ap.add_argument("--layers", default="", help="comma-separated subset (a partial checkpoint does not serve)")
    args = ap.parse_args()

    from sharar.kernels import w3, w3_moe
    from sharar.model.flashnext.moe import load_nvfp4_experts
    from sharar.target import FLASHNEXT
    from sharar.weights.safetensors import SnapshotWeights
    from sharar.weights.w3_checkpoint import CalibratedWeights, save_weight, sha256

    source = CalibratedWeights(args.source, revision=FLASHNEXT.model_revision)
    if source.expert_layers():
        raise SystemExit("the source checkpoint already carries experts")
    out = args.output
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, entry in source.weights.items():  # trunk entries unchanged, verified on the way
        src, dst = args.source / entry["file"], partial / entry["file"]
        if not dst.is_file() or sha256(dst) != entry["sha256"]:
            shutil.copyfile(src, dst)
            if sha256(dst) != entry["sha256"]:
                raise SystemExit(f"checksum mismatch copying {name}")
        weights[name] = entry

    dev = torch.device("cuda")
    w = SnapshotWeights(FLASHNEXT.snapshot_dir())
    layers = [int(v) for v in args.layers.split(",")] if args.layers else list(range(FLASHNEXT.num_layers))
    levels = w3._codebook(args.codebook)
    errors, started = {}, time.perf_counter()
    for i in layers:
        nv = load_nvfp4_experts(w, f"model.language_model.layers.{i}.mlp", FLASHNEXT.num_experts, dev)
        w.close()
        ex, report = w3_moe.calibrate(nv["w13"], nv["s13"], nv["scale2_13"], nv["w2"], nv["s2"], nv["scale2_2"], levels)
        del nv
        for key in ("w13", "w2"):
            name = f"layers.{i}.experts.{key}"
            weights[name] = dict(save_weight(partial / f"{name}.safetensors", ex.stacked(key == "w13")), experts=ex.experts)
        errors[str(i)] = {k: round(v, 5) for k, v in report.items()}
        del ex
        torch.cuda.empty_cache()
        print(f"layer {i}: relative weight error w13 {report['w13']:.4f}, w2 {report['w2']:.4f} ({time.perf_counter() - started:.0f}s)", flush=True)

    manifest = dict(source.manifest)
    manifest["weights"] = weights
    manifest["experts"] = {
        "source": "checkpoint NVFP4 experts, dequantized", "method": "nearest level, group scales on a weight-error grid",
        "codebook": list(levels), "group_size": w3.GROUP, "scale_grid": list(w3_moe.RATIOS), "fc1_rows": "w3_moe.fc1_order",
        "trunk_from": str(args.source.resolve()), "trunk_manifest_sha256": source.identity,
        "relative_weight_error": errors,
        "mean_relative_weight_error": {k: round(sum(e[k] for e in errors.values()) / len(errors), 5) for k in ("w13", "w2")},
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=1))
    partial.rename(out)
    print(json.dumps(manifest["experts"]["mean_relative_weight_error"]), "->", out, "manifest", sha256(out / "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
