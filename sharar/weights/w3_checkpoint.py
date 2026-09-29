"""Versioned, tensor-only calibrated decode copies for the pinned target.

The NVFP4/FP8 prefill weights remain in the original checkpoint. These files
only replace the optional W3 copies, before CUDA graph capture. The launcher
serves `$SHARAR_RESULTS/w3-checkpoints/<target>-current`; when that is missing
it runs `fetch` in the image for a target with a published checkpoint:

    python3 -m sharar.weights.w3_checkpoint fetch --target flashnext --root /w3-checkpoints
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sharar.model.qwen38_config import MODEL_REVISION

FORMAT_VERSION = 1


def model_linears(model):
    """The 27B's projections by checkpoint-independent name (`Qwen38.linears()`)."""
    for layer in model.layers:
        for name in ("gate_up", "down"):
            yield f"layers.{layer.index}.mlp.{name}", getattr(layer.mlp, name)
        for name in ("in_qkv", "in_z", "out", "qkv", "o"):
            linear = getattr(layer.mixer, name, None)
            if linear is not None:
                yield f"layers.{layer.index}.mixer.{name}", linear


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_weight(path: Path, weight) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file({"planes": weight.planes.cpu().contiguous(), "scales": weight.scales.cpu().contiguous(),
               "row_scale": weight.row_scale.cpu().contiguous()}, str(temporary))
    temporary.replace(path)
    return {"file": path.name, "sha256": sha256(path), "out_features": weight.out_features,
            "in_features": weight.in_features, "codebook": list(weight.codebook)}


class CalibratedWeights:
    def __init__(self, directory: str | Path, revision: str = MODEL_REVISION):
        """Calibrated copies for the target at `revision` (the 27B's by default; Flash-Next passes its own)."""
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT_VERSION or self.manifest.get("target_revision") != revision:
            raise ValueError("calibrated W3 checkpoint format or target revision does not match")
        self.weights = self.manifest["weights"]
        self.identity = sha256(self.directory / "manifest.json")

    def load(self, name: str, linear):
        from sharar.kernels.w3 import W3Weight, _codebook, lut_of
        entry = self.weights[name]
        n = linear.out_features
        k = getattr(linear, "in_features", linear.weight.shape[1])
        if (entry["out_features"], entry["in_features"]) != (n, k):
            raise ValueError(f"calibrated W3 shape mismatch for {name}")
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("calibrated W3 files must be directly inside the checkpoint")
        path = self.directory / filename
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"calibrated W3 checksum mismatch for {name}")
        tensors = load_file(str(path), device="cpu")
        expected = {"planes": ((n // 32, k // 128, 3, 32, 4), torch.uint32),
                    "scales": ((n // 32, k // 128, 4, 32), torch.uint8),
                    "row_scale": ((n,), torch.float32)}
        if tensors.keys() != expected.keys() or any(tuple(tensors[key].shape) != shape or tensors[key].dtype != dtype
                                                     for key, (shape, dtype) in expected.items()):
            raise ValueError(f"invalid calibrated W3 tensors for {name}")
        levels = _codebook(entry["codebook"])
        # On GB10, copying a file-backed mmap directly to CUDA is much slower
        # than staging it in ordinary CPU storage, as the draft loader does.
        # Only one projection is staged at a time.
        return W3Weight(*(tensors[key].clone().to(linear.weight.device) for key in expected), n, k, levels, lut_of(levels))


def fetch(target, root: str | Path, log=print) -> Path:
    """Download `target`'s published copies (`w3_repo` at the pinned `w3_revision`) into
    root/<target>-<revision>, verify every file against the manifest, and point root/<target>-current
    at them unless that link exists already. Returns the directory."""
    if not target.w3_repo:
        raise LookupError(f"{target.name} has no published 3-bit checkpoint; calibrate one (docs/w3-calibration.md)")
    from huggingface_hub import snapshot_download
    root = Path(root)
    dest = root / f"{target.name}-{target.w3_revision[:12]}"
    if not (dest / "manifest.json").is_file():
        partial = dest.with_name(dest.name + ".partial")
        log(f"fetching {target.w3_repo}@{target.w3_revision[:12]} into {dest}")
        snapshot_download(target.w3_repo, revision=target.w3_revision, local_dir=partial,
                          allow_patterns=["manifest.json", "*.safetensors"])
        calibrated = CalibratedWeights(partial, revision=target.model_revision)
        for name, entry in calibrated.weights.items():
            if sha256(partial / entry["file"]) != entry["sha256"]:
                raise ValueError(f"fetched W3 checksum mismatch for {name}")
        partial.rename(dest)
    link = root / f"{target.name}-current"
    if not link.exists():
        link.unlink(missing_ok=True)  # a dangling link
        link.symlink_to(dest.name)
    return dest


if __name__ == "__main__":
    import argparse
    from sharar.target import TARGETS
    ap = argparse.ArgumentParser(prog="python3 -m sharar.weights.w3_checkpoint")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the target's published calibrated copies")
    f.add_argument("--target", choices=sorted(TARGETS), required=True)
    f.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(fetch(TARGETS[args.target], args.root))
    except LookupError as error:
        raise SystemExit(str(error))
