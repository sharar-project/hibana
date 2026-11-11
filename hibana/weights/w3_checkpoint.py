"""Versioned, tensor-only calibrated decode copies for the pinned target.

The NVFP4/FP8 prefill weights remain in the original checkpoint. These files
only replace the optional W3 copies, before CUDA graph capture. The launcher
serves `$HIBANA_RESULTS/w3-checkpoints/<target>-current` and runs `fetch` on every serve for a target with a
published checkpoint, which keeps that link on the pinned revision (a local calibration stays; see `fetch`):

    python -m hibana.weights.w3_checkpoint fetch --target flashnext --root $HIBANA_RESULTS/w3-checkpoints
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from hibana.model.qwen38_config import MODEL_REVISION
from hibana.weights import files
from hibana.weights.sharding import REPLICATED, Shard

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

    def paths(self, names) -> list[tuple[Path, str]]:
        """The files of `names` and their recorded checksums."""
        return [(self.directory / self.weights[name]["file"], self.weights[name]["sha256"]) for name in names]

    def ahead(self, names) -> None:
        """Start reading (and checking) the files of `names` on the read pool (`weights/files.py`)."""
        for path, digest in self.paths(names):
            files.ahead(path, digest)

    @staticmethod
    def expert_names(layer: int) -> list[str]:
        return [f"layers.{layer}.experts.w13", f"layers.{layer}.experts.w2"]

    def load(self, name: str, linear, shard: Shard = REPLICATED):
        """`linear`'s copy: its tensor-parallel part `shard` of the checkpoint's whole matrix, in `linear`'s row order
        (an MLP's gate|up interleaved, `modelopt.interleave_pairs`)."""
        weight = self._load(name, linear.out_features, linear.in_features, linear.weight.device, shard)
        if getattr(linear, "pairs", 0):
            from hibana.kernels.w3 import interleave_rows
            weight = interleave_rows(weight, linear.pairs)
        return weight

    def load_many(self, items, window: int = 8):
        """`load` for each (name, linear, shard) in order, reading `window` files ahead."""
        items = list(items)
        try:
            for i, (name, linear, shard) in enumerate(items):
                self.ahead(name for name, _, _ in items[i:i + window])
                yield self.load(name, linear, shard)
        except BaseException:
            files.discard()
            raise

    def expert_layers(self) -> set[int]:
        """The layers whose routed experts the checkpoint carries (`layers.N.experts.w13` / `.w2`, kernels/w3_moe.py)."""
        return {int(name.split(".")[1]) for name in self.weights if name.startswith("layers.") and ".experts." in name}

    def load_experts(self, layer: int, experts: int, inter: int, hidden: int, device):
        """Layer `layer`'s routed experts: fc1 (rows in w3_moe.fc1_order) and fc2 stacked over the experts, whole
        experts; under tensor parallelism this rank's range of them (`sharding.ExpertParallel`)."""
        from hibana.kernels.w3_moe import W3Experts
        from hibana.weights.sharding import ExpertParallel, expert_range
        lo, hi = expert_range(experts)
        w13, w2 = self.expert_names(layer)
        fc1 = self._load(w13, (hi - lo) * 2 * inter, hidden, device, ExpertParallel(experts, 2 * inter))
        fc2 = self._load(w2, (hi - lo) * hidden, inter, device, ExpertParallel(experts, hidden))
        return W3Experts.from_stacked(fc1, fc2, hi - lo, base=lo)

    def _load(self, name: str, out_features: int, in_features: int, device, shard: Shard = REPLICATED):
        from hibana.kernels.w3 import W3Weight, _codebook, lut_of
        entry = self.weights[name]
        n, k = entry["out_features"], entry["in_features"]
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("calibrated W3 files must be directly inside the checkpoint")
        # Read once, checked against the manifest's checksum as it is read (`weights/files.py`).
        data = files.take(self.directory / filename, entry["sha256"])
        raw = files.as_tensor(data)
        tensors = {key: files.tensor(raw, value) for key, value in files.layout(data).items()}
        expected = {"planes": ((n // 32, k // 128, 3, 32, 4), torch.uint32),
                    "scales": ((n // 32, k // 128, 4, 32), torch.uint8),
                    "row_scale": ((n,), torch.float32)}
        if tensors.keys() != expected.keys() or any(tuple(tensors[key].shape) != shape or tensors[key].dtype != dtype
                                                     for key, (shape, dtype) in expected.items()):
            raise ValueError(f"invalid calibrated W3 tensors for {name}")
        levels = _codebook(entry["codebook"])
        # The fragment tensors index 32 rows by dim 0 and a 128-wide K chunk by dim 1; row scales follow the rows.
        planes, scales = (shard.cols(shard.rows(tensors[key], 32), dim=1) for key in ("planes", "scales"))
        row_scale = shard.rows(tensors["row_scale"])
        n, k = row_scale.shape[0], planes.shape[1] * 128
        if (n, k) != (out_features, in_features):
            raise ValueError(f"calibrated W3 shape mismatch for {name}")
        return W3Weight(*(t.contiguous().to(device) for t in (planes, scales, row_scale)), n, k, levels, lut_of(levels))


def fetched_name(target, revision: str) -> str:
    """The directory `fetch` downloads `revision` of `target`'s published copies into."""
    return f"{target.name}-{revision[:12]}"


def is_fetched(target, name: str) -> bool:
    """Whether a directory name is one `fetch` made (any revision), not a local calibration."""
    prefix = f"{target.name}-"
    rest = name[len(prefix):] if name.startswith(prefix) else ""
    return len(rest) == 12 and all(c in "0123456789abcdef" for c in rest)


def fetch(target, root: str | Path, log=print, switch: bool = False) -> Path:
    """Keep root/<target>-current on `target`'s published copies (`w3_repo` at the pinned `w3_revision`).

    - No link (or a dangling one), or a link to an earlier revision this function fetched: download the pinned
      revision into root/<target>-<revision> (verified against its manifest) and point the link at it.
    - A link to anything else is a local calibration the user chose: it stays (unless `switch`), and a note says
      that the published copies exist and how to switch.
    Runs on every serve; with the link already on the pinned revision it downloads nothing.
    Returns the directory the link points at afterwards."""
    if not target.w3_repo:
        raise LookupError(f"{target.name} has no published 3-bit checkpoint; calibrate one (docs/w3-calibration.md)")
    root = Path(root)
    dest = root / fetched_name(target, target.w3_revision)
    link = root / f"{target.name}-current"
    current = os.readlink(link) if link.is_symlink() else None
    if current is not None and link.exists() and not is_fetched(target, Path(current).name) and not switch:
        if Path(current).name != dest.name:
            log(f"note: {link} is a local calibration ({current}); the published {target.w3_repo}@{target.w3_revision[:12]} "
                f"is not served. To switch: python -m hibana.weights.w3_checkpoint fetch --target {target.name} --root {root} --switch")
        return link.resolve()
    if not (dest / "manifest.json").is_file():
        from huggingface_hub import snapshot_download
        partial = dest.with_name(dest.name + ".partial")
        log(f"fetching {target.w3_repo}@{target.w3_revision[:12]} into {dest}")
        snapshot_download(target.w3_repo, revision=target.w3_revision, local_dir=partial,
                          allow_patterns=["manifest.json", "*.safetensors"])
        calibrated = CalibratedWeights(partial, revision=target.model_revision)
        for name, entry in calibrated.weights.items():
            if sha256(partial / entry["file"]) != entry["sha256"]:
                raise ValueError(f"fetched W3 checksum mismatch for {name}")
        partial.rename(dest)
    if current != dest.name:
        if current is not None and link.exists():
            log(f"{link}: {current} -> {dest.name} (the pinned revision; the old directory is kept)")
        temporary = root / f".{link.name}.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(dest.name)
        temporary.replace(link)
    return dest


if __name__ == "__main__":
    import argparse
    from hibana.target import TARGETS
    ap = argparse.ArgumentParser(prog="python3 -m hibana.weights.w3_checkpoint")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the target's published calibrated copies")
    f.add_argument("--target", choices=sorted(TARGETS), required=True)
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--switch", action="store_true", help="also replace a link to a local calibration with the published copies")
    args = ap.parse_args()
    try:
        print(fetch(TARGETS[args.target], args.root, log=lambda message: print(message, file=sys.stderr), switch=args.switch))
    except LookupError as error:
        raise SystemExit(str(error))
