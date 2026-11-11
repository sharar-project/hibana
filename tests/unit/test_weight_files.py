"""Whole-file weight reads (`weights/files.py`): the views equal what safetensors loads, a recorded checksum is
checked, and the NVFP4 experts built from whole files equal the per-tensor stacks they replace."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from hibana.model.flashnext.moe import load_nvfp4_experts
from hibana.weights import files
from hibana.weights.safetensors import SnapshotWeights


def _tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(3, 5, generator=g), "b": torch.randint(0, 255, (7,), generator=g, dtype=torch.uint8),
            "c": torch.randn(4, 2, generator=g).to(torch.bfloat16), "d": torch.tensor(2.5),
            "e": torch.randn(2, 8, generator=g).to(torch.float8_e4m3fn)}


def test_views_equal_what_safetensors_loads(tmp_path):
    path = tmp_path / "x.safetensors"
    save_file(_tensors(), str(path))
    data = files.read(path)
    raw = files.as_tensor(data)
    got = {name: files.tensor(raw, entry) for name, entry in files.layout(data).items()}
    want = load_file(str(path))
    assert got.keys() == want.keys()
    for name in want:
        assert got[name].dtype == want[name].dtype and got[name].shape == want[name].shape
        assert torch.equal(got[name].reshape(-1).view(torch.uint8), want[name].reshape(-1).view(torch.uint8))


def test_checksum_is_checked_ahead_and_on_the_spot(tmp_path):
    path = tmp_path / "x.safetensors"
    save_file(_tensors(), str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert files.read(path, digest) == path.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        files.read(path, "0" * 64)
    files.ahead(path, "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        files.take(path)
    files.ahead(path, digest)
    assert files.take(path) == path.read_bytes()
    files.ahead(path)
    files.discard()
    assert files.take(path, digest) == path.read_bytes()  # read again on the spot


def _experts_snapshot(tmp_path, experts=3, rows=4, cols=8):
    g = torch.Generator().manual_seed(1)
    prefix = "model.language_model.layers.0.mlp"
    shards = {}
    for e in range(experts):
        part = shards.setdefault(f"layer-00000-experts-{e // 2}.safetensors", {})
        for proj in ("gate_proj", "up_proj", "down_proj"):
            p = f"{prefix}.experts.{e}.{proj}"
            part[f"{p}.weight"] = torch.randint(0, 255, (rows, cols // 2), generator=g, dtype=torch.uint8)
            part[f"{p}.weight_scale"] = torch.rand(rows, cols // 16 or 1, generator=g).to(torch.float8_e4m3fn)
            part[f"{p}.weight_scale_2"] = torch.tensor(0.5 + e)
            part[f"{p}.input_scale"] = torch.rand((), generator=g)
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, str(tmp_path / shard))
        weight_map.update(dict.fromkeys(tensors, shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return prefix


def _per_tensor_reference(w, prefix, experts, rows, cols):
    """The loader as it was: one tensor at a time, stacked on the host."""
    p = f"{prefix}.experts"
    out = {k: [] for k in ("w13", "w2", "s13", "s2", "scale2_13", "scale2_2", "in13", "in2")}
    for e in range(experts):
        g, u, d = (f"{p}.{e}.gate_proj", f"{p}.{e}.up_proj", f"{p}.{e}.down_proj")
        out["w13"].append(torch.cat([rows(w.get(f"{u}.weight")), rows(w.get(f"{g}.weight"))], dim=0))
        out["w2"].append(cols(w.get(f"{d}.weight")))
        out["s13"].append(torch.cat([rows(w.get(f"{u}.weight_scale")), rows(w.get(f"{g}.weight_scale"))], dim=0))
        out["s2"].append(cols(w.get(f"{d}.weight_scale")))
        out["scale2_13"].append(w.get(f"{g}.weight_scale_2").float().reshape(()))
        out["scale2_2"].append(w.get(f"{d}.weight_scale_2").float().reshape(()))
        out["in13"].append(torch.maximum(w.get(f"{g}.input_scale").float().reshape(()), w.get(f"{u}.input_scale").float().reshape(())))
        out["in2"].append(w.get(f"{d}.input_scale").float().reshape(()))
    return {k: torch.stack(v) for k, v in out.items()}


@pytest.mark.parametrize("split", [False, True])
def test_whole_file_experts_equal_the_per_tensor_stacks(tmp_path, split):
    prefix = _experts_snapshot(tmp_path)
    rows = (lambda t: t[: t.shape[0] // 2]) if split else (lambda t: t)
    cols = (lambda t: t[:, : t.shape[1] // 2]) if split else (lambda t: t)
    w = SnapshotWeights(tmp_path)
    want = _per_tensor_reference(w, prefix, 3, rows, cols)
    w.ahead([f"{prefix}.experts.0.gate_proj.weight"])
    got = load_nvfp4_experts(w, prefix, 3, "cpu", rows, cols)
    assert got.keys() == want.keys()
    for k in want:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape, k
        assert torch.equal(got[k].reshape(-1).view(torch.uint8), want[k].reshape(-1).view(torch.uint8)), k
