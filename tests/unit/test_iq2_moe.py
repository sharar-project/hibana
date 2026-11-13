"""Lossless expert packing and shared-source cache contracts, without a GPU."""
import os
import pytest
import torch
from hibana.kernels import iq2_moe


def blocks(quant, e=3, h=256, intermediate=512):
    size = 74 if quant == "IQ2_XS" else 66
    generator = torch.Generator().manual_seed(208)
    def one(n, k):
        raw = torch.randint(0, 256, (e, n, k // 256, size), dtype=torch.uint8, generator=generator)
        base = (torch.randn(e, n, k // 256, 1, generator=generator) / 1024).half()
        raw[..., :2] = base.view(torch.uint8)
        return raw
    return one(intermediate, h), one(h, intermediate)


@pytest.mark.parametrize("quant", iq2_moe.FORMATS_IQ2)
def test_expert_packing_preserves_all_native_bytes_and_owns_storage(quant):
    up, down = blocks(quant)
    saved = up.clone(), down.clone()
    experts = iq2_moe.pack_experts(up, down, quant, base=17)
    assert experts.bytes == up.numel() + down.numel()
    assert (experts.hidden, experts.intermediate, experts.experts, experts.base) == (256, 512, 3, 17)
    for e in range(3):
        for is_up, expected in [(True, saved[0][e]), (False, saved[1][e])]:
            matrix = experts.matrix(e, is_up)
            assert torch.equal(matrix.blocks(), expected)
            assert torch._C._overlaps(matrix.values, experts.up_values if is_up else experts.down_values)
    up.zero_(); down.zero_()
    assert torch.equal(experts.matrix(1, True).blocks(), saved[0][1])
    assert torch.equal(experts.to("cpu").matrix(2, False).blocks(), saved[1][2])


@pytest.mark.parametrize("quant", iq2_moe.FORMATS_IQ2)
def test_strided_expert_axis_and_private_workspace(quant):
    up, down = blocks(quant, e=6)
    experts = iq2_moe.pack_experts(up[::2], down[::2], quant)
    assert torch.equal(experts.matrix(1, True).blocks(), up[2])
    ws = iq2_moe.make_workspace(experts, 64, 9, tile_rows=16)
    assert ws.work.shape == (3 + 36, 3)
    assert ws.hq.shape == (576, 512) and ws.pair_outputs.shape == (576, 256)
    assert all(not torch._C._overlaps(a, b) for i, a in enumerate(ws.tensors()) for b in ws.tensors()[:i])


def test_packing_rejects_mismatched_geometry_and_nonfinite_bases():
    up, down = blocks("IQ2_XXS")
    for kwargs in [dict(quant="Q8_0"), dict(base=-1), dict(base=2**31 - 1)]:
        with pytest.raises(ValueError):
            iq2_moe.pack_experts(up, down, **kwargs)
    with pytest.raises(ValueError, match="geometry"):
        iq2_moe.pack_experts(up, down[:2])
    up[0, 0, 0, :2] = torch.tensor([float("inf")], dtype=torch.float16).view(torch.uint8)
    with pytest.raises(ValueError, match="finite"):
        iq2_moe.pack_experts(up, down)


def test_jit_cache_includes_shared_codec_bytes(tmp_path, monkeypatch):
    backend = iq2_moe
    from torch.utils import cpp_extension
    source, header = tmp_path / "kernel.cu", tmp_path / "codec.cuh"
    source.write_bytes(b"unchanged kernel")
    monkeypatch.setattr(backend, "_SRC", source)
    monkeypatch.setattr(backend, "_CODEC", header)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.1a")
    loaded, cache = [], {}
    monkeypatch.setattr(cpp_extension, "load", lambda *, name, **kwargs: cache.setdefault(name, object()))
    try:
        for code in (b"original", b"modified", b"original"):
            header.write_bytes(code); os.utime(header, (1000, 1000))
            backend._module.cache_clear()
            loaded.append(backend._module())
        assert loaded[0] is loaded[2] and loaded[0] is not loaded[1]
    finally:
        backend._module.cache_clear()
