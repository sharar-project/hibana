"""Packed native GGML GEMM: exact weight recovery, FP8 compute and graph replay."""
import pytest
import torch

from sharar.kernels import ggml, w3
from sharar.weights.ggml import FORMATS, GgmlLinear, dequantize


def blocks(quant, n, k):
    block, size = FORMATS[quant]
    generator = torch.Generator().manual_seed(81)
    raw = torch.randint(0, 256, (n, k // block, size), dtype=torch.uint8, generator=generator)
    bases = torch.randn((n, k // block, 1), generator=generator).half() / 128
    bases[::7] = 0
    raw[..., :2] = bases.contiguous().view(torch.uint8)
    return raw


def assert_close(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
    assert error < 0.004, error.item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.03125)


@pytest.mark.parametrize("quant", FORMATS)
@pytest.mark.parametrize("m", [1, 3, 16, 17, 33, 64, 65, 129])
@pytest.mark.parametrize("n", [8, 136])
@pytest.mark.parametrize("tile_n", [32, 128])
def test_fp8_compute_matches_exact_decoded_weights(quant, m, n, tile_n):
    k = 288 if quant == "Q8_0" else 768
    raw = blocks(quant, n, k)
    weight = ggml.pack_weight(raw.cuda(), quant)
    x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(m, k // 32, device="cuda") + 0.1
    source = x.float() * scales.repeat_interleave(32, 1)
    expected = (source @ dequantize(raw, quant).cuda().T).bfloat16()
    assert_close(ggml.linear(x, weight, scales, tile_n=tile_n), expected)


@pytest.mark.parametrize("quant", FORMATS)
@pytest.mark.parametrize("scale_mode", ["scalar", "row", "group", "bf16"])
@pytest.mark.parametrize("tile_n", [32, 128])
def test_preallocated_splitk_and_graph_replay(quant, scale_mode, tile_n):
    m, n, k = 17, 136, 768
    raw = blocks(quant, n, k)
    weight = GgmlLinear(raw.cuda(), quant).prepare()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    if scale_mode == "bf16":
        scales = None
        q, s = w3.quant_groups(x)
        reference_x = q.float() * s.repeat_interleave(32, 1)
    else:
        x = x.to(torch.float8_e4m3fn)
        shape = () if scale_mode == "scalar" else (m,) if scale_mode == "row" else (m, k // 32)
        scales = torch.rand(shape, device="cuda") + 0.1
        reference_x = x.float() * (scales if scale_mode == "scalar" else scales[:, None]
                                   if scale_mode == "row" else scales.repeat_interleave(32, 1))
    expected = (reference_x @ dequantize(raw, quant).cuda().T).bfloat16()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(4 * m * n, device="cuda")
    run = lambda: ggml.linear(x, weight, scales, splits=4, tile_n=tile_n, tile_m=16, out=out, workspace=scratch)
    assert run().data_ptr() == out.data_ptr()
    assert_close(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # FP8 negation is not implemented by torch; negate the sign bits instead.
    if x.dtype == torch.float8_e4m3fn:
        x.view(torch.uint8).bitwise_xor_(128)
    else:
        x.neg_()
    out.fill_(float("nan"))
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert_close(out, -expected)


@pytest.mark.parametrize("quant", ["IQ2_XS", "IQ2_XXS", "Q8_0"])
@pytest.mark.parametrize("tile_n", [32, 128])
def test_all_descriptors_are_recovered_exactly(quant, tile_n):
    n = 2048 if quant == "IQ2_XS" else 1024 if quant == "IQ2_XXS" else 8
    raw = blocks(quant, n, 256)
    raw[..., :2] = torch.tensor([0.03125], dtype=torch.float16).view(torch.uint8)
    if quant == "IQ2_XS":
        raw[..., 2:66] = torch.arange(65536).to(torch.uint16).view(torch.uint8).reshape(n, 1, 64)
        raw[..., 66:] = torch.arange(n * 8).to(torch.uint8).reshape(n, 1, 8)
    elif quant == "IQ2_XXS":
        index = torch.arange(n * 32).reshape(n, 1, 8, 4)
        grids, signs = index % 256, (index // 256) % 128
        control = sum(signs[..., j] << (7 * j) for j in range(4)) | ((index[..., 0] // 2048 % 16) << 28)
        grid_word = sum(grids[..., j] << (8 * j) for j in range(4))
        raw[..., 2:] = torch.stack((grid_word, control), -1).int().view(torch.uint8).reshape(n, 1, 64)
    else:
        raw[..., 2:] = torch.arange(n * 256).to(torch.uint8).reshape(n, 8, 32)
    weight = ggml.pack_weight(raw.cuda(), quant)
    x = torch.eye(256, device="cuda").to(torch.float8_e4m3fn)
    actual = ggml.linear(x, weight, torch.ones((), device="cuda"), tile_n=tile_n)
    expected = dequantize(raw, quant).T.contiguous().cuda().bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multidimensional_and_empty_linear():
    weight = ggml.pack_weight(blocks("IQ2_XXS", 8, 256).cuda(), "IQ2_XXS")
    x = torch.randn(2, 3, 256, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(weight(x).reshape(6, 8), weight(x.reshape(6, 256)))
    assert weight(x[:0]).shape == (0, 3, 8)


@pytest.mark.parametrize("quant", FORMATS)
@pytest.mark.parametrize("tile_m", [16, 32, 64])
@pytest.mark.parametrize("tile_n", [32, 128])
def test_row_tiles_cover_multiple_blocks_and_partial_rows(quant, tile_m, tile_n):
    raw = blocks(quant, 136, 768)
    weight = ggml.pack_weight(raw.cuda(), quant)
    x = torch.randn(65, 768, device="cuda").to(torch.float8_e4m3fn)
    expected = (x.float() @ dequantize(raw, quant).cuda().T).bfloat16()
    assert_close(ggml.linear(x, weight, torch.ones((), device="cuda"), tile_m=tile_m, tile_n=tile_n), expected)


def test_rejects_unsafe_buffers_and_scales():
    weight = ggml.pack_weight(blocks("IQ2_XXS", 8, 256).cuda(), "IQ2_XXS")
    x = torch.zeros(2, 256, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.ones((), device="cuda")
    with pytest.raises(ValueError, match="scales"):
        ggml.linear(x, weight)
    with pytest.raises(ValueError, match="overlap"):
        ggml.linear(x, weight, scales, out=x.view(torch.bfloat16).flatten()[:16].view(2, 8))
    with pytest.raises(ValueError, match="workspace"):
        ggml.linear(x, weight, scales, splits=4, workspace=torch.empty(1, device="cuda"))
    with pytest.raises(ValueError, match="aligned"):
        ggml.linear(torch.empty(513, dtype=x.dtype, device="cuda")[1:].view(2, 256), weight, scales)


@pytest.mark.parametrize("m", [1, 17, 65])
@pytest.mark.parametrize("tile_m", [16, 32, 64])
@pytest.mark.parametrize("tile_n", [32, 128])
@pytest.mark.parametrize("splits", [1, 2, 4])
def test_activation_pipeline_partial_rows_and_k_partitions(m, tile_m, tile_n, splits):
    raw = blocks("IQ2_XXS", 136, 768)
    weight = ggml.pack_weight(raw.cuda(), "IQ2_XXS")
    x = torch.randn(m, 768, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(m, 24, device="cuda") + 0.1
    expected = ((x.float() * scales.repeat_interleave(32, 1)) @ dequantize(raw, "IQ2_XXS").cuda().T).bfloat16()
    assert_close(ggml.linear(x, weight, scales, tile_m=tile_m, tile_n=tile_n, splits=splits, stage_a=True), expected)


def test_activation_pipeline_graph_replay():
    weight = ggml.pack_weight(blocks("IQ2_XXS", 136, 768).cuda(), "IQ2_XXS")
    x = torch.randn(17, 768, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(17, 136, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(4 * out.numel(), device="cuda")
    run = lambda: ggml.linear(x, weight, out=out, workspace=workspace, tile_m=16, splits=4, stage_a=True)
    expected = run().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.neg_()
    graph.replay()
    torch.cuda.synchronize()
    assert_close(out, -expected)
