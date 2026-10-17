"""One linear interface over every weight format."""
import torch

from sharar.weights.linear import Linear


def _formats():
    from sharar.model.draft_linear import DraftFp8Linear
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.weights.modelopt import Fp8Linear, Nvfp4Linear, swizzle_block_scales
    torch.manual_seed(0)
    bf16 = torch.randn(32, 64).to(torch.bfloat16)
    fp8 = Fp8Linear(weight=torch.randn(32, 64).to(torch.float8_e4m3fn), weight_scale=torch.tensor(0.5), input_scale=torch.tensor(0.25))
    packed = torch.randint(0, 256, (128, 32), dtype=torch.uint8)
    scales = torch.full((128, 4), 1.0).to(torch.float8_e4m3fn)
    nvfp4 = Nvfp4Linear(weight=packed, weight_scale_swizzled=swizzle_block_scales(scales).view(torch.float8_e4m3fn),
                        alpha=torch.tensor(2.0), input_scale_inv=torch.tensor(4.0), out_features=128, in_features=64)
    return {"fp8": fp8, "nvfp4": nvfp4, "trunk-fp8": TrunkLinear.from_bf16(bf16, "fp8"), "trunk-bf16": TrunkLinear.from_bf16(bf16, "bf16"),
            "draft": DraftFp8Linear.from_bf16(bf16)}


def test_every_format_implements_the_interface():
    for name, lin in _formats().items():
        assert isinstance(lin, Linear), name
        w = lin.dequantize()
        assert w.dtype == torch.float32 and w.shape == (lin.out_features, lin.in_features), name
        assert lin.bytes > 0 and lin.w3 is None, name


def test_dequantize_matches_the_packed_sources():
    f = _formats()
    assert torch.equal(f["fp8"].dequantize(), f["fp8"].weight.float() * f["fp8"].weight_scale.float())
    assert torch.equal(f["draft"].dequantize(), f["draft"].weight.float() * f["draft"].scale.float())
    assert torch.equal(f["trunk-bf16"].dequantize(), f["trunk-bf16"].weight.float())
    from sharar.weights.modelopt import unpack_fp4
    n = f["nvfp4"]
    ref = unpack_fp4(n.weight, torch.ones(128, 4)) * (n.alpha * n.input_scale_inv).float()
    assert torch.equal(n.dequantize(), ref)
