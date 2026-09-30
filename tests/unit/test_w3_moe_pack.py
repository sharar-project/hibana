"""Three-bit routed experts (kernels/w3_moe.py) on the CPU: calibration from NVFP4 experts, fc1's row order (every
up row paired with its gate row inside a 32-row block), the scale unswizzle, and the checkpoint entries."""
import json

import torch

from sharar.kernels import w3, w3_moe
from sharar.model.flashnext.moe import swizzle_block_scales_batched
from sharar.weights.w3_checkpoint import FORMAT_VERSION, CalibratedWeights, save_weight

E, I, H = 2, 128, 256  # small shapes the packing accepts (rows and k multiples of 128, I of 64)


def _nvfp4(rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (E, rows, k // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand(E, rows, k // 16, generator=g) * 0.02 + 0.001).to(torch.float8_e4m3fn)
    return packed, scales


def _layer():
    w13, s13 = _nvfp4(2 * I, H, 0)
    w2, s2 = _nvfp4(H, I, 1)
    return w13, s13, torch.tensor([0.5, 0.25]), w2, s2, torch.tensor([0.125, 2.0])


def test_fc1_order_pairs_up_and_gate_rows():
    order = w3_moe.fc1_order(I)
    assert sorted(order.tolist()) == list(range(2 * I))
    blocks = order.reshape(-1, 32)
    assert torch.equal(blocks[:, 16:], blocks[:, :16] + I)  # gate row of the same intermediate column


def test_calibration_fits_the_nvfp4_experts():
    w13, s13, g13, w2, s2, g2 = _layer()
    ex, report = w3_moe.calibrate(w13, s13, g13, w2, s2, g2, chunk=1)
    _, nearest = w3_moe.calibrate(w13, s13, g13, w2, s2, g2, ratios=(1.0,), chunk=1)
    assert ex.planes13.shape == (E, 2 * I // 32, H // 128, 3, 32, 4) and ex.planes2.shape == (E, H // 32, I // 128, 3, 32, 4)
    inverse = torch.argsort(w3_moe.fc1_order(I))
    for e in range(E):  # the report is the error of what the kernel reads, in the checkpoint's row order
        source = w3_moe.nvfp4_dequant(w13[e:e + 1], s13[e:e + 1], g13[e:e + 1])[0]
        fc1 = w3.dequantize(ex.matrix(e, True))[inverse]
        assert float((fc1 - source).norm() / source.norm()) < 0.25
    assert report["w13"] <= nearest["w13"] and report["w2"] <= nearest["w2"]  # the scale search only helps
    per_weight = 8 * ex.bytes / (E * 3 * I * H)
    assert 3.25 < per_weight < 3.5  # codes and group scales, plus one FP32 multiplier a row


def test_unswizzle_inverts_the_layer_swizzle():
    g = torch.Generator().manual_seed(3)
    s = (torch.rand(E, 2 * I, H // 16, generator=g) + 0.01).to(torch.float8_e4m3fn)
    back = w3_moe.unswizzle_batched(swizzle_block_scales_batched(s), 2 * I, H // 16)
    assert torch.equal(back.view(torch.uint8), s.view(torch.uint8))


def test_expert_entries_round_trip_through_the_checkpoint(tmp_path):
    ex, _ = w3_moe.calibrate(*_layer(), chunk=1)
    weights = {f"layers.3.experts.{key}": save_weight(tmp_path / f"layers.3.experts.{key}.safetensors", ex.stacked(key == "w13"))
               for key in ("w13", "w2")}
    (tmp_path / "manifest.json").write_text(json.dumps({"format": FORMAT_VERSION, "target_revision": "r", "weights": weights}))
    calibrated = CalibratedWeights(tmp_path, revision="r")
    assert calibrated.expert_layers() == {3}
    back = calibrated.load_experts(3, E, I, H, torch.device("cpu"))
    for name in ("planes13", "scales13", "row13", "planes2", "scales2", "row2"):
        assert torch.equal(getattr(back, name), getattr(ex, name))
    assert back.codebook == ex.codebook and back.lut == ex.lut
