"""Nemotron's calibrated 3-bit copies: the names the fit writes are the model's, the capture sees what the projection
reads, a fitted file loads onto an `Fp8Linear` through `Nemotron.attach_calibrated`, and `ops.fp8_linear` reads the
copy for at most 64 rows and the FP8 weights above (no model load; small synthetic projections)."""

import json
from types import SimpleNamespace

import torch

from sharar.engine import config
from sharar.kernels import ops, w3
from sharar.kernels.fp8_quant import quant_fp8
from sharar.target import NEMOTRON
from sharar.weights.modelopt import Fp8Linear
from sharar.weights.nemotron_w3_calibrate import errors, projection_names, source_prefixes
from sharar.weights.w3_calibrate import InputCovariances
from sharar.weights.w3_calibration import quantize
from sharar.weights.w3_checkpoint import FORMAT_VERSION, CalibratedWeights, save_weight


def _fp8_linear(n: int, k: int, seed: int) -> Fp8Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    scale = w.abs().max() / 448
    return Fp8Linear(weight=(w / scale).to(torch.float8_e4m3fn), weight_scale=scale.float().reshape(()),
                     input_scale=torch.tensor(0.02, device="cuda"))


def test_names_cover_the_model_projections():
    names = projection_names(NEMOTRON)
    assert len(names) == 2 * 40 + 2 * 8 + 1 and len(set(names)) == len(names)
    assert source_prefixes("layers.0.mixer.in_proj") == ["language_model.backbone.layers.0.mixer.in_proj"]
    assert [p.rsplit(".", 1)[1] for p in source_prefixes("layers.7.mixer.qkv")] == ["q_proj", "k_proj", "v_proj"]
    assert source_prefixes("lm_head") == ["language_model.lm_head"]
    for name in names[:-1]:
        kind, layer = name.rsplit(".", 1)[1], int(name.split(".")[1])
        assert NEMOTRON.layer_types[layer] == ("linear_attention" if kind in ("in_proj", "out_proj") else "full_attention")


@torch.inference_mode()
def test_capture_sees_the_served_fp8_input():
    lin = _fp8_linear(256, 512, 1)
    x = torch.randn(40, 512, device="cuda", dtype=torch.bfloat16)
    original = ops.fp8_linear
    with InputCovariances([("p", lin)], served_fp8=True) as cap:
        ops.fp8_linear(x, lin)  # BF16 in: quantized at the static scale, as the copy will read it
        ops.fp8_linear(None, lin, x_q=quant_fp8(x, lin.input_scale))
        cap.holdout = True  # a held-out sample: 64 evenly spaced probe rows, no covariance
        ops.fp8_linear(x[:20], lin)
    assert ops.fp8_linear is original
    seen = quant_fp8(x, lin.input_scale).float() * lin.input_scale
    torch.testing.assert_close(cap.matrices["p"], 2 * seen.T @ seen, rtol=1e-5, atol=1e-3)
    assert cap.counts["p"] == 80 and torch.equal(torch.cat(cap.held["p"]), seen[:20].bfloat16())


@torch.inference_mode()
def test_fitted_copy_loads_and_serves_decode_only(tmp_path):
    n, k = 512, 1024
    lin = _fp8_linear(n, k, 2)
    x = torch.randn(4096, k, device="cuda") * torch.linspace(0.2, 3, k, device="cuda")  # unequal channel energy
    h = x.T @ x / len(x)
    packed, _ = quantize(lin.dequantize(), h)
    weight = lin.dequantize()
    gptq, nearest = errors(weight, w3.dequantize(packed), h, x[:64]), errors(weight, w3.dequantize(w3.pack(weight)), h, x[:64])
    assert gptq["output_cov"] < nearest["output_cov"]
    entry = save_weight(tmp_path / "layers.0.mixer.out_proj.safetensors", packed)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": FORMAT_VERSION, "target_revision": NEMOTRON.model_revision,
                                                        "weights": {"layers.0.mixer.out_proj": entry}}))
    other = _fp8_linear(256, 512, 3)
    model = SimpleNamespace(target=NEMOTRON, linears=lambda: iter([("layers.0.mixer.out_proj", lin), ("lm_head", other)]))
    from sharar.model.nemotron.model import Nemotron
    identity = Nemotron.attach_calibrated(model, CalibratedWeights(tmp_path, revision=NEMOTRON.model_revision), log=None)
    assert len(identity) == 64 and other.w3 is None
    assert torch.equal(lin.w3.planes, packed.planes) and torch.equal(lin.w3.row_scale, packed.row_scale)

    xs = torch.randn(80, k, device="cuda", dtype=torch.bfloat16)
    x_q = quant_fp8(xs, lin.input_scale)
    copy, lin.w3 = lin.w3, None
    fp8_wide = ops.fp8_linear(None, lin, x_q=x_q)
    lin.w3 = copy
    with config.using(config.current().replace(w3_decode=True)):
        for m in (1, 4, 16, 64):
            y = ops.fp8_linear(None, lin, x_q=x_q[:m])
            ref = (x_q[:m].float() * lin.input_scale) @ w3.dequantize(packed).T
            assert float((y.float() - ref).norm() / ref.norm()) < 4e-3
        assert torch.equal(ops.fp8_linear(None, lin, x_q=x_q), fp8_wide)  # 80 rows: prefill keeps FP8
    with config.using(config.current().replace(w3_decode=False)):
        y = ops.fp8_linear(None, lin, x_q=x_q[:4])
        lin.w3 = None
        assert torch.equal(y, ops.fp8_linear(None, lin, x_q=x_q[:4]))  # the copy is inert without SHARAR_W3_DECODE
