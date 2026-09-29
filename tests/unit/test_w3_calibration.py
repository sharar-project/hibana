import json
from types import SimpleNamespace

import pytest
import torch

from sharar.kernels import w3
from sharar.model.qwen38_config import MODEL_REVISION
from sharar.weights.w3_calibration import quantize
from sharar.weights.w3_checkpoint import CalibratedWeights, save_weight


def test_calibrated_codes_round_trip_without_requantization():
    q = (torch.arange(128 * 256).reshape(128, 256) % 6 + 1).int()  # deliberately omit extreme codes
    scales = torch.full((128, 8), 2., dtype=torch.float8_e4m3fn)
    row_scale = torch.arange(1, 129).float() / 128
    packed = w3.pack_codes(q, scales, row_scale)
    assert torch.equal(w3.codes(packed), q)
    assert torch.equal(w3.group_scales(packed), scales.float() * row_scale[:, None])


def test_gptq_reduces_error_on_correlated_inputs():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(1024, 128, generator=generator)
    x[:, 1::2] = x[:, ::2] * .9 + x[:, 1::2] * .1
    weight = torch.randn(128, 128, generator=generator) * .1
    covariance = x.T @ x / len(x)
    packed, report = quantize(weight, covariance, capture=False)
    original_error = (x @ (w3.dequantize(w3.pack(weight)) - weight).T).square().mean()
    calibrated_error = (x @ (w3.dequantize(packed) - weight).T).square().mean()
    assert calibrated_error < original_error * .9
    assert report['bytes'] == packed.bytes



def test_unbias_restores_the_row_gain_without_changing_codes():
    """The MSE-optimal codes shrink each row; `unbias` scales it back through the row scale only (#61)."""
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(1024, 256, generator=generator)
    x[:, 1::2] = x[:, ::2] * .9 + x[:, 1::2] * .1
    weight = torch.randn(128, 256, generator=generator) * .1
    covariance = x.T @ x / len(x)
    plain, _ = quantize(weight, covariance, capture=False)
    shrink = ((w3.dequantize(plain) * weight).sum(1) / weight.square().sum(1)).mean()
    assert shrink < .99
    for mode, gram in (("weight", torch.eye(256)), ("output", covariance)):
        packed, report = quantize(weight, covariance, capture=False, unbias=mode)
        assert torch.equal(packed.planes, plain.planes) and torch.equal(packed.scales, plain.scales)
        projected = weight @ gram
        before = (w3.dequantize(plain) * projected).sum(1) / (weight * projected).sum(1)
        gain = (w3.dequantize(packed) * projected).sum(1) / (weight * projected).sum(1)
        shrunk = before < 1  # rows that were not shrunk keep their scale (the correction never shrinks)
        assert shrunk.float().mean() > .5
        assert torch.allclose(gain[shrunk], torch.ones_like(gain[shrunk]), atol=1e-4)
        assert torch.equal(gain[~shrunk], before[~shrunk])
        assert report["unbias"] == mode and report["gain_mean"] > 1
    with pytest.raises(ValueError, match="unbias"):
        quantize(weight, covariance, capture=False, unbias="rows")

def test_checkpoint_round_trip_and_identity_checks(tmp_path):
    weight = w3.pack(torch.randn(128, 128))
    entry = save_weight(tmp_path / 'layer.safetensors', weight)
    manifest = {'format': 1, 'target_revision': MODEL_REVISION, 'weights': {'layer': entry}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    linear = SimpleNamespace(weight=torch.zeros(128, 128), out_features=128)
    loaded = CalibratedWeights(tmp_path).load('layer', linear)
    assert torch.equal(w3.dequantize(loaded), w3.dequantize(weight))
    (tmp_path / 'layer.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        CalibratedWeights(tmp_path).load('layer', linear)
    manifest['target_revision'] = 'different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='revision'):
        CalibratedWeights(tmp_path)


def test_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match='energy'):
        quantize(torch.zeros(128, 128), torch.zeros(128, 128), capture=False)
    with pytest.raises(ValueError, match='finite'):
        quantize(torch.full((128, 128), float('nan')), torch.eye(128), capture=False)
