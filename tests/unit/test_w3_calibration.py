import json
from types import SimpleNamespace

import pytest
import torch

from hibana.distributed import set_tp_info
from hibana.kernels import w3
from hibana.model.qwen38_config import MODEL_REVISION
from hibana.weights.w3_calibration import quantize
from hibana.weights.sharding import ROW_PARALLEL, ColumnParallel
from hibana.weights.w3_checkpoint import CalibratedWeights, save_weight


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



def test_checkpoint_round_trip_and_identity_checks(tmp_path):
    weight = w3.pack(torch.randn(128, 128))
    entry = save_weight(tmp_path / 'layer.safetensors', weight)
    manifest = {'format': 1, 'target_revision': MODEL_REVISION, 'weights': {'layer': entry}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    linear = SimpleNamespace(weight=torch.zeros(128, 128), out_features=128, in_features=128)  # Flash-Next's: no shard
    loaded = CalibratedWeights(tmp_path).load('layer', linear)
    assert torch.equal(w3.dequantize(loaded), w3.dequantize(weight))
    (tmp_path / 'layer.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        CalibratedWeights(tmp_path).load('layer', linear)
    manifest['target_revision'] = 'different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='revision'):
        CalibratedWeights(tmp_path)


@pytest.mark.parametrize('rank', [0, 1])
def test_checkpoint_loads_a_tensor_parallel_part(tmp_path, rank):
    """A rank's copy is exactly its part of the whole matrix: rows of each fused block, or a K slice."""
    weight = w3.pack(torch.randn(256, 512))
    entry = save_weight(tmp_path / 'layer.safetensors', weight)
    (tmp_path / 'manifest.json').write_text(json.dumps({'format': 1, 'target_revision': MODEL_REVISION, 'weights': {'layer': entry}}))
    full = w3.dequantize(weight)
    column = torch.cat([full[64 * rank:64 * (rank + 1)], full[128 + 64 * rank:128 + 64 * (rank + 1)]])
    set_tp_info(rank, 2)
    try:
        for shard, part in ((ColumnParallel((64, 64)), column), (ROW_PARALLEL, full[:, 256 * rank:256 * (rank + 1)])):
            linear = SimpleNamespace(weight=torch.zeros(0), out_features=part.shape[0], in_features=part.shape[1])
            assert torch.equal(w3.dequantize(CalibratedWeights(tmp_path).load('layer', linear, shard)), part)
    finally:
        set_tp_info(0, 1)


def test_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match='energy'):
        quantize(torch.zeros(128, 128), torch.zeros(128, 128), capture=False)
    with pytest.raises(ValueError, match='finite'):
        quantize(torch.full((128, 128), float('nan')), torch.eye(128), capture=False)
