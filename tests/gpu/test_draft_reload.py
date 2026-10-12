"""A reload of the served checkpoint must reproduce the constructor's tensors bit for bit (both precisions)."""
import json
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GB10 runtime")


def _bits(model):
    out = {name: weight.detach().clone() for name, weight in model.weights.items()}
    for name, linear in model.fp8_weights.items():
        out[name] = (linear.weight.view(torch.uint8).clone(), linear.scale.clone())
        if linear.w3 is not None:
            out[name + '.w3'] = tuple(getattr(linear.w3, key).clone() for key in ('planes', 'scales', 'row_scale'))
    return out


def _pointers(model):
    pointers = {name: weight.data_ptr() for name, weight in model.weights.items()}
    for name, linear in model.fp8_weights.items():
        pointers[name] = linear.weight.data_ptr()
        if linear.w3 is not None:
            for key in ('planes', 'scales', 'row_scale'):
                pointers[name + '.w3.' + key] = getattr(linear.w3, key).data_ptr()
    return pointers


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_reload_of_the_same_snapshot_is_bit_identical(precision):
    from sharar.model.parallel_draft import ParallelDraftModel
    from sharar.proposers.factory import draft_snapshot
    snapshot = draft_snapshot("dflash2")
    model = ParallelDraftModel(snapshot, SimpleNamespace(device=torch.device("cuda"), embed=torch.empty(1, 5120)), 4096, precision=precision)
    before = _bits(model)
    pointers = _pointers(model)
    # perturb a trainable weight so the reload has something to restore
    name = "layers.0.self_attn.q_proj.weight"
    (model.fp8_weights[name].weight.view(torch.uint8) if precision == "fp8" else model.weights[name]).zero_()
    if precision == 'fp8' and model.fp8_weights[name].w3 is not None:
        model.fp8_weights[name].w3.planes.zero_()
    model.weights["norm.weight"].add_(1)
    model.load_weights(snapshot)
    after = _bits(model)
    assert _pointers(model) == pointers
    for key in before:
        if isinstance(before[key], tuple):
            assert len(before[key]) == len(after[key]) and all(torch.equal(a, b) for a, b in zip(before[key], after[key])), key
        else:
            assert torch.equal(before[key], after[key]), key
    assert model.source == str(snapshot)
    del model
    torch.cuda.empty_cache()
