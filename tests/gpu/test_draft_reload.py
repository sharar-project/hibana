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
    return out


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_reload_of_the_same_snapshot_is_bit_identical(precision):
    from sharar.model.parallel_draft import ParallelDraftModel
    from sharar.proposers.factory import draft_snapshot
    snapshot = draft_snapshot("dflash2")
    model = ParallelDraftModel(snapshot, SimpleNamespace(device=torch.device("cuda")), 4096, precision=precision)
    before = _bits(model)
    pointers = {n: w.data_ptr() for n, w in model.weights.items()} | {n: l.weight.data_ptr() for n, l in model.fp8_weights.items()}
    # perturb a trainable weight so the reload has something to restore
    name = "layers.0.self_attn.q_proj.weight"
    (model.fp8_weights[name].weight.view(torch.uint8) if precision == "fp8" else model.weights[name]).zero_()
    model.weights["norm.weight"].add_(1)
    model.load_weights(snapshot)
    after = _bits(model)
    assert {n: w.data_ptr() for n, w in model.weights.items()} | {n: l.weight.data_ptr() for n, l in model.fp8_weights.items()} == pointers
    for key in before:
        if isinstance(before[key], tuple):
            assert torch.equal(before[key][0], after[key][0]) and torch.equal(before[key][1], after[key][1]), key
        else:
            assert torch.equal(before[key], after[key]), key
    assert model.source == str(snapshot)
    del model
    torch.cuda.empty_cache()
