"""Startup dispatch, memory accounting and calibration resume without a GPU backend."""
import json
import sys
from types import SimpleNamespace

import pytest

from hibana.cache.spec_budget import verify_bytes
from hibana.engine.contracts import ProposerContext
from hibana.target import NEMOTRON
from tests.ngram_proposer import NGramProposer
from tests.unit.test_runtime_safety import tiny_executor


def test_reconstruction_uses_the_recurrences_constructor(monkeypatch):
    ex = tiny_executor(monkeypatch)
    ex.target = NEMOTRON
    calls = []

    class AcceptedState:
        def __init__(self, pool, batch, steps, key_heads):
            self.layer_inputs = {}
            calls.append((pool, batch, steps, key_heads))

    monkeypatch.setitem(sys.modules, "hibana.kernels.mamba2", SimpleNamespace(AcceptedState=AcceptedState))
    proposer = NGramProposer(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3)
    ex.set_proposer(proposer, state_strategy="reconstruct")
    assert calls == [(ex.state_pool, ex.max_batch, 4, 8)]
    assert isinstance(ex.spec.replay, AcceptedState)


def test_verify_budget_includes_both_mamba_groups_and_shared_positions():
    # Two requests, four verified inputs, forty layers: BF16 x|B|C, FP32 dt/dA,
    # one shared int32 position per input, plus BF16 three-token convolution windows.
    assert verify_bytes(2, 4, "reconstruct", NEMOTRON) == 26_542_112
    assert verify_bytes(2, 4, "snapshots", NEMOTRON) == 1_361_838_080


@pytest.mark.parametrize("kv", ["nvfp4", "sim-fp4", "sim-k8v4", "sim-bf16"])
def test_executor_refuses_unsupported_kv_before_allocating(kv):
    from hibana.engine import config
    from hibana.engine.executor import Executor
    with config.using(config.current().replace(kv_format=kv)):
        with pytest.raises(ValueError, match="supports HIBANA_KV_FORMAT=fp8"):
            Executor(SimpleNamespace(target=NEMOTRON))


def test_stop_ids_resolve_from_the_executor_target(monkeypatch):
    from hibana.engine.sampler import SamplingParams
    ex = tiny_executor(monkeypatch)
    ex.target = NEMOTRON
    params = SamplingParams()
    req = ex.admit([1000], params)
    assert req.params.stop_token_ids == (11, 2) and params.stop_token_ids is None
    explicit = SamplingParams(stop_token_ids=())
    assert ex.admit([1000], explicit).params.stop_token_ids == ()


@pytest.mark.parametrize("changed", ["capture", "source", "damping", "codebooks", "chunk", "min_tokens", "prior_tokens", "revision", "method"])
def test_expert_resume_rejects_other_inputs_before_changing_files(tmp_path, changed):
    from hibana.weights.nemotron_w3_experts import check_progress, fit_identity
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"inputs_sha256": "inputs", "rows": {"mtp": 12}}))
    args = SimpleNamespace(activations=tmp_path, codebooks="lloyd,zero8", chunk=16, damping=.01,
                           min_tokens=16, prior_tokens=256)
    identity = fit_identity(args, "revision", "source")
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"identity": identity, "weights": {"done": {"sha256": "checked later"}}}))
    original = progress.read_bytes()
    assert check_progress(progress, identity) == {"done": {"sha256": "checked later"}}
    if changed == "capture":
        capture.write_text(json.dumps({"inputs_sha256": "other inputs", "rows": {"mtp": 12}}))
        candidate = fit_identity(args, "revision", "source")
    elif changed in ("source", "revision"):
        candidate = fit_identity(args, "other" if changed == "revision" else "revision",
                                 "other" if changed == "source" else "source")
    elif changed == "method":
        candidate = {**identity, "method": "cheap"}
    else:
        setattr(args, changed, "lloyd" if changed == "codebooks" else getattr(args, changed) * 2)
        candidate = fit_identity(args, "revision", "source")
    with pytest.raises(SystemExit, match="another fit"):
        check_progress(progress, candidate)
    assert progress.read_bytes() == original
