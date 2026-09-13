"""Swapping a draft checkpoint into the serving draft without a restart."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import torch
from safetensors.torch import save_file

from sharar.api.server import create_app
from sharar.engine.loop import EngineLoop
from sharar.engine.sampler import InvalidRequest
from sharar.engine.scheduler import Scheduler
from sharar.model.draft_linear import DraftFp8Linear
from sharar.model.parallel_draft import ParallelDraftModel
from tests.unit.test_scheduler import FakeExecutor

CFG = {"architectures": ["DFlash2DraftModel"], "num_hidden_layers": 1, "hidden_size": 8, "sliding_window": 4}


def _checkpoint(directory, tensors, cfg=CFG):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(cfg))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(directory / "model.safetensors"))
    return directory


def _tensors(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g).to(torch.bfloat16)
    return {"fc.weight": r(8, 16), "hidden_norm.weight": r(8), "norm.weight": r(8),
            "layers.0.self_attn.k_proj.weight": r(4, 8), "layers.0.self_attn.v_proj.weight": r(4, 8),
            "layers.0.self_attn.k_norm.weight": r(2), "layers.0.self_attn.q_proj.weight": r(8, 8),
            "layers.0.self_attn.o_proj.weight": r(8, 8), "layers.0.mlp.gate_proj.weight": r(6, 8),
            "layers.0.mlp.up_proj.weight": r(6, 8), "layers.0.mlp.down_proj.weight": r(8, 6),
            "candidate_selector.predecessor_codebook": r(10, 4)}


def _served(snapshot, precision):
    """A model object as the constructor leaves it, without the target or the real dimension gate."""
    model = ParallelDraftModel.__new__(ParallelDraftModel)
    model.cfg, model.precision, model.device, model.source = json.loads((snapshot / "config.json").read_text()), precision, torch.device("cpu"), str(snapshot)
    model.weights, model.fp8_weights = ParallelDraftModel.read_weights(snapshot, model.cfg), {}
    if precision == "fp8":
        for name in list(model.weights):
            if ParallelDraftModel.is_fp8_name(name):
                model.fp8_weights[name] = DraftFp8Linear.from_bf16(model.weights.pop(name))
    return model


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_load_weights_copies_into_the_served_storage(tmp_path, precision):
    base = _tensors(0)
    trained = dict(base)
    for name in ("layers.0.self_attn.q_proj.weight", "layers.0.mlp.gate_proj.weight", "norm.weight", "candidate_selector.predecessor_codebook"):
        trained[name] = _tensors(1)[name]  # context tensors (fc, hidden_norm, k/v, k_norm) stay as trained by default
    model = _served(_checkpoint(tmp_path / "pinned", base), precision)
    pointers = {n: w.data_ptr() for n, w in model.weights.items()} | {n: (l.weight.data_ptr(), l.scale.data_ptr()) for n, l in model.fp8_weights.items()}
    model.load_weights(_checkpoint(tmp_path / "best", trained))
    assert model.source == str(tmp_path / "best")
    assert {n: w.data_ptr() for n, w in model.weights.items()} | {n: (l.weight.data_ptr(), l.scale.data_ptr()) for n, l in model.fp8_weights.items()} == pointers
    expected = _served(tmp_path / "best", precision)
    for name, weight in expected.weights.items():
        assert torch.equal(model.weights[name], weight), name
    for name, linear in expected.fp8_weights.items():
        assert torch.equal(model.fp8_weights[name].weight.view(torch.uint8), linear.weight.view(torch.uint8)) and torch.equal(model.fp8_weights[name].scale, linear.scale), name
    assert torch.equal(model.weights["layers.0.mlp.gate_up"][:6], trained["layers.0.mlp.gate_proj.weight"]) if precision == "bf16" else True


def test_load_weights_refuses_context_shape_and_config_changes(tmp_path):
    base = _tensors(0)
    model = _served(_checkpoint(tmp_path / "pinned", base), "bf16")
    context = dict(base, **{"layers.0.self_attn.v_proj.weight": _tensors(1)["layers.0.self_attn.v_proj.weight"]})
    with pytest.raises(ValueError, match="context weight"):
        model.load_weights(_checkpoint(tmp_path / "context", context))
    reshaped = dict(base, **{"norm.weight": torch.ones(9, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="shape"):
        model.load_weights(_checkpoint(tmp_path / "shape", reshaped))
    with pytest.raises(ValueError, match="sliding_window"):
        model.load_weights(_checkpoint(tmp_path / "cfg", base, cfg=dict(CFG, sliding_window=8)))
    with pytest.raises(ValueError, match="tensors differ"):
        model.load_weights(_checkpoint(tmp_path / "keys", {k: v for k, v in base.items() if k != "norm.weight"}))
    assert model.source == str(tmp_path / "pinned") and torch.equal(model.weights["norm.weight"], base["norm.weight"])


def test_engine_reloads_between_steps_and_reports_invalid_checkpoints():
    async def scenario():
        ex = FakeExecutor()
        calls = []

        def reload_draft(draft_dir):
            calls.append(draft_dir)
            if draft_dir == "/runs/bad/best":
                raise ValueError("draft changes the context weight fc.weight")
            return draft_dir or "pinned-snapshot"
        ex.reload_draft = reload_draft
        engine = EngineLoop(Scheduler(ex))
        engine.start()
        try:
            assert await asyncio.wait_for(engine.reload_draft("/runs/run-1/best"), 2) == "/runs/run-1/best"
            assert await asyncio.wait_for(engine.reload_draft(None), 2) == "pinned-snapshot"
            with pytest.raises(InvalidRequest, match="context weight"):
                await asyncio.wait_for(engine.reload_draft("/runs/bad/best"), 2)
            assert engine.healthy and calls == ["/runs/run-1/best", None, "/runs/bad/best"]
        finally:
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


def test_reload_route_updates_the_served_draft_description():
    async def scenario():
        engine = SimpleNamespace(reload_draft=AsyncMock(side_effect=lambda d: d or "/pinned"), stats=lambda: {}, sched=SimpleNamespace(ex=SimpleNamespace(spec=None)))
        describe = {"speculative": {"algorithm": "dflash2", "draft": "/pinned"}}
        app = create_app(engine, None, None, describe)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            reply = await client.post("/reload_draft", json={"draft": "/runs/run-1/best"})
            assert reply.status_code == 200 and reply.json()["draft"] == "/runs/run-1/best"
            assert describe["speculative"]["draft"] == "/runs/run-1/best"
            assert (await client.get("/get_server_info")).json()["speculative"]["draft"] == "/runs/run-1/best"
            assert (await client.post("/reload_draft", json={"draft": "pinned"})).json()["draft"] == "/pinned"
            assert (await client.post("/reload_draft")).json()["draft"] == "/pinned"
            assert engine.reload_draft.await_args_list[-2].args == (None,)
            assert (await client.post("/reload_draft", json={"draft": 3})).status_code == 400
            engine.reload_draft.side_effect = InvalidRequest("draft /x tensors differ")
            reply = await client.post("/reload_draft", json={"draft": "/x"})
            assert reply.status_code == 400 and "tensors differ" in reply.json()["error"]["message"]
    asyncio.run(scenario())
