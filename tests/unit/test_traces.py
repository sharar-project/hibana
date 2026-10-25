"""Opt-in request traces: engine hooks, writer, retention, API and CLI plumbing."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch

from sharar.api.server import create_app
from sharar.engine.contracts import VerifiedFeatures
from sharar.engine.request import Request
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.engine.spec import SpecDecoder
from sharar.engine.traces import TraceSink, iter_records, summarize
from sharar.cache.request_state import Reservation, StateManager
from tests.unit.test_scheduler import FakeExecutor


def _drain(sink, expected):
    deadline = time.time() + 5
    while sink.records + sink.dropped < expected and time.time() < deadline:
        time.sleep(0.01)
    sink.close()


def test_finished_requests_are_recorded_unless_opted_out(tmp_path):
    ex = FakeExecutor(pages=1000, slots=4, eos_after={2: 3})
    sched = Scheduler(ex)
    sink = sched.sink = TraceSink(tmp_path / "traces")
    kept = sched.submit([1, 2, 3], SamplingParams(max_new_tokens=2, temperature=0.5, seed=7))
    skipped = sched.submit([4, 5], SamplingParams(max_new_tokens=2, trace=False))
    assert kept.trace_rounds == [] and skipped.trace_rounds is None
    sched.run_until_idle()
    _drain(sink, 1)
    docs = list(iter_records(tmp_path / "traces"))
    assert [d["id"] for d in docs] == [kept.request_id]
    doc = docs[0]
    assert doc["prompt_ids"] == [1, 2, 3] and doc["output_ids"] == kept.output_ids and len(doc["output_ids"]) == 2
    assert doc["reason"] == "length" and doc["spec"] is None and doc["cached"] == 0 and doc["v"] == 4 and "features" not in doc
    assert doc["params"] == {"max_new_tokens": 2, "temperature": 0.5, "top_p": 0.95, "top_k": 20, "seed": 7, "ignore_eos": False,
                             "accept_top_k": 1, "accept_margin": 0.0}
    assert doc["arrival"] <= doc["finish"] <= time.time()
    stats = sched.stats()["traces"]
    assert stats["records"] == 1 and stats["dropped"] == 0 and stats["files"] == 1 and stats["bytes"] == stats["bytes_written"]
    assert summarize(tmp_path / "traces") == {"requests": 1, "prompt_tokens": 3, "output_tokens": 2, "reasons": {"length": 1},
                                              "spec_rounds": 0, "accepted_tokens": 0, "tokens_per_round": None}


def test_no_sink_means_no_recording_state():
    sched = Scheduler(FakeExecutor(pages=100, slots=4))
    req = sched.submit([1], SamplingParams(max_new_tokens=1))
    sched.run_until_idle()
    assert req.trace_rounds is None and "traces" not in sched.stats()


def test_cancelled_request_with_output_is_recorded(tmp_path):
    ex = FakeExecutor(pages=1000, slots=4)
    sched = Scheduler(ex)
    sink = sched.sink = TraceSink(tmp_path)
    req = sched.submit([1, 2], SamplingParams(max_new_tokens=8))
    sched.step()
    while not req.output_ids:
        sched.step()
    sched.cancel(req)
    sched.step()
    _drain(sink, 1)
    (doc,) = iter_records(tmp_path)
    assert doc["reason"] == "cancelled" and doc["output_ids"] == req.output_ids


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_speculative_rounds_record_proposals_and_targets(accepted):
    pool = SimpleNamespace(ssm_all=torch.zeros((1, 2, 1)), conv_all=torch.zeros((1, 2, 1)), num_slots=2)
    ex = SimpleNamespace(state_pool=pool, kv_pool=SimpleNamespace(free_ids=lambda ids: None))
    ex.state = StateManager(ex.kv_pool, pool)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex, spec.k, spec.steps, spec.timing = ex, 3, 4, False
    spec.proposer = SimpleNamespace(accepted_batch=lambda *args: None)
    spec.inter_ssm_all = torch.arange(1, 5).reshape(1, 1, 4, 1).float()
    spec.conv_windows_all = spec.inter_ssm_all.clone()
    spec.replay = None
    spec.reset_stats()
    draft = [20, 30, 40]
    target = draft + [50]
    if accepted < 3:
        target[accepted] = 99
    r = Request([1, 2, 3], SamplingParams(max_new_tokens=20, stop_token_ids=()), slot=1, pages=torch.tensor([101, 102, 103]), prefill_pos=3)
    r.committed_len = 3
    r.push(10)
    r.reservation = Reservation(torch.tensor([104, 105, 106, 107]), [104, 105, 106, 107])
    r.trace_rounds = []
    now = time.perf_counter()
    spec._accept([r], torch.tensor([draft]), torch.tensor([target]), VerifiedFeatures(), now, now)
    assert r.trace_rounds == [{"a": accepted, "y": draft, "z": target, "c": target[accepted]}]
    assert r.output_ids[1:] == draft[:accepted] + [target[accepted]]
    doc = json.loads(json.dumps({"spec": {"k": 3, "rounds": r.trace_rounds}}))
    assert doc["spec"]["rounds"][0]["y"] == draft


@pytest.mark.parametrize("greedy", [True, False])
def test_rows_in_sampled_batches_keep_argmax_only_when_greedy(greedy):
    pool = SimpleNamespace(ssm_all=torch.zeros((1, 2, 1)), conv_all=torch.zeros((1, 2, 1)), num_slots=2)
    ex = SimpleNamespace(state_pool=pool, kv_pool=SimpleNamespace(free_ids=lambda ids: None))
    ex.state = StateManager(ex.kv_pool, pool)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex, spec.k, spec.steps, spec.timing = ex, 2, 3, False
    spec.proposer = SimpleNamespace(accepted_batch=lambda *args: None)
    spec.inter_ssm_all = torch.arange(1, 4).reshape(1, 1, 3, 1).float()
    spec.conv_windows_all = spec.inter_ssm_all.clone()
    spec.replay = None
    spec.reset_stats()
    # Rows inside a sampled batch take the per-row path: z is [k+1, V] logits.
    params = SamplingParams(max_new_tokens=20, temperature=0.0 if greedy else 1.0, top_k=1, top_p=1.0, seed=3, stop_token_ids=())
    r = Request([1], params, slot=0, pages=torch.tensor([101]), prefill_pos=1)
    r.generator = torch.Generator().manual_seed(3)
    r.committed_len = 1
    r.push(10)
    r.reservation = Reservation(torch.tensor([102, 103, 104]), [102, 103, 104])
    r.trace_rounds = []
    logits = torch.full((1, 3, 8), -1.0)
    logits[0, 0, 5] = 1.0  # target argmax 5 == first proposal
    logits[0, 1, 6] = 1.0  # rejects proposal 7 with correction 6
    now = time.perf_counter()
    spec._accept([r], torch.tensor([[5, 7]]), logits, VerifiedFeatures(), now, now, proposal=SimpleNamespace(probabilities=None, support=None))
    # top_k=1 makes the sampled row deterministic: it follows the argmax too.
    assert r.trace_rounds == [{"a": 1, "y": [5, 7], "z": [5, 6, 0] if greedy else None, "c": 6}]
    assert r.output_ids == [10, 5, 6]


def test_rotation_and_retention_prune_oldest_files(tmp_path):
    sink = TraceSink(tmp_path, max_bytes=2000, rotate_bytes=600)
    params = SamplingParams(max_new_tokens=4)
    for i in range(40):
        req = Request(prompt_ids=[i] * 40, params=params, request_id=i)
        req.output_ids = [1, 2, 3]
        req.finish_reason = "length"
        req.t_finish = time.perf_counter()
        assert sink.record(req)
    _drain(sink, 40)
    files = sorted(tmp_path.glob("traces-*.jsonl"))
    assert 1 < len(files) < 40  # rotated, then pruned
    assert sum(f.stat().st_size for f in files) <= 2000
    ids = [d["id"] for d in iter_records(tmp_path)]
    assert ids == list(range(ids[0], 40)) and ids[0] > 0  # only the oldest records were dropped
    assert sink.records == 40 and sink.dropped == 0
    assert sink.stats()["bytes"] == sum(f.stat().st_size for f in files)
    assert sink.stats()["files"] == len(files)


def test_stats_include_existing_storage_without_scanning_on_the_engine_thread(tmp_path, monkeypatch):
    """Traces recorded before format 4 kept feature files beside them: they count toward the cap."""
    trace = tmp_path / "traces-20260911-000000-0001.jsonl"
    trace.write_bytes(b"{}\n")
    features = tmp_path / "features" / trace.stem
    features.mkdir(parents=True)
    (features / "1.bin").write_bytes(b"abcdef")
    sink = TraceSink(tmp_path)
    try:
        def no_scan(*args, **kwargs):
            raise AssertionError("stats must not scan storage")
        monkeypatch.setattr(sink, "_files", no_scan)
        monkeypatch.setattr(sink, "_size", no_scan)
        assert sink.stats()["files"] == 1
        assert sink.stats()["bytes"] == 9
    finally:
        sink.close()


def test_partial_trailing_line_is_skipped(tmp_path):
    path = tmp_path / "traces-20260911-000000-0001.jsonl"
    good = json.dumps({"v": 1, "id": 1, "prompt_ids": [1], "output_ids": [2], "reason": "stop", "spec": None})
    path.write_text(good + "\n" + '{"v": 1, "id": 2, "prompt_ids": [1')
    assert [d["id"] for d in iter_records(tmp_path)] == [1]


def test_trace_flag_is_validated():
    with pytest.raises(InvalidRequest):
        SamplingParams(trace="no").validate()
    SamplingParams(trace=False).validate()


@pytest.mark.parametrize("route,body", [
    ("/generate", {"input_ids": [1, 2]}),
    ("/v1/completions", {"prompt": "hello"}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
])
@pytest.mark.parametrize("value,expected", [(None, True), (False, False), (True, True)])
def test_http_trace_field_reaches_the_engine(route, body, value, expected):
    async def scenario():
        queue = asyncio.Queue()
        queue.put_nowait(([2], True, "stop"))
        ticket = SimpleNamespace(queue=queue, req=SimpleNamespace(prefix_len=0, spec_rounds=0))
        engine = SimpleNamespace(submit=AsyncMock(return_value=ticket), cancel=Mock())
        tok = SimpleNamespace(encode=Mock(return_value=[1]), decode=lambda ids, **kwargs: "ok")
        template = SimpleNamespace(render=Mock(return_value="prompt"))
        app = create_app(engine, tok, template, {})
        payload = dict(body)
        if value is not None:
            payload["trace"] = value
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post(route, json=payload)
        assert response.status_code == 200, response.text
        params = engine.submit.await_args.args[1]
        assert params.trace is expected

    asyncio.run(scenario())


def test_http_rejects_non_boolean_trace():
    async def scenario():
        engine = SimpleNamespace(submit=AsyncMock())
        app = create_app(engine, SimpleNamespace(encode=Mock(return_value=[1])), SimpleNamespace(render=Mock()), {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.post("/generate", json={"input_ids": [1], "trace": "yes"})
        assert response.status_code == 400
        engine.submit.assert_not_awaited()

    asyncio.run(scenario())


def test_serve_cli_passes_trace_settings(monkeypatch, tmp_path):
    from sharar import cli

    calls = []
    monkeypatch.setattr("sharar.api.server.serve", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setenv("SHARAR_TRACES", str(tmp_path))
    monkeypatch.setenv("SHARAR_TRACES_MAX_MIB", "7")
    assert cli.main(["engine", "--spec", "0"]) == 0
    assert calls[-1]["traces"] == str(tmp_path) and calls[-1]["traces_max_mib"] == 7
    assert cli.main(["engine", "--spec", "0", "--traces", "/elsewhere", "--traces-max-mib", "3"]) == 0
    assert calls[-1]["traces"] == "/elsewhere" and calls[-1]["traces_max_mib"] == 3
    monkeypatch.delenv("SHARAR_TRACES")
    monkeypatch.delenv("SHARAR_TRACES_MAX_MIB")
    assert cli.main(["engine", "--spec", "0"]) == 0
    assert calls[-1]["traces"] is None and calls[-1]["traces_max_mib"] == 32768
    with pytest.raises(SystemExit):
        cli.main(["engine", "--spec", "0", "--traces", "/x", "--traces-max-mib", "0"])


def test_traces_cli_summarizes(monkeypatch, tmp_path, capsys):
    from sharar import cli

    sink = TraceSink(tmp_path)
    req = Request(prompt_ids=[1, 2], params=SamplingParams(max_new_tokens=4), request_id=3)
    req.output_ids, req.finish_reason, req.t_finish = [5, 6, 7], "stop", time.perf_counter()
    req.trace_rounds = [{"a": 2, "y": [5, 6, 9], "z": [5, 6, 7, 8], "c": 7}]
    sink.record(req)
    _drain(sink, 1)
    assert cli.main(["traces", "--dir", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 1 and out["spec_rounds"] == 1 and out["tokens_per_round"] == 3.0
    monkeypatch.delenv("SHARAR_TRACES", raising=False)
    monkeypatch.setenv("SHARAR_RESULTS", str(tmp_path / "nothing-recorded"))
    with pytest.raises(SystemExit, match="no traces"):
        cli.main(["traces"])
