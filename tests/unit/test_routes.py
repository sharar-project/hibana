"""Router traces (#32): the tap, the per-request recorder and its aggregates, the sink files, the spec hook,
the draft confidence, and the offline dataset and statistics (bench/flashnext_routes.py, bench/route_stats.py)."""

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bench import route_stats
from sharar.engine import routes
from sharar.engine.config import EngineConfig, using
from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.engine.spec import SpecDecoder
from sharar.engine.traces import TraceSink, iter_records, summarize
from sharar.proposers.draft_head import TargetHead, confidence

L, K, E = 3, 4, 16


def _rows(n, seed=0):
    """Route rows as the kernel writes them: distinct ids per row and layer, softmax probabilities in descending order."""
    rng = np.random.default_rng(seed)
    ids = np.stack([np.stack([rng.permutation(E)[:K] for _ in range(L)]) for _ in range(n)]).astype(np.uint16)
    probs = -np.sort(-rng.dirichlet(np.ones(E), size=(n, L))[..., :K], axis=-1).astype(np.float32)
    return ids, probs


def _brute(ids, probs, kind, blocks):
    agg = routes.empty_aggregate(L, K, E)
    for n in range(ids.shape[0]):
        agg["rows"][kind[n]] += 1
        for layer in range(L):
            w = probs[n, layer].astype(np.float64) / probs[n, layer].sum()
            agg["topk_mass"][layer] += probs[n, layer].sum()
            for r in range(K):
                agg["counts"][layer, ids[n, layer, r]] += 1
                agg["mass"][layer, ids[n, layer, r]] += w[r]
                agg["rank_mass"][layer, r] += w[r]
    for first, size in blocks:
        agg["block_rounds"][size] += 1
        for layer in range(L):
            agg["block_distinct"][size, layer] += len(set(ids[first:first + size, layer].reshape(-1).tolist()))
    return agg


def test_fold_matches_a_row_by_row_count():
    ids, probs = _rows(11)
    kind = np.array([0, 0] + [1] * 9, dtype=np.int8)
    blocks = [(2, 5), (7, 4)]
    got = routes.fold(routes.empty_aggregate(L, K, E), ids, probs, kind, blocks)
    want = _brute(ids, probs, kind, blocks)
    for name in want:
        assert np.allclose(got[name], want[name]), name


def test_distinct_and_share_helpers():
    ids = np.array([[[0, 1], [2, 3]], [[1, 4], [2, 3]]], dtype=np.uint16)  # 2 rows, 2 layers, k=2
    assert routes.distinct_per_layer(ids, 8).tolist() == [3, 2]
    assert routes.experts_for_share(np.array([5.0, 3.0, 1.0, 1.0]), 0.5) == 1
    assert routes.experts_for_share(np.array([5.0, 3.0, 1.0, 1.0]), 0.8) == 2
    assert routes.experts_for_share(np.array([5.0, 3.0, 1.0, 1.0]), 0.9) == 3
    assert routes.experts_for_share(np.zeros(4), 0.5) == 0


def test_tap_stages_the_real_rows_and_the_draft_steps():
    tap = routes.RouteTap(L, K, E, "cpu", draft_depth=3)
    ids_view, probs_view = tap.layer(1)
    assert ids_view.shape == (routes.ROWS, K) and ids_view.stride(0) == L * K and ids_view.stride(1) == 1
    ids_view[:3] = torch.arange(12, dtype=torch.int32).view(3, 4)
    probs_view[:3] = 0.25
    for step in range(2):
        tap.draft_slot[0][:1] = 100 + step
        tap.draft_slot[1][:1] = 0.5
        tap.keep_draft(step, 1)
    assert tap.stage(3, draft_rows=1, forward_rows=8)
    ids, probs, draft_ids, draft_probs = tap.take()
    assert ids.dtype == np.uint16 and ids.shape == (3, L, K) and ids[2, 1].tolist() == [8, 9, 10, 11]
    assert probs[0, 1].tolist() == [0.25] * 4 and probs[0, 0].tolist() == [0.0] * 4
    assert draft_ids.shape == (1, 2, K) and draft_ids[0, :, 0].tolist() == [100, 101]
    assert tap.take() is None  # taken once
    assert not tap.stage(3, forward_rows=routes.ROWS + 1)  # a padded graph too large to have recorded
    assert tap.take() is None


def test_recorder_keeps_rows_in_forward_order_and_folds_the_rest(tmp_path):
    rec = routes.RouteRecorder(L, K, E, full=True)
    d_ids, d_probs = _rows(1, seed=1)
    v_ids, v_probs = _rows(5, seed=2)
    draft_ids = np.arange(4 * K, dtype=np.uint16).reshape(4, K)
    rec.add(routes.DECODE, -1, 10, d_ids, d_probs)
    rec.add(routes.VERIFY, 0, 11, v_ids, v_probs, draft_ids, np.full((4, K), 0.1, np.float32))
    rec.add(routes.VERIFY, 1, 14, v_ids[:3], v_probs[:3])
    info = rec.write(tmp_path / "r.npz")
    assert info["rows"] == 9 and info["full"] and info["layers"] == L
    data = routes.load_routes(tmp_path / "r.npz")
    assert data["kind"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert data["round"].tolist() == [-1, 0, 0, 0, 0, 0, 1, 1, 1]
    assert data["offset"].tolist() == [0, 0, 1, 2, 3, 4, 0, 1, 2]
    assert data["position"].tolist() == [10, 11, 12, 13, 14, 15, 14, 15, 16]
    assert data["probs"].dtype == np.float16 and np.array_equal(data["ids"][1:6], v_ids)
    assert data["draft_ids"].shape == (1, 4, K) and data["draft_round"].tolist() == [0] and data["draft_steps"].tolist() == [4]
    want = _brute(np.concatenate([d_ids, v_ids, v_ids[:3]]), np.concatenate([d_probs, v_probs, v_probs[:3]]),
                  np.array([0] + [1] * 8), [(1, 5), (6, 3)])
    for name in want:
        assert np.allclose(data[f"agg_{name}"], want[name]), name


def test_aggregate_only_recorder_folds_on_the_engine_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "FOLD_ROWS", 4)
    rec = routes.RouteRecorder(L, K, E, full=False)
    ids, probs = _rows(10)
    rec.add(routes.VERIFY, 0, 0, ids[:5], probs[:5])
    assert rec._parts == []  # folded past FOLD_ROWS
    rec.add(routes.VERIFY, 1, 5, ids[5:], probs[5:])
    data = routes.load_routes(tmp_path / "r.npz") if rec.write(tmp_path / "r.npz") else None
    assert "ids" not in data and data["agg_rows"].tolist() == [0, 10] and data["agg_block_rounds"][5] == 2
    s = routes.summarize_aggregate({name[4:]: v for name, v in data.items()})
    assert s["rows"] == {"decode": 0, "verify": 10} and len(s["rank_share"]) == K and abs(sum(s["rank_share"]) - 1) < 1e-6
    assert set(s["distinct_experts_per_verify_block"]) == {5}


def test_sink_samples_full_rows_and_stores_the_route_file(tmp_path):
    tap = SimpleNamespace(layers=L, k=K, experts=E)
    sink = TraceSink(tmp_path)
    with using(EngineConfig(trace_router=True, trace_router_sample=0.5)):
        recs = [sink.route_recorder(tap) for _ in range(4)]
        assert [r.full for r in recs] == [False, True, False, True]
        assert sink.route_recorder(None) is None
    req = Request(prompt_ids=[1, 2], params=SamplingParams(max_new_tokens=4, accept_top_k=3, accept_margin=2.0), request_id=7)
    req.output_ids, req.finish_reason, req.t_finish = [5, 6, 7], "length", time.perf_counter()
    req.trace_rounds = [{"a": 1, "y": [6, 9], "z": [6, 7, 3], "c": 7, "p1": [0.9, 0.2], "m": [0.8, 0.05]}]
    req.trace_routes = recs[1]
    ids, probs = _rows(3)
    req.trace_routes.add(routes.VERIFY, 0, 2, ids, probs)
    assert sink.record(req)
    deadline = time.time() + 5
    while sink.records < 1 and time.time() < deadline:
        time.sleep(0.01)
    sink.close()
    doc = next(iter_records(tmp_path))
    assert doc["v"] == 3 and doc["params"]["accept_top_k"] == 3 and doc["spec"]["rounds"][0]["p1"] == [0.9, 0.2]
    assert doc["routes"]["rows"] == 3 and doc["routes"]["full"] and doc["routes"]["path"].endswith("7.routes.npz")
    assert routes.load_routes(doc["routes"]["path"])["ids"].shape == (3, L, K)
    s = summarize(tmp_path)["routes"]
    assert s["requests"] == 1 and s["requests_with_every_row"] == 1 and s["rows"] == {"decode": 0, "verify": 3}


def test_spec_files_each_requests_verify_block_under_its_round():
    tap = routes.RouteTap(L, K, E, "cpu", draft_depth=2)
    for row in range(6):  # two requests, k = 2: rows [r0: pending, d0, d1, r1: pending, d0, d1]
        tap.ids[row] = row
    tap.draft_slot[0][:2] = torch.tensor([[40], [41]], dtype=torch.int32)
    tap.keep_draft(0, 2)
    tap.keep_draft(1, 2)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex = SimpleNamespace(route_tap=tap)
    reqs = []
    for i in range(2):
        r = Request(prompt_ids=[1, 2, 3], params=SamplingParams(max_new_tokens=8), request_id=i)
        r.output_ids = [4, 5]
        r.spec_rounds = 3 + i
        r.trace_rounds = [{}] * r.spec_rounds + [{}]
        r.trace_routes = routes.RouteRecorder(L, K, E, full=True) if i == 1 else None
        reqs.append(r)
    batch = SimpleNamespace(requests=reqs, tokens=torch.zeros(6))
    spec._stage_routes(batch, forward_rows=6)
    assert spec._routes_staged
    spec._trace_routes(reqs, 2)
    assert not spec._routes_staged
    data = reqs[1].trace_routes.arrays()
    assert data["ids"][:, 0, 0].tolist() == [3, 4, 5] and data["round"].tolist() == [4, 4, 4]
    assert data["position"].tolist() == [4, 5, 6]  # the pending token sits at context_len - 1
    assert data["draft_ids"][0, :, 0].tolist() == [41, 41]  # the second request's draft row at both steps
    spec._trace_routes(reqs, 2)  # nothing staged: no-op
    assert reqs[1].trace_routes.rows == 3


def test_no_request_recording_stages_nothing():
    tap = routes.RouteTap(L, K, E, "cpu")
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex = SimpleNamespace(route_tap=tap)
    reqs = [Request(prompt_ids=[1], params=SamplingParams(), request_id=0)]
    spec._stage_routes(SimpleNamespace(requests=reqs, tokens=torch.zeros(3)), 3)
    assert not spec._routes_staged and tap.take() is None
    spec.ex = SimpleNamespace()  # executors without a tap (the 27B, recording off)
    spec._stage_routes(SimpleNamespace(requests=reqs, tokens=torch.zeros(3)), 3)
    assert not spec._routes_staged


def test_draft_confidence_is_the_heads_softmax_top1_and_margin():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 5.0, 0.0]])
    head = TargetHead(lambda x: x)
    token, conf = head.argmax_confidence(logits)
    p = torch.softmax(logits, -1)
    assert token.tolist() == head.argmax(logits).tolist() == [0, 2]
    assert torch.allclose(conf[:, 0], p.max(-1).values)
    assert torch.allclose(conf[0, 1], p[0, 0] - p[0, 1])
    assert torch.allclose(confidence(torch.zeros(3, 5)), torch.tensor([[0.2, 0.0]] * 3))


def test_route_stats_first_use_and_prefix_counts():
    # 3 rows (pending, d0, d1), 1 layer, k=2: row 2 brings experts 5 and 6; row 1 brings 2
    ids = np.array([[[0, 1]], [[1, 2]], [[5, 6]]], dtype=np.uint16)
    assert route_stats.distinct_by_prefix(ids, 8)[:, 0].tolist() == [2, 3, 5]
    assert [x.tolist() for x in route_stats.first_use_split(ids, accepted=0, experts=8)] == [[5], [3]]  # rows 1, 2 rejected
    assert [x.tolist() for x in route_stats.first_use_split(ids, accepted=1, experts=8)] == [[5], [2]]
    assert [x.tolist() for x in route_stats.first_use_split(ids, accepted=2, experts=8)] == [[5], [0]]


def test_route_stats_masses_ranks_and_acceptance():
    kind, offset = np.array([0, 1, 1, 1]), np.array([0, 0, 1, 2])
    assert route_stats.kept_rows(kind, offset, np.array([0, 1, 1, 1])).tolist() == [True, True, True, False]
    ids = np.array([[[0, 1]], [[0, 2]]], dtype=np.uint16)
    probs = np.array([[[0.6, 0.2]], [[0.3, 0.1]]], dtype=np.float32)
    mass, counts = route_stats.mass_per_expert(ids, probs, 4)
    assert np.allclose(mass[0], [0.75 + 0.75, 0.25, 0.25, 0.0]) and counts[0].tolist() == [2, 1, 1, 0]
    assert route_stats.experts_for_share(mass, 0.5).tolist() == [1] and route_stats.experts_for_share(mass, 0.9).tolist() == [3]
    assert np.isclose(route_stats.effective_experts(np.array([1.0, 1.0, 0.0])), 2.0)
    a, b = np.array([[5.0, 4.0, 0.0, 1.0]]), np.array([[5.0, 0.0, 4.0, 1.0]])
    assert route_stats.top_set_overlap(a, b, 0.8).tolist() == [1 / 3]  # {0, 1} against {0, 2}
    share, topk = route_stats.rank_shares(probs)
    assert np.allclose(share[0], [0.75, 0.25]) and np.isclose(topk[0], 0.6)
    p1 = np.array([[0.95, 0.4], [0.2, 0.9], [0.96, 0.96]])
    table = route_stats.acceptance_by_confidence(p1, np.array([2, 0, 1]), bins=(0.0, 0.5, 1.0001))
    cell = {(c["position"], c["p1_lo"]): (c["n"], c["accept"]) for c in table}
    assert cell[(0, 0.5)] == (2, 1.0) and cell[(0, 0.0)] == (1, 0.0)
    assert cell[(1, 0.0)] == (1, 1.0) and cell[(1, 0.5)] == (1, 0.0)  # round 1 never reached position 1


def test_dataset_and_statistics_from_a_trace_directory(tmp_path):
    """bench/flashnext_routes.py dataset over sink output, then bench/route_stats.py on it (CPU, synthetic routes)."""
    from bench import flashnext_routes
    run = tmp_path / "run" / "arm-greedy"
    sink = TraceSink(run / "traces")
    rows = []
    for i, cat in enumerate(("code", "prose")):
        req = Request(prompt_ids=[1, 2], params=SamplingParams(max_new_tokens=8), request_id=i + 1)
        req.output_ids, req.finish_reason, req.t_finish = [5, 6, 7, 8, 9], "length", time.perf_counter()
        req.trace_routes = routes.RouteRecorder(L, K, E, full=True)
        ids, probs = _rows(7, seed=i)
        req.trace_routes.add(routes.DECODE, -1, 2, ids[:1], probs[:1])
        req.trace_rounds = [{"a": 2, "y": [6, 7], "z": [6, 7, 8], "c": 8, "p1": [0.9, 0.8], "m": [0.5, 0.4]},
                            {"a": 0, "y": [1, 2], "z": [9, 3, 4], "c": 9, "p1": [0.3, 0.9], "m": [0.1, 0.8]}]
        req.trace_routes.add(routes.VERIFY, 0, 3, ids[1:4], probs[1:4], ids[4:6, 0], probs[4:6, 0])
        req.trace_routes.add(routes.VERIFY, 1, 6, ids[4:7], probs[4:7], ids[1:3, 0], probs[1:3, 0])
        assert sink.record(req)
        rows.append({"trace_id": req.request_id, "prompt_id": f"{cat}-x", "cat": cat, "mode": "greedy", "tok_s": 50.0})
    deadline = time.time() + 5
    while sink.records < 2 and time.time() < deadline:
        time.sleep(0.01)
    sink.close()
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "meta.json").write_text(json.dumps({"arm": "arm", "profile": "exact", "k": 2, "mode": "greedy"}))
    flashnext_routes.dataset(argparse.Namespace(run=str(tmp_path / "run"), out=str(tmp_path / "data")))
    arm = route_stats.load_arm(tmp_path / "data" / "arm-greedy")
    assert arm["rounds"]["row0"].tolist() == [1, 4, 8, 11] and arm["rounds"]["accepted"].tolist() == [2, 0, 2, 0]
    assert arm["rows"]["accepted_of_row"].tolist() == [0, 2, 2, 2, 0, 0, 0] * 2
    b = route_stats.blocks(arm, E)
    assert b["rounds"] == 4 and b["k"] == 2
    ids = arm["rows"]["ids"]
    d, w = zip(*(route_stats.first_use_split(ids[r0:r0 + 3], a, E) for r0, a in ((1, 2), (4, 0), (8, 2), (11, 0))))
    assert b["rejected_first_use"]["fraction_of_round_expert_bytes"] == round(float(np.sum(w) / np.sum(d)), 4)
    s = route_stats.skew(arm, E)
    assert s["all"]["tokens"] == 10 and set(s) == {"all", "code", "prose", "overlap_50pct_sets"}  # 1 decode + 3 + 1 kept verify rows per request
    a = route_stats.acceptance(arm)
    assert a["accept_rate_by_position"] == [0.5, 1.0]
    assert "### arm-greedy" in route_stats.tables({"arms": {"arm-greedy": {"tokens": 10, "skew": s, "ranks": route_stats.ranks(arm),
                                                                           "blocks": b, "acceptance": a}}})
