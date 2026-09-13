"""Draft training pieces that run on the CPU: objective, labels, data, mask, export.

The objective is checked against the naive formulas SpecForge's own test suite
uses as ground truth (``tests/test_utils/test_dflash_losses.py``:
``_targets_and_mask``, ``_neg_log_q``, ``_naive_dflash_loss``), vendored below.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sharar.engine.request import Request
from sharar.engine.sampler import SamplingParams
from sharar.engine.traces import TraceSink
from sharar.train.data import load_samples, split, summary
from sharar.train.draft import TrainableDraft, context_key, dequantize_nvfp4
from sharar.train.loss import block_targets, block_terms, greedy_walk, sample_anchors, selector_scores


# ---- SpecForge naive references (tests/test_utils/test_dflash_losses.py) ----
def _targets_and_mask(input_ids, loss_mask, anchors, keep_mask, block_size):
    bsz, seq_len = input_ids.shape
    n_blocks = anchors.shape[1]
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices)
    binary_mask = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    binary_mask = binary_mask * (label_indices < seq_len).double()
    binary_mask = binary_mask * (offsets > 0).double()
    gathered_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices)
    binary_mask = binary_mask * gathered_loss_mask
    return targets, binary_mask


def _neg_log_q(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view_as(targets)


def _naive_dflash_loss(neg_log_q, binary_mask, gamma):
    weight = binary_mask
    if gamma is not None and gamma > 0:
        block_size = neg_log_q.shape[-1]
        positions = torch.arange(block_size, dtype=neg_log_q.dtype).view(1, 1, -1)
        decay = torch.exp(-(positions - 1).clamp(min=0) / gamma)
        weight = weight * decay
    return (neg_log_q * weight).sum() / weight.sum()


@pytest.mark.parametrize("gamma", [None, 7.0, 2.5])
def test_block_objective_matches_specforge_naive_formula(gamma):
    torch.manual_seed(3)
    length, prompt_len, block, vocab = 40, 12, 6, 50
    ids = torch.randint(0, vocab, (length,))
    anchors = sample_anchors(prompt_len, length, 9, torch.Generator().manual_seed(1))
    assert anchors.tolist() == sorted(anchors.tolist()) and (anchors >= prompt_len).all() and (anchors <= length - 2).all()
    anchors = torch.cat([anchors, torch.tensor([length - 3])])  # a block running past the end
    logits = torch.randn(anchors.shape[0], block, vocab, dtype=torch.float64)
    target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(ids, anchors, block, prompt_len, gamma)
    loss_mask = (torch.arange(length) >= prompt_len).double()[None]
    want_targets, want_mask = _targets_and_mask(ids[None], loss_mask, anchors[None], torch.ones(1, anchors.shape[0], dtype=torch.bool), block)
    assert torch.equal(target_ids, want_targets[0]) and torch.equal(weight_mask.double(), want_mask[0])
    assert torch.equal(predecessor_ids[:, 1:], target_ids[:, :-1]) and torch.equal(predecessor_ids[:, 0], target_ids[:, 0])
    terms = block_terms(logits, torch.zeros(anchors.shape[0], block, 4), target_ids, predecessor_ids, weight_mask, loss_weights,
                        top_k=4, hidden_projection=None, predecessor_codebook=None, successor_codebook=None)
    want = _naive_dflash_loss(_neg_log_q(logits, want_targets), want_mask, gamma)
    torch.testing.assert_close(terms.loss(1.0).double(), want, rtol=0, atol=1e-6)  # decay weights are float32
    assert terms.selector_ce_num == 0 and terms.label_count == weight_mask.sum()


def test_selector_objective_is_classification_over_candidates_with_shared_denominator():
    torch.manual_seed(5)
    rows, block, vocab, top_k, rank, hidden_size = 3, 5, 30, 4, 6, 8
    ids = torch.randint(0, vocab, (rows * block + 4,))
    anchors = torch.tensor([0, 5, 10])
    logits = torch.randn(rows, block, vocab)
    hidden = torch.randn(rows, block, hidden_size)
    weights = (torch.randn(rank, hidden_size), torch.randn(vocab, rank), torch.randn(vocab, rank))
    target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(ids, anchors, block, 0, 7.0)
    terms = block_terms(logits, hidden, target_ids, predecessor_ids, weight_mask, loss_weights, top_k=top_k,
                        hidden_projection=weights[0], predecessor_codebook=weights[1], successor_codebook=weights[2])
    unary, candidates = logits.topk(top_k, dim=-1)
    scores = selector_scores(unary, candidates, hidden, predecessor_ids, *weights)
    # SpecForge CandidateSelector.score_candidates: unary + einsum(pred[pred] * proj(h), succ[cand])
    want_scores = unary + torch.einsum("...r,...kr->...k", weights[1][predecessor_ids] * (hidden @ weights[0].T), weights[2][candidates])
    torch.testing.assert_close(scores, want_scores)
    matches = candidates.eq(target_ids[..., None])
    is_candidate = matches.any(-1)
    ce = F.cross_entropy(scores.reshape(-1, top_k), matches.long().argmax(-1).reshape(-1), reduction="none").reshape(rows, block)
    want_num = (ce * loss_weights * is_candidate).sum()
    torch.testing.assert_close(terms.selector_ce_num, want_num)
    base = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none").reshape(rows, block)
    want_loss = ((base * loss_weights).sum() + 0.5 * want_num) / loss_weights.sum()
    torch.testing.assert_close(terms.loss(0.5), want_loss)
    assert terms.selector_covered_num == (weight_mask * is_candidate).sum()
    # Rows whose label is outside the top-k carry no selector weight.
    assert terms.selector_weight_den == (loss_weights * is_candidate).sum()


def test_terms_add_detach_and_metrics_with_and_without_teacher():
    logits = torch.randn(2, 3, 9, requires_grad=True)
    ids = torch.arange(12)
    t = block_targets(ids, torch.tensor([0, 4]), 3, 0, 7.0)
    a = block_terms(logits, torch.zeros(2, 3, 2), *t, top_k=3, hidden_projection=None, predecessor_codebook=None, successor_codebook=None)
    total = (a + a).detached()
    assert total.ce_num == 2 * a.ce_num.detach() and total.loss(1.0).item() == pytest.approx(a.loss(1.0).item())
    assert all(v == v for v in total.metrics().values()) and "kl" not in total.metrics()
    d = a.detached()
    assert d.kl_num is None and not d.ce_num.requires_grad and a.ce_num.requires_grad
    teacher = (torch.randint(0, 9, (2, 3, 4)), torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
    b = block_terms(logits, torch.zeros(2, 3, 2), *t, top_k=3, hidden_projection=None, predecessor_codebook=None, successor_codebook=None, teacher=teacher)
    assert b.kl_num is not None and "kl" in b.metrics() and b.loss(1.0, 1.0) > b.loss(1.0, 0.0)
    assert (b + b).kl_num == 2 * b.kl_num and b.detached().kl_num is not None and not b.detached().kl_num.requires_grad


def test_anchor_sampling_is_bounded_and_deterministic():
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    a, b = sample_anchors(10, 100, 20, g1), sample_anchors(10, 100, 20, g2)
    assert torch.equal(a, b) and a.shape[0] == 20 and a.min() >= 10 and a.max() <= 98 and len(set(a.tolist())) == 20
    assert sample_anchors(10, 13, 50).tolist() == [10, 11]
    assert sample_anchors(5, 6, 3).numel() == 0
    with pytest.raises(ValueError):
        sample_anchors(6, 6, 3)


def test_greedy_walk_matches_sequential_argmax():
    torch.manual_seed(2)
    candidates = torch.randint(0, 100, (3, 4, 5))
    scores = torch.randn(3, 4, 5, 5)
    out = greedy_walk(candidates, scores)
    for row in range(3):
        previous = 0
        for step in range(4):
            previous = int(scores[row, step, previous].argmax())
            assert out[row, step] == candidates[row, step, previous]


def test_attention_mask_sees_context_before_anchor_within_window_and_own_block():
    draft = TrainableDraft.__new__(TrainableDraft)
    draft.block, draft.window = 3, 4
    anchors = torch.tensor([2, 5])
    mask = draft.attention_mask(anchors, 6)
    assert mask.shape == (6, 12)
    # query (anchor 2, offset 0): context 0..1; offset 2: window keeps kv >= 2 + 2 - 3 = 1
    assert mask[0, :6].tolist() == [True, True, False, False, False, False]
    assert mask[2, :6].tolist() == [False, True, False, False, False, False]
    # own block only, never the other block
    assert mask[0, 6:].tolist() == [True, True, True, False, False, False]
    assert mask[4, 6:].tolist() == [False, False, False, True, True, True]
    # anchor 5 offset 0: context kv in [5 - 3, 5) = 2..4
    assert mask[3, :6].tolist() == [False, False, True, True, True, False]


def test_context_projections_are_frozen_by_default():
    assert context_key("fc.weight") and context_key("layers.3.self_attn.kv") and context_key("layers.0.self_attn.k_norm.weight")
    assert not context_key("layers.0.self_attn.q_proj.weight") and not context_key("candidate_selector.successor_codebook")


def test_state_dict_unfuses_projections_and_sync_writes_masters(tmp_path):
    weights = {
        "layers.0.self_attn.kv": torch.arange(8, dtype=torch.bfloat16).view(4, 2),
        "layers.0.mlp.gate_up": torch.arange(12, dtype=torch.bfloat16).view(6, 2),
        "fc.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "norm.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    model = SimpleNamespace(precision="bf16", fp8_weights={}, weights=weights, window=4, cfg={"num_hidden_layers": 1, "dflash_config": {"selector_top_k": 2}})
    draft = TrainableDraft(model, head=None, block=2)
    assert set(draft.masters) == {"layers.0.mlp.gate_up", "norm.weight"}  # kv and fc are context weights
    draft.masters["norm.weight"].data.fill_(3.0)
    out = draft.state_dict()
    assert torch.equal(out["layers.0.self_attn.k_proj.weight"], weights["layers.0.self_attn.kv"][:2])
    assert torch.equal(out["layers.0.self_attn.v_proj.weight"], weights["layers.0.self_attn.kv"][2:])
    assert torch.equal(out["layers.0.mlp.gate_proj.weight"], weights["layers.0.mlp.gate_up"][:3])
    assert torch.equal(out["layers.0.mlp.up_proj.weight"], weights["layers.0.mlp.gate_up"][3:])
    assert torch.equal(weights["norm.weight"], torch.full((2,), 3.0, dtype=torch.bfloat16))  # sync happened in place
    assert "layers.0.self_attn.kv" not in out and out["fc.weight"].dtype == torch.bfloat16
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"architectures": ["DFlash2DraftModel"]}')
    draft.save(tmp_path / "ckpt", snapshot, {"step": 1})
    from safetensors.torch import load_file
    saved = load_file(tmp_path / "ckpt" / "model.safetensors")
    assert set(saved) == set(out) and json.loads((tmp_path / "ckpt" / "manifest.json").read_text()) == {"step": 1}


def test_dequantize_nvfp4_applies_block_and_global_scales():
    # two rows, K=32: nibble pairs; row 0 all +1.0 (code 2), row 1 low nibble 6 (code 7), high nibble -0.5 (code 9)
    weight = torch.tensor([[0x22] * 16, [0x97] * 16], dtype=torch.uint8)
    scale = torch.tensor([[1.0, 2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
    out = dequantize_nvfp4(weight, scale, torch.tensor(0.25))
    assert out.shape == (2, 32)
    torch.testing.assert_close(out[0, :16].float(), torch.full((16,), 0.25), rtol=0, atol=0)
    torch.testing.assert_close(out[0, 16:].float(), torch.full((16,), 0.5), rtol=0, atol=0)
    assert out[1, 0].item() == 6 * 0.5 * 0.25 and out[1, 1].item() == -0.5 * 0.5 * 0.25 and out[1, 17].item() == -0.5 * 4 * 0.25


def _write_traces(directory, count, features=False):
    sink = TraceSink(directory)
    for i in range(count):
        req = Request(prompt_ids=[1] * (10 + i), params=SamplingParams(max_new_tokens=64), request_id=i + 1)
        req.output_ids = [2] * (4 if i == 0 else 40)
        req.finish_reason, req.t_finish = "stop", time.perf_counter()
        if features:
            req.trace_features = sink.recorder()
            rows = len(req.prompt_ids) + len(req.output_ids) - 1  # every position but the last output token
            req.trace_features.append(torch.arange(rows, dtype=torch.bfloat16)[:, None].expand(rows, 5120).contiguous(), 0)
        sink.record(req)
        time.sleep(0.001)
    deadline = time.time() + 5
    while sink.records < count and time.time() < deadline:
        time.sleep(0.01)
    sink.close()


def test_samples_load_split_and_truncate(tmp_path):
    _write_traces(tmp_path, 6)
    samples = load_samples(tmp_path, max_length=30, min_output=16)
    assert [s.request_id for s in samples] == [2, 3, 4, 5, 6]  # the four-token output is dropped
    assert all(s.length == 30 and s.prompt_len == 1 and s.output_len == 29 for s in samples)  # tails are kept
    train, holdout = split(samples, holdout_tokens=30, min_requests=1)
    assert [s.request_id for s in holdout] == [6] and len(train) == 4  # one request already covers 30 tokens
    train, holdout = split(samples, holdout_tokens=50, min_requests=1, max_fraction=0.5)
    assert [s.request_id for s in holdout] == [5, 6] and len(train) == 3
    assert len(split(samples, holdout_tokens=10**6, min_requests=1, max_fraction=0.5)[1]) == 2  # capped by the fraction
    assert len(split(samples, holdout_tokens=0, min_requests=3, max_fraction=0.9)[1]) == 3  # floor of requests
    assert split(samples[:1], 100)[1] == [] and split([], 100) == ([], [])
    assert summary(samples)["output_tokens"] == 5 * 29
    full = load_samples(tmp_path, max_length=8192)
    assert full[0].prompt_len == 11 and full[0].output_len == 40 and full[0].features is None
    assert load_samples(tmp_path, require_features=True) == []


def test_samples_carry_recorded_context_features(tmp_path):
    _write_traces(tmp_path, 3, features=True)
    samples = load_samples(tmp_path, require_features=True)
    assert [s.request_id for s in samples] == [2, 3]
    s = samples[0]
    assert s.features["start"] == 0 and s.features["rows"] == s.length - 1 and s.features["skip"] == 0
    context = s.context("cpu")
    assert context.shape == (s.length, 5120) and context[7, 0].item() == 7 and context[-1].abs().sum() == 0
    # truncation keeps the tail and shifts the recorded window
    cut = load_samples(tmp_path, max_length=30, require_features=True)
    t = cut[0]
    assert t.length == 30 and t.features["skip"] == 51 - 30 and t.features["start"] == 0
    assert t.context("cpu")[0, 0].item() == 21 and summary(cut)["with_features"] == 2


def test_promote_link_is_relative_and_atomic(tmp_path):
    from sharar.train.trainer import promote_link, resolve_draft
    runs = tmp_path / "runs"
    best = runs / "run-1" / "best"
    best.mkdir(parents=True)
    link = runs / "current"
    assert resolve_draft(link) is None
    promote_link(link, best)
    assert link.is_symlink() and os.readlink(link) == "run-1/best"
    assert resolve_draft(link) is None  # not a complete checkpoint yet
    (best / "config.json").write_text("{}")
    (best / "model.safetensors").write_bytes(b"")
    assert resolve_draft(link) == best.resolve()
    other = runs / "run-2" / "best"
    other.mkdir(parents=True)
    promote_link(link, other)
    assert os.readlink(link) == "run-2/best" and not (runs / ".current.tmp").exists()


def test_run_trains_until_stopped_then_evaluates_and_promotes(tmp_path, monkeypatch):
    from sharar.train import trainer as module
    _write_traces(tmp_path / "traces", 6, features=True)
    events = []

    class FakeTrainer:
        def __init__(self, cfg, draft, log):
            self.cfg, self.output = cfg, Path(cfg.output)
            self.output.mkdir(parents=True, exist_ok=True)
            self.history = self.output / "log.jsonl"
            self.scores = iter([1.0, 1.5, 1.2, 2.0])

        def _record(self, row):
            events.append(row)

        def evaluate(self, samples):
            return {"tokens_per_round": next(self.scores), "rounds": 10, "emitted": 0, "accepted": 0, "seconds": 0.0, "per_sample": []}

        def step(self, batch, step):
            if step == 5:
                stop.requested = True  # as SIGTERM would
            return {"step": step, "loss": 1.0, "grad_norm": 1.0, "lr": 0.0, "labels": 1.0, "seconds": 0.0, "ce": 1.0,
                    "accuracy": 0.5, "selector_ce": 0.0, "selector_accuracy": 0.0, "selector_coverage": 0.0}

    saved = []

    class FakeDraft:
        def parameter_count(self):
            return 7

        def save(self, directory, snapshot, manifest):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors").write_bytes(b"")
            saved.append(directory.name)

    monkeypatch.setattr(module, "Trainer", FakeTrainer)
    stop = module.StopRequest()
    runs = tmp_path / "runs"
    cfg = module.TrainConfig(traces=str(tmp_path / "traces"), output=str(runs / "run-1"), steps=0, eval_every=2,
                             promote_to=str(runs / "current"), sequences_per_step=1, holdout_tokens=40)
    result = module.run(cfg, FakeDraft(), tmp_path, log=lambda *a: None, stop=stop)
    assert result == {"base": 1.0, "best": {"step": 5, "tokens_per_round": 2.0}, "steps": 5, "stopped": True}
    evals = [e["step"] for e in events if e["event"] == "eval"]
    assert evals == [0, 2, 4, 5]  # periodic evaluations plus the final one after the stop
    assert saved == ["step-000002", "best", "step-000004", "step-000005", "best"]  # 1.2 at step 4 was not promoted
    assert os.readlink(runs / "current") == "run-1/best"
    manifest = json.loads((runs / "run-1" / "manifest.json").read_text())
    assert manifest["best"]["step"] == 5 and events[-1] == {"event": "end", "step": 5, "stopped": True, "best": {"step": 5, "tokens_per_round": 2.0}}


def test_stop_request_installs_and_restores_handlers():
    import signal

    from sharar.train.trainer import StopRequest
    previous = signal.getsignal(signal.SIGTERM)
    stop = StopRequest()
    stop.install()
    assert not stop.requested
    signal.raise_signal(signal.SIGTERM)
    assert stop.requested
    stop.restore()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_train_cli_continues_from_a_current_link(monkeypatch, tmp_path):
    from sharar import cli
    from sharar.train.trainer import promote_link
    runs = tmp_path / "runs"
    best = runs / "run-0" / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text("{}")
    (best / "model.safetensors").write_bytes(b"")
    promote_link(runs / "current", best)
    seen = {}
    monkeypatch.setattr("sharar.train.trainer.load_runtime", lambda cfg, log=print: (seen.__setitem__("cfg", cfg) or cfg, "pinned"))
    monkeypatch.setattr("sharar.train.trainer.run", lambda cfg, draft, pinned, stop=None: {"ok": True})
    assert cli.main(["train", "--traces", str(tmp_path), "--output", str(runs / "run-1"), "--steps", "0",
                     "--promote-to", str(runs / "current"), "--continue-from", str(runs / "current")]) == 0
    cfg = seen["cfg"]
    assert cfg.draft == str(best.resolve()) and cfg.steps == 0 and cfg.promote_to == str(runs / "current")
    assert cli.main(["train", "--traces", str(tmp_path), "--output", str(runs / "run-2"), "--continue-from", str(runs / "missing")]) == 0
    assert seen["cfg"].draft is None


def test_teacher_kl_matches_manual_computation():
    from sharar.train.loss import teacher_kl
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 50)
    ids = torch.stack([torch.randperm(50)[:4] for _ in range(6)]).view(2, 3, 4)
    t_logits = torch.randn(2, 3, 4)
    weights = torch.tensor([[0.0, 1.0, 0.5], [1.0, 1.0, 0.0]])
    num, den = teacher_kl(logits, ids, t_logits, weights)
    manual = 0.0
    for a in range(2):
        for b in range(3):
            p = torch.softmax(t_logits[a, b], -1)
            q = torch.softmax(logits[a, b][ids[a, b]], -1)
            manual += weights[a, b] * (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(num, manual, rtol=1e-5, atol=1e-6)
    assert den == weights.sum()
    # a student that matches the teacher on those ids has zero KL
    matched = torch.full((2, 3, 50), -30.0)
    matched.scatter_(-1, ids, t_logits)
    assert teacher_kl(matched, ids, t_logits, weights)[0].abs() < 1e-4


def test_block_teacher_aligns_rows_to_labels():
    from sharar.train.loss import block_teacher
    length = 10
    ids = torch.arange(length)[:, None].expand(length, 3).clone()  # row p carries value p
    logits = torch.zeros(length, 3)
    valid = torch.tensor([False, False, True, True, True, True, True, False, False, False])
    t_ids, t_logits, keep = block_teacher(torch.tensor([2, 6]), 4, ids, logits, valid)
    assert t_ids[0, 1:, 0].tolist() == [2, 3, 4]  # label at a+k uses row a+k-1
    assert keep[0].tolist() == [False, True, True, True]  # position 0 never; rows 2..4 valid
    assert keep[1].tolist() == [False, True, False, False]  # rows 6 valid, 7.. invalid
    assert t_ids.shape == (2, 4, 3) and t_logits.shape == (2, 4, 3)


def test_samples_expose_teacher_topk(tmp_path):
    sink = TraceSink(tmp_path)
    req = Request(prompt_ids=[1] * 5, params=SamplingParams(max_new_tokens=32), request_id=1)
    req.output_ids, req.finish_reason, req.t_finish = [2] * 20, "stop", time.perf_counter()
    req.trace_features = sink.recorder()
    req.trace_features.append(torch.zeros(24, 5120, dtype=torch.bfloat16), 0)
    req.trace_topk = sink.topk_recorder()
    ids = (torch.arange(19)[:, None] + torch.arange(16)[None, :]).to(torch.int32)
    req.trace_topk.append(ids, torch.ones(19, 16, dtype=torch.bfloat16), 5)  # rows for positions 5..23
    sink.record(req)
    deadline = time.time() + 5
    while sink.records < 1 and time.time() < deadline:
        time.sleep(0.01)
    sink.close()
    sample = load_samples(tmp_path, require_features=True)[0]
    assert sample.topk["start"] == 5 and sample.topk["rows"] == 19 and summary([sample])["with_topk"] == 1
    t_ids, t_logits, valid = sample.teacher("cpu")
    assert t_ids.shape == (25, 16) and valid.tolist() == [False] * 5 + [True] * 19 + [False]
    assert t_ids[5, 0] == 0 and t_ids[23, 0] == 18 and t_logits[6, 3] == 1.0
    cut = load_samples(tmp_path, max_length=20, require_features=True)[0]  # 5 positions dropped
    assert cut.topk["start"] == 0 and cut.topk["skip"] == 0 and cut.teacher("cpu")[0][0, 0] == 0
    assert load_samples(tmp_path, max_length=15, min_output=8, require_features=True)[0].teacher("cpu")[0][0, 0] == 5


def test_parameter_subsets_select_expected_weights():
    from sharar.train.draft import trainable_key
    names = ["fc.weight", "hidden_norm.weight", "norm.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.kv",
             "layers.0.self_attn.k_norm.weight", "layers.0.mlp.gate_up", "layers.0.mlp.down_proj.weight",
             "layers.0.attention_conv.base_kernel", "candidate_selector.successor_codebook"]
    picked = lambda mode: {n for n in names if trainable_key(n, mode, False)}
    assert picked("all") == set(names) - {"fc.weight", "hidden_norm.weight", "layers.0.self_attn.kv", "layers.0.self_attn.k_norm.weight"}
    assert "layers.0.mlp.gate_up" not in picked("attn") and "layers.0.self_attn.q_proj.weight" in picked("attn") and "candidate_selector.successor_codebook" in picked("attn")
    assert picked("selector") == {"candidate_selector.successor_codebook"}
    with pytest.raises(ValueError):
        trainable_key("norm.weight", "mlp", False)
