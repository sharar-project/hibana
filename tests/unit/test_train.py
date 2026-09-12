"""Draft training pieces that run on the CPU: objective, labels, data, mask, export.

The objective is checked against the naive formulas SpecForge's own test suite
uses as ground truth (``tests/test_utils/test_dflash_losses.py``:
``_targets_and_mask``, ``_neg_log_q``, ``_naive_dflash_loss``), vendored below.
"""

import json
import time
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


def test_terms_add_and_metrics_are_finite():
    logits = torch.randn(2, 3, 9)
    ids = torch.arange(12)
    t = block_targets(ids, torch.tensor([0, 4]), 3, 0, 7.0)
    a = block_terms(logits, torch.zeros(2, 3, 2), *t, top_k=3, hidden_projection=None, predecessor_codebook=None, successor_codebook=None)
    total = a + a
    assert total.ce_num == 2 * a.ce_num and total.loss(1.0) == pytest.approx(a.loss(1.0).item())
    assert all(v == v for v in total.metrics().values())


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
    train, holdout = split(samples, 0.2)
    assert [s.request_id for s in holdout] == [6] and len(train) == 4
    assert split(samples, 0.0)[1] == [samples[-1]]  # at least one held out
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
