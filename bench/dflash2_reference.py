"""Check the training forward and objective against the serving path, the
publisher's reference model and SpecForge, then run a short training smoke
on recorded traces.

Runs inside the runtime container on an exclusive GPU. Optional reference
sources are mounted read-only:

* ``--oracle-source DIR``: the publisher's ``model.py`` (DFlash2DraftModel);
  ``--oracle-python DIR`` supplies its transformers.
* ``--specforge DIR``: a SpecForge checkout; its ``CandidateSelector`` scoring,
  ``create_dflash_sdpa_mask`` and the naive loss formulas from its tests.

``--traces`` must hold traces recorded with features (a server started with
``--traces``). Every comparison is recorded in ``--output/report.json``;
failed checks are listed and make the exit status nonzero. Nothing here is a
serving change.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def relative(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    error = actual.float() - expected.float()
    return {"relative_rms": (error.square().mean() / expected.float().square().mean().clamp_min(1e-30)).sqrt().item(),
            "max_abs": error.abs().max().item(),
            "cosine": F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--oracle-source", type=Path, default=None)
    ap.add_argument("--oracle-python", type=Path, default=None)
    ap.add_argument("--specforge", type=Path, default=None)
    ap.add_argument("--smoke-steps", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.time()}
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)
            print("CHECK FAILED:", what, flush=True)

    from sharar import config
    from sharar.engine.contracts import StepInputs
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    from sharar.kernels import ops
    from sharar.model.qwen38 import Qwen38
    from sharar.train.data import load_samples
    from sharar.train.draft import Head, TrainableDraft
    from sharar.train.features import FeatureExtractor
    from sharar.train.loss import block_targets, block_terms, sample_anchors
    from sharar.train.trainer import TrainConfig, Trainer, pinned_draft_dir
    from sharar.weights.safetensors import SnapshotWeights

    snap = config.snapshot_dir()
    pinned = pinned_draft_dir(snap)
    target = Qwen38(snap, log=print)
    serving = Executor(target, num_pages=35000, num_slots=12, use_cache=False)
    serving.enable_spec(15, proposer="dflash2", state_strategy="reconstruct", draft_precision="bf16")
    plain = Executor(target, num_pages=20000, num_slots=4, use_cache=False)
    adapter = serving.spec.proposer
    model = adapter.model
    device = target.device
    # The training draft shares the serving adapter's weight dictionary.
    draft = TrainableDraft(model, Head(target, SnapshotWeights(snap)), block=16)
    extractor = FeatureExtractor(plain, model.features)

    samples = load_samples(args.traces, max_length=4096, require_features=True)
    check(len(samples) >= 2, f"traces with features: {len(samples)}")
    short = [s for s in samples if s.length <= 1500]
    sample = short[0] if short else samples[0]
    print(f"reference sample: request {sample.request_id}, {sample.length} tokens, prompt {sample.prompt_len}", flush=True)
    ids = torch.tensor(sample.ids, dtype=torch.int64, device=device)
    recorded = sample.context(device)

    # ---- 1. head: NVFP4 kernel vs dequantized BF16 weight on real draft hidden states ----
    with torch.no_grad():
        probe_anchors = sample_anchors(sample.prompt_len, sample.length, 8, torch.Generator().manual_seed(0)).to(device)
        hidden = draft.forward_blocks(recorded, ids, probe_anchors)[:, 1:].reshape(-1, 5120)
        kernel = ops.nvfp4_linear(hidden, target.lm_head)
        dense = hidden @ draft.head.dequantized.T
        report["head_dequant"] = relative(kernel, dense)
        report["head_dequant"]["argmax_agreement"] = (kernel.argmax(-1) == dense.argmax(-1)).float().mean().item()
    print("head", report["head_dequant"], flush=True)
    check(report["head_dequant"]["cosine"] > 0.99 and report["head_dequant"]["argmax_agreement"] > 0.9, f"head {report['head_dequant']}")

    # ---- 2. training forward vs the serving proposer on the proposer's own ring features ----
    prompt = list(sample.ids[:sample.prompt_len])
    with torch.inference_mode():
        req = serving.admit(prompt, SamplingParams(max_new_tokens=16, ignore_eos=True))
        serving.prefill(req)
        native = {}
        native_forward = model.forward

        def capture(*a):
            out = native_forward(*a)
            native["tokens"], native["hidden"] = a[0].clone(), out.clone()
            return out
        model.forward = capture
        inputs = StepInputs(torch.tensor([req.output_ids[-1]], device=device), torch.tensor([req.committed_len], device=device),
                            torch.tensor([req.slot], dtype=torch.int32, device=device))
        proposal = adapter.propose([req], None, inputs, graphs=False)
        model.forward = native_forward
        first = int(req.output_ids[-1])
        ring = adapter.projected_rows(req, max(0, len(prompt) - adapter.window), len(prompt)).clone()
    ids_serving = torch.tensor(prompt + [first], dtype=torch.int64, device=device)
    with torch.no_grad():
        projected = torch.zeros((len(prompt) + 1, 5120), dtype=torch.bfloat16, device=device)
        projected[len(prompt) - ring.shape[0]:len(prompt)] = ring
        anchor = torch.tensor([len(prompt)], device=device)
        mine = draft.forward_blocks(projected, ids_serving, anchor)[0]
        report["serving_forward"] = relative(mine, native["hidden"][0])
        report["serving_forward"]["tokens_match"] = bool(torch.equal(native["tokens"][0], torch.cat([ids_serving[-1:], torch.full((15,), 248070, device=device)])))
        proposals = draft.propose(projected, ids_serving, anchor)[0]
        # A chain diverges for good after one near-tie flip, so also score each
        # step teacher-forced on serving's own path: given serving's previous
        # token, does the training selector pick serving's next token?
        forced = draft.forced_agreement(projected, ids_serving, anchor, proposal.tokens[0])
        report["serving_proposals"] = {"mine": proposals.tolist(), "serving": proposal.tokens[0].tolist(),
                                       "agreement": (proposals == proposal.tokens[0]).float().mean().item(),
                                       "teacher_forced_agreement": forced}
        # the recorded trace of this request must hold the same rows the ring holds now
        recorded_prompt = recorded[max(0, len(prompt) - adapter.window):len(prompt)]
        report["recorded_vs_ring"] = relative(recorded_prompt, ring)
    print("serving forward", report["serving_forward"], "proposal agreement", report["serving_proposals"]["agreement"], flush=True)
    print("recorded vs ring", report["recorded_vs_ring"], flush=True)
    check(report["serving_forward"]["relative_rms"] < 0.02, f"serving forward {report['serving_forward']}")
    check(report["serving_proposals"]["teacher_forced_agreement"] >= 0.8, f"proposals {report['serving_proposals']}")
    check(report["recorded_vs_ring"]["relative_rms"] < 0.02, f"recorded features {report['recorded_vs_ring']}")
    with torch.inference_mode():
        serving.release(req)

    # ---- 3. publisher oracle on recomputed raw features (same inputs to both models) ----
    if args.oracle_source is not None:
        if args.oracle_python is not None:
            sys.path.insert(0, str(args.oracle_python))
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RotaryEmbedding
        from safetensors.torch import load_file
        spec = importlib.util.spec_from_file_location("published_draft_oracle", args.oracle_source / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        oracle_cfg = Qwen3Config(**json.loads((pinned / "config.json").read_text()))
        with torch.device("meta"):
            oracle = module.DFlash2DraftModel(oracle_cfg)
        weights = load_file(pinned / "model.safetensors")
        for key in ("predecessor_codebook", "successor_codebook"):
            weights[f"candidate_selector.{key}.weight"] = weights.pop(f"candidate_selector.{key}")
        result = oracle.load_state_dict(weights, strict=False, assign=True)
        check(not result.missing_keys, f"oracle keys {result}")
        oracle.rotary_emb = Qwen3RotaryEmbedding(oracle_cfg, device="cuda")
        oracle = oracle.to(device="cuda", dtype=torch.bfloat16).eval()
        anchor_pos = sample.prompt_len
        raw_features = extractor(prompt)  # context = every position before the anchor; window is index-based, so keep it short
        block_ids = torch.cat([ids[anchor_pos:anchor_pos + 1], torch.full((15,), 248070, device=device)])
        with torch.no_grad():
            raw = torch.cat([raw_features[i] for i in model.features], dim=-1)[None]
            expected = oracle(position_ids=torch.arange(anchor_pos + 16, device=device)[None],
                              noise_embedding=F.embedding(block_ids[None], target.embed), target_hidden=raw, use_cache=False)[0]
            projected_raw = torch.zeros((sample.length, 5120), dtype=torch.bfloat16, device=device)
            projected_raw[:anchor_pos] = draft.project_context(raw_features)
            mine = draft.forward_blocks(projected_raw, ids, torch.tensor([anchor_pos], device=device))[0]
            report["oracle_forward"] = relative(mine, expected)
            report["oracle_forward"]["anchor"] = anchor_pos
        probe = draft.forward_blocks(projected_raw, ids, torch.tensor([anchor_pos], device=device))
        probe.float().square().mean().backward()
        with_grad = {name for name, p in draft.masters.items() if p.grad is not None}
        backbone = {name for name in draft.masters if not name.startswith("candidate_selector.")}
        report["oracle_forward"]["parameters_with_grad"] = len(with_grad)
        for p in draft.parameters():
            p.grad = None
        del oracle, weights
        torch.cuda.empty_cache()
        print("oracle forward", report["oracle_forward"], flush=True)
        check(report["oracle_forward"]["relative_rms"] < 0.02, f"oracle forward {report['oracle_forward']}")
        check(with_grad == backbone, f"gradients reach every backbone parameter: missing {sorted(backbone - with_grad)[:5]}")

    # ---- 4. objective vs SpecForge ----
    if args.specforge is not None:
        sys.path.insert(0, str(args.specforge))
        sys.path.insert(0, str(args.specforge / "tests"))
        checks: dict = {}
        try:
            from specforge.modeling.draft.dflash2 import CandidateSelector
            from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
            checks["imports"] = "specforge"
        except Exception as exc:  # noqa: BLE001 - report which reference was reachable
            CandidateSelector = create_dflash_sdpa_mask = None
            checks["imports"] = f"unavailable: {type(exc).__name__}: {exc}"
        try:
            from test_utils.test_dflash_losses import _naive_dflash_loss, _neg_log_q, _targets_and_mask
            checks["naive_loss_source"] = "specforge tests"
        except Exception as exc:  # noqa: BLE001
            checks["naive_loss_source"] = f"unavailable: {type(exc).__name__}: {exc}"
            _naive_dflash_loss = _neg_log_q = _targets_and_mask = None
        check(CandidateSelector is not None and _naive_dflash_loss is not None, f"specforge references importable: {checks}")
        anchors = sample_anchors(sample.prompt_len, sample.length, 8, torch.Generator().manual_seed(1)).to(device)
        target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(ids, anchors, 16, sample.prompt_len, 7.0)
        with torch.no_grad():
            hidden = draft.forward_blocks(recorded, ids, anchors)
            logits = draft.head(hidden).float()
            hp, pc, sc = draft.selector_weights()
            terms = block_terms(logits, hidden, target_ids, predecessor_ids, weight_mask, loss_weights, top_k=16,
                                hidden_projection=hp, predecessor_codebook=pc, successor_codebook=sc)
            if _naive_dflash_loss is not None:
                loss_mask = (torch.arange(sample.length) >= sample.prompt_len).double()[None]
                want_t, want_m = _targets_and_mask(ids.cpu()[None], loss_mask, anchors.cpu()[None], torch.ones(1, anchors.shape[0], dtype=torch.bool), 16)
                check(torch.equal(want_t[0], target_ids.cpu()) and torch.equal(want_m[0].float(), weight_mask.cpu()), "labels and masks match SpecForge")
                want = _naive_dflash_loss(_neg_log_q(logits.double().cpu(), want_t), want_m, 7.0)
                mine_loss = (terms.ce_num / terms.loss_den).double()
                checks["dflash_loss"] = {"mine": mine_loss.item(), "specforge_naive": want.item(), "abs_diff": abs(mine_loss.item() - want.item())}
                check(checks["dflash_loss"]["abs_diff"] < 1e-4, f"dflash loss {checks['dflash_loss']}")
            if CandidateSelector is not None:
                selector = CandidateSelector(hidden_size=5120, vocab_size=248320, state_rank=256, top_k=16, initializer_range=0.02).to(device)
                with torch.no_grad():
                    selector.predecessor_codebook.copy_(pc.float())
                    selector.successor_codebook.copy_(sc.float())
                    selector.hidden_projection.weight.copy_(hp.float())
                unary, candidates = logits.topk(16, dim=-1)
                theirs = selector.score_candidates(candidate_ids=candidates, unary_logits=unary, hidden_states=hidden.float(), predecessor_ids=predecessor_ids)
                from sharar.train.loss import selector_scores
                ours = selector_scores(unary, candidates, hidden.float(), predecessor_ids, hp.float(), pc.float(), sc.float())
                checks["selector_scores"] = relative(ours, theirs)
                check(checks["selector_scores"]["relative_rms"] < 1e-4, f"selector scores {checks['selector_scores']}")
                matches = candidates.eq(target_ids[..., None])
                ce = F.cross_entropy(theirs.reshape(-1, 16), matches.long().argmax(-1).reshape(-1), reduction="none").reshape_as(loss_weights)
                want_sel = (ce * loss_weights * matches.any(-1).float()).sum()
                checks["selector_ce_num"] = {"mine": terms.selector_ce_num.item(), "specforge": want_sel.item()}
                check(abs(terms.selector_ce_num.item() - want_sel.item()) <= 1e-3 * max(1.0, want_sel.item()), f"selector ce {checks['selector_ce_num']}")
            if create_dflash_sdpa_mask is not None:
                T = sample.length
                keep = torch.ones(1, anchors.shape[0], dtype=torch.bool, device=device)
                full = create_dflash_sdpa_mask(anchors[None], keep, T, 16, device)[0, 0]
                sliding = create_dflash_sdpa_mask(anchors[None], keep, T, 16, device, sliding_window=2048)[0, 0]
                ours = draft.attention_mask(anchors, T)
                checks["mask"] = {
                    "context_matches_specforge_sliding": bool(torch.equal(ours[:, :T], sliding[:, :T])),
                    "block_matches_specforge_full": bool(torch.equal(ours[:, T:], full[:, T:])),
                    "block_matches_specforge_sliding": bool(torch.equal(ours[:, T:], sliding[:, T:])),
                    "note": "the publisher's config sets is_causal false, so the block is bidirectional as in serving; SpecForge's sliding mask is causal inside the block",
                }
                check(checks["mask"]["context_matches_specforge_sliding"] and checks["mask"]["block_matches_specforge_full"], f"mask {checks['mask']}")
        report["specforge"] = checks
        print("specforge", json.dumps(checks, indent=1), flush=True)

    # ---- 5. training smoke on recorded traces, then a reload of the saved checkpoint ----
    cfg = TrainConfig(traces=args.traces, output=str(args.output / "smoke"), steps=args.smoke_steps, sequences_per_step=1,
                      anchors_per_sequence=32, anchor_chunk=16, eval_every=args.smoke_steps, holdout_tokens=200, max_length=4096)
    trainer = Trainer(cfg, draft, log=print)
    before = trainer.evaluate(samples[:2])
    rows = []
    for step in range(1, args.smoke_steps + 1):
        row = trainer.step([samples[step % max(1, len(samples) - 2)]], step)
        rows.append(row)
        print("step", json.dumps(row), flush=True)
        check(row["loss"] == row["loss"] and row["grad_norm"] == row["grad_norm"], f"finite step {row}")
    after = trainer.evaluate(samples[:2])
    report["smoke"] = {"steps": rows, "eval_before": {k: v for k, v in before.items() if k != "per_sample"},
                       "eval_after": {k: v for k, v in after.items() if k != "per_sample"}}
    print("smoke eval before/after tokens per round:", before["tokens_per_round"], after["tokens_per_round"], flush=True)
    draft.save(args.output / "smoke-checkpoint", pinned, {"smoke": True})
    from sharar.model.parallel_draft import ParallelDraftModel
    reloaded = ParallelDraftModel(args.output / "smoke-checkpoint", target, 32768, precision="bf16")
    diffs = {name: (reloaded.weights[name].float() - model.weights[name].float()).abs().max().item() for name in model.weights}
    report["reload"] = {"max_abs_diff": max(diffs.values()), "keys": len(diffs)}
    check(report["reload"]["max_abs_diff"] == 0.0, f"reload {report['reload']}")
    # after training, the serving proposer (same weight dictionary) must propose what the training forward proposes
    with torch.inference_mode():
        req = serving.admit(prompt, SamplingParams(max_new_tokens=16, ignore_eos=True))
        serving.prefill(req)
        inputs = StepInputs(torch.tensor([req.output_ids[-1]], device=device), torch.tensor([req.committed_len], device=device),
                            torch.tensor([req.slot], dtype=torch.int32, device=device))
        trained_serving = adapter.propose([req], None, inputs, graphs=False).tokens[0]
        ring = adapter.projected_rows(req, max(0, len(prompt) - adapter.window), len(prompt)).clone()
        first = int(req.output_ids[-1])
        serving.release(req)
    with torch.no_grad():
        projected = torch.zeros((len(prompt) + 1, 5120), dtype=torch.bfloat16, device=device)
        projected[len(prompt) - ring.shape[0]:len(prompt)] = ring
        sequence = torch.tensor(prompt + [first], device=device)
        trained_mine = draft.propose(projected, sequence, torch.tensor([len(prompt)], device=device))[0]
        forced = draft.forced_agreement(projected, sequence, torch.tensor([len(prompt)], device=device), trained_serving)
    report["trained_proposals"] = {"chain_agreement": (trained_mine == trained_serving).float().mean().item(), "teacher_forced_agreement": forced}
    check(forced >= 0.8, f"trained proposals {report['trained_proposals']}")
    report["finished"] = time.time()
    report["failures"] = failures
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if failures:
        print("REFERENCE CHECKS FAILED:", failures, flush=True)
        return 1
    print("REFERENCE CHECKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
