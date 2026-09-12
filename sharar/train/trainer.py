"""Fine-tune the DFlash2 drafter on request traces and promote on held-out acceptance.

One step: a few traced sequences with the projected context features recorded
while they were served, random anchors per sequence, the block objective from
``loss.py`` accumulated over anchor chunks with one shared denominator, AdamW
on FP32 masters with gradient clipping. Evaluation replays serving's greedy
proposals along the held-out responses and reports tokens per round. A
checkpoint is promoted to ``best/`` only when held-out tokens per round beat
the previous best, which starts at the untrained drafter. The target is not
loaded: only its embeddings and vocabulary head are needed.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from sharar.train.data import Sample, load_samples, split, summary
from sharar.train.loss import Terms, block_targets, block_terms, sample_anchors


@dataclass
class TrainConfig:
    traces: str
    output: str
    draft: str | None = None  # checkpoint directory to start from; default: the pinned publisher draft
    steps: int = 200
    sequences_per_step: int = 2
    anchors_per_sequence: int = 128
    anchor_chunk: int = 32
    block: int = 16  # serving verifies k=15 proposals, one block of 16
    gamma: float = 7.0
    selector_alpha: float = 1.0
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 8192
    holdout_fraction: float = 0.1
    eval_every: int = 50
    eval_sequences: int = 32
    greedy_only: bool = False
    train_context: bool = False
    seed: int = 0
    log_every: int = 1
    tags: dict = field(default_factory=dict)


def learning_rate(cfg: TrainConfig, step: int) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    return cfg.learning_rate


class Trainer:
    def __init__(self, cfg: TrainConfig, draft, log=print):
        self.cfg, self.draft, self.log = cfg, draft, log
        self.device = draft.model.device
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95),
                                           weight_decay=cfg.weight_decay, foreach=True)
        self.output = Path(cfg.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history = self.output / "log.jsonl"

    def _record(self, row: dict) -> None:
        with self.history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def denominator(self, samples: list[Sample], anchor_sets: list[torch.Tensor]) -> torch.Tensor:
        den = torch.zeros((), device=self.device)
        for sample, anchors in zip(samples, anchor_sets):
            ids = torch.tensor(sample.ids, dtype=torch.int64, device=self.device)
            den = den + block_targets(ids, anchors.to(self.device), self.cfg.block, sample.prompt_len, self.cfg.gamma)[3].sum()
        return den.clamp_min(torch.finfo(torch.float32).tiny)

    # ---- one step ----
    def step(self, samples: list[Sample], step: int) -> dict:
        cfg = self.cfg
        started = time.perf_counter()
        # Anchors are drawn once so the shared denominator matches the anchors trained on.
        anchor_sets = [sample_anchors(s.prompt_len, s.length, cfg.anchors_per_sequence, self.generator) for s in samples]
        total_den = self.denominator(samples, anchor_sets)
        self.optimizer.zero_grad(set_to_none=True)
        total = None
        for sample, anchors in zip(samples, anchor_sets):
            terms = self._sequence_terms_with_anchors(sample, anchors, total_den)
            total = terms if total is None else total + terms
        grad_norm = torch.nn.utils.clip_grad_norm_(self.draft.parameters(), cfg.max_grad_norm, foreach=True)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(cfg, step)
        self.optimizer.step()
        self.draft.sync()
        row = {"step": step, "loss": total.loss(cfg.selector_alpha).item(), "grad_norm": float(grad_norm),
               "lr": learning_rate(cfg, step), "labels": total.label_count.item(), "seconds": time.perf_counter() - started,
               **total.metrics()}
        return row

    def _sequence_terms_with_anchors(self, sample: Sample, anchors: torch.Tensor, total_den: torch.Tensor) -> Terms:
        cfg = self.cfg
        ids = torch.tensor(sample.ids, dtype=torch.int64, device=self.device)
        projected = sample.context(self.device)
        anchors = anchors.to(self.device)
        hidden_projection, predecessor_codebook, successor_codebook = self.draft.selector_weights()
        total = None
        for start in range(0, anchors.shape[0], cfg.anchor_chunk):
            chunk = anchors[start:start + cfg.anchor_chunk]
            target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(ids, chunk, cfg.block, sample.prompt_len, cfg.gamma)
            hidden = self.draft.forward_blocks(projected, ids, chunk)
            logits = self.draft.head(hidden)
            terms = block_terms(logits, hidden, target_ids, predecessor_ids, weight_mask, loss_weights, top_k=self.draft.top_k,
                                hidden_projection=hidden_projection, predecessor_codebook=predecessor_codebook,
                                successor_codebook=successor_codebook)
            ((terms.ce_num + cfg.selector_alpha * terms.selector_ce_num) / total_den).backward()
            detached = Terms(*(t.detach() for t in terms.__dict__.values()))
            total = detached if total is None else total + detached
            del hidden, logits, terms
        return total

    # ---- evaluation ----
    @torch.no_grad()
    def evaluate(self, samples: list[Sample]) -> dict:
        """Teacher-forced serving replay: greedy proposals from each round's
        anchor, accepted prefix plus one correction, along the held-out output."""
        cfg = self.cfg
        k = cfg.block - 1
        rounds = emitted = accepted = 0
        per_sample = []
        started = time.perf_counter()
        for sample in samples:
            ids = torch.tensor(sample.ids, dtype=torch.int64, device=self.device)
            projected = sample.context(self.device)
            position, sample_rounds, sample_emitted = sample.prompt_len, 0, 0
            while position < sample.length - 1:
                proposals = self.draft.propose(projected, ids, torch.tensor([position], device=self.device))[0]
                truth = ids[position + 1:position + 1 + k]
                matched = 0
                for a, b in zip(proposals.tolist(), truth.tolist()):
                    if a != b:
                        break
                    matched += 1
                count = min(matched + 1, sample.length - 1 - position)
                position += count
                sample_rounds += 1
                sample_emitted += count
                accepted += min(matched, count)
            rounds += sample_rounds
            emitted += sample_emitted
            per_sample.append({"id": sample.request_id, "rounds": sample_rounds, "emitted": sample_emitted})
        return {"sequences": len(samples), "rounds": rounds, "emitted": emitted, "accepted": accepted,
                "tokens_per_round": emitted / rounds if rounds else None, "seconds": time.perf_counter() - started,
                "per_sample": per_sample}


def run(cfg: TrainConfig, draft, snapshot: Path, log=print) -> dict:
    samples = load_samples(cfg.traces, max_length=cfg.max_length, greedy_only=cfg.greedy_only, require_features=True)
    if len(samples) < 2:
        raise ValueError(f"need at least two traced requests with recorded features, found {len(samples)}")
    train_set, holdout = split(samples, cfg.holdout_fraction)
    holdout_eval = holdout[-cfg.eval_sequences:]
    trainer = Trainer(cfg, draft, log)
    manifest = {"config": asdict(cfg), "train": summary(train_set), "holdout": summary(holdout),
                "trainable_parameters": draft.parameter_count(), "base": cfg.draft or "pinned"}
    log(f"train {manifest['train']['requests']} requests / {manifest['train']['output_tokens']} output tokens; "
        f"holdout {manifest['holdout']['requests']} requests; trainable {draft.parameter_count() / 1e9:.2f}B")
    base = trainer.evaluate(holdout_eval)
    log(f"holdout tokens/round before training: {base['tokens_per_round']:.3f} ({base['rounds']} rounds, {base['seconds']:.0f}s)")
    trainer._record({"event": "eval", "step": 0, **{k: v for k, v in base.items() if k != "per_sample"}})
    best = {"step": 0, "tokens_per_round": base["tokens_per_round"]}
    (trainer.output / "manifest.json").write_text(json.dumps({**manifest, "base_eval": base, "best": best}, indent=2))
    order = list(range(len(train_set)))
    rng = random.Random(cfg.seed)
    rng.shuffle(order)
    cursor = 0
    for step in range(1, cfg.steps + 1):
        batch = []
        for _ in range(cfg.sequences_per_step):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch.append(train_set[order[cursor]])
            cursor += 1
        row = trainer.step(batch, step)
        trainer._record({"event": "step", **row})
        if step % cfg.log_every == 0:
            log(f"step {step}: loss {row['loss']:.4f} ce {row['ce']:.4f} acc {row['accuracy']:.3f} "
                f"sel_acc {row['selector_accuracy']:.3f} grad {row['grad_norm']:.2f} {row['seconds']:.1f}s")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            result = trainer.evaluate(holdout_eval)
            log(f"holdout tokens/round at step {step}: {result['tokens_per_round']:.3f} (best {best['tokens_per_round']:.3f} at step {best['step']})")
            trainer._record({"event": "eval", "step": step, **{k: v for k, v in result.items() if k != "per_sample"}})
            draft.save(trainer.output / f"step-{step:06d}", snapshot, {**manifest, "step": step, "eval": result})
            if result["tokens_per_round"] > best["tokens_per_round"]:
                best = {"step": step, "tokens_per_round": result["tokens_per_round"]}
                draft.save(trainer.output / "best", snapshot, {**manifest, "step": step, "eval": result, "promoted": True})
                log(f"promoted step {step} to best/")
            (trainer.output / "manifest.json").write_text(json.dumps({**manifest, "base_eval": base, "best": best}, indent=2))
    return {"base": base["tokens_per_round"], "best": best}


class HeadOnlyTarget:
    """The parts of the target the draft needs: embeddings and the NVFP4 vocabulary head."""

    def __init__(self, snap: Path, device="cuda"):
        from sharar.kernels import ops
        from sharar.model.qwen38 import PREFIX
        from sharar.weights.modelopt import load_nvfp4_linear
        from sharar.weights.safetensors import SnapshotWeights
        self.device = torch.device(device)
        weights = SnapshotWeights(snap)
        self.embed = weights.get(f"{PREFIX}.embed_tokens.weight").to(self.device)
        self.lm_head = load_nvfp4_linear(weights, ["lm_head"], self.device)
        self.weights = weights
        self._ops = ops

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._ops.nvfp4_linear(hidden, self.lm_head)


def pinned_draft_dir(snap: Path) -> Path:
    from sharar.proposers.parallel import PINS
    name, revision = PINS["dflash2"]
    return snap.parents[2] / ("models--" + name.replace("/", "--")) / "snapshots" / revision


def load_runtime(cfg: TrainConfig, log=print):
    """BF16 draft (pinned or from a checkpoint) over a head-only target; returns (draft, pinned draft dir)."""
    from sharar import config
    from sharar.model.parallel_draft import ParallelDraftModel
    from sharar.train.draft import Head, TrainableDraft

    if cfg.train_context:
        raise ValueError("training the feature and K/V projections needs raw target features; traces record the projected context only")
    snap = config.snapshot_dir()
    target = HeadOnlyTarget(snap)
    pinned = pinned_draft_dir(snap)
    draft_dir = Path(cfg.draft) if cfg.draft else pinned
    model = ParallelDraftModel(draft_dir, target, config.CONTEXT_LENGTH, precision="bf16")
    head = Head(target, target.weights)
    draft = TrainableDraft(model, head, block=cfg.block, train_context=cfg.train_context)
    log(f"draft {draft_dir}: {draft.parameter_count() / 1e9:.2f}B trainable parameters, head-only target")
    return draft, pinned
