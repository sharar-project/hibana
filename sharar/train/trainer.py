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
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sharar.train.data import Sample, load_samples, split, summary
from sharar.train.loss import Terms, block_targets, block_teacher, block_terms, sample_anchors


@dataclass
class TrainConfig:
    traces: str
    output: str
    draft: str | None = None  # checkpoint directory to start from; default: the pinned publisher draft
    steps: int = 200  # 0: until stopped (SIGTERM/SIGINT), then a final evaluation
    promote_to: str | None = None  # symlink updated to point at the promoted checkpoint (relative target)
    sequences_per_step: int = 2
    anchors_per_sequence: int = 128
    anchor_chunk: int = 128
    compile: bool = True  # torch.compile the block forward (about a minute of warmup; falls back to eager)
    block: int = 16  # serving verifies k=15 proposals, one block of 16
    gamma: float = 7.0
    selector_alpha: float = 1.0
    kl_alpha: float = 1.0  # weight of the KL to the target's recorded top-k; ignored for traces without it
    train_params: str = "attn"  # attn | all | selector (see draft.PARAMETER_SETS)
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 8192
    holdout_tokens: int = 20_000  # newest requests held out until they cover this many output tokens
    eval_every: int = 50
    greedy_only: bool = False
    train_context: bool = False
    seed: int = 0
    log_every: int = 1


def promote_link(link: Path, target: Path) -> None:
    """Atomically point ``link`` at ``target`` with a relative path, so the
    link resolves both inside the container and on the host mount."""
    link = Path(link)
    relative = os.path.relpath(target, link.parent)
    temporary = link.parent / f".{link.name}.tmp"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(relative)
    temporary.replace(link)


def resolve_draft(link: Path | None) -> Path | None:
    """The checkpoint a ``current`` link points at, if it is a complete checkpoint."""
    if link is None or not Path(link).is_symlink():
        return None
    target = Path(link).resolve()
    return target if (target / "config.json").is_file() and (target / "model.safetensors").is_file() else None


class StopRequest:
    """Set by SIGTERM/SIGINT; the loop finishes the current step, evaluates once more and exits."""

    def __init__(self):
        self.requested = False
        self._previous = {}

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, frame) -> None:
        self.requested = True

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


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
        self.forward_blocks = draft.forward_blocks
        if cfg.compile and self.device.type == "cuda":
            try:
                self.forward_blocks = torch.compile(draft.forward_blocks, dynamic=True)
                log("block forward: torch.compile (dynamic shapes); first steps include compilation")
            except Exception as exc:  # noqa: BLE001 - compile support varies by stack
                log(f"torch.compile unavailable ({type(exc).__name__}); training eager")

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
        row = {"step": step, "loss": total.loss(cfg.selector_alpha, cfg.kl_alpha).item(), "grad_norm": float(grad_norm),
               "lr": learning_rate(cfg, step), "labels": total.label_count.item(), "seconds": time.perf_counter() - started,
               **total.metrics()}
        return row

    def _sequence_terms_with_anchors(self, sample: Sample, anchors: torch.Tensor, total_den: torch.Tensor) -> Terms:
        cfg = self.cfg
        ids = torch.tensor(sample.ids, dtype=torch.int64, device=self.device)
        projected = sample.context(self.device)
        anchors = anchors.to(self.device)
        hidden_projection, predecessor_codebook, successor_codebook = self.draft.selector_weights()
        teacher_ids, teacher_logits, teacher_valid = sample.teacher(self.device) if cfg.kl_alpha and sample.topk else (None, None, None)
        total = None
        for start in range(0, anchors.shape[0], cfg.anchor_chunk):
            chunk = anchors[start:start + cfg.anchor_chunk]
            target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(ids, chunk, cfg.block, sample.prompt_len, cfg.gamma)
            teacher = block_teacher(chunk, cfg.block, teacher_ids, teacher_logits, teacher_valid) if teacher_ids is not None else None
            hidden = self.forward_blocks(projected, ids, chunk)
            logits = self.draft.head(hidden)
            terms = block_terms(logits, hidden, target_ids, predecessor_ids, weight_mask, loss_weights, top_k=self.draft.top_k,
                                hidden_projection=hidden_projection, predecessor_codebook=predecessor_codebook,
                                successor_codebook=successor_codebook, teacher=teacher)
            (terms.loss(cfg.selector_alpha, cfg.kl_alpha) * terms.loss_den / total_den).backward()
            detached = terms.detached()
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


def run(cfg: TrainConfig, draft, snapshot: Path, log=print, stop: StopRequest | None = None) -> dict:
    samples = load_samples(cfg.traces, max_length=cfg.max_length, greedy_only=cfg.greedy_only, require_features=True)
    if len(samples) < 2:
        raise ValueError(f"need at least two traced requests with recorded features, found {len(samples)}")
    train_set, holdout = split(samples, cfg.holdout_tokens)
    holdout_eval = holdout
    trainer = Trainer(cfg, draft, log)
    manifest = {"config": asdict(cfg), "train": summary(train_set), "holdout": summary(holdout),
                "trainable_parameters": draft.parameter_count(), "base": cfg.draft or "pinned"}
    log(f"train {manifest['train']['requests']} requests / {manifest['train']['output_tokens']} output tokens "
        f"({manifest['train']['with_topk']} with target top-k); holdout {manifest['holdout']['requests']} requests; "
        f"trainable {draft.parameter_count() / 1e9:.2f}B ({cfg.train_params})")
    base = trainer.evaluate(holdout_eval)
    log(f"holdout tokens/round before training: {base['tokens_per_round']:.3f} ({base['rounds']} rounds, {base['seconds']:.0f}s)")
    trainer._record({"event": "eval", "step": 0, **{k: v for k, v in base.items() if k != "per_sample"}})
    best = {"step": 0, "tokens_per_round": base["tokens_per_round"]}

    def write_manifest():
        (trainer.output / "manifest.json").write_text(json.dumps({**manifest, "base_eval": base, "best": best}, indent=2))

    def evaluate_and_promote(step: int) -> dict:
        nonlocal best
        result = trainer.evaluate(holdout_eval)
        log(f"holdout tokens/round at step {step}: {result['tokens_per_round']:.3f} (best {best['tokens_per_round']:.3f} at step {best['step']})")
        trainer._record({"event": "eval", "step": step, **{k: v for k, v in result.items() if k != "per_sample"}})
        draft.save(trainer.output / f"step-{step:06d}", snapshot, {**manifest, "step": step, "eval": result})
        if result["tokens_per_round"] > best["tokens_per_round"]:
            best = {"step": step, "tokens_per_round": result["tokens_per_round"]}
            draft.save(trainer.output / "best", snapshot, {**manifest, "step": step, "eval": result, "promoted": True})
            log(f"promoted step {step} to best/")
            if cfg.promote_to:
                promote_link(Path(cfg.promote_to), trainer.output / "best")
                log(f"{cfg.promote_to} -> {trainer.output / 'best'}")
        write_manifest()
        return result

    write_manifest()
    order = list(range(len(train_set)))
    rng = random.Random(cfg.seed)
    rng.shuffle(order)
    cursor = 0
    step = 0
    stopped = False
    while cfg.steps <= 0 or step < cfg.steps:
        if stop is not None and stop.requested:
            stopped = True
            break
        step += 1
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
            evaluate_and_promote(step)
    if stopped or (cfg.steps <= 0 and step > 0):
        if step % cfg.eval_every != 0:
            log(f"stop requested after step {step}: final evaluation")
            evaluate_and_promote(step)
    trainer._record({"event": "end", "step": step, "stopped": stopped, "best": best})
    return {"base": base["tokens_per_round"], "best": best, "steps": step, "stopped": stopped}


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
    draft = TrainableDraft(model, head, block=cfg.block, train_context=cfg.train_context, train_params=cfg.train_params)
    log(f"draft {draft_dir}: {draft.parameter_count() / 1e9:.2f}B trainable parameters ({cfg.train_params}), head-only target")
    return draft, pinned
