"""Checkpoint-specific construction at startup; the executor consumes contracts."""

from sharar.target import DEFAULT_TARGET

MAX_DEPTHS = {"mtp": 8, "dflash2": 15}


def draft_snapshot(proposer, draft_dir=None, target=None):
    """The pinned publisher snapshot, or a DFlash2 checkpoint directory (`--draft DIR`)."""
    import json
    from pathlib import Path
    from sharar.proposers.parallel import PINS
    name, revision = PINS[proposer]
    if draft_dir is None:
        root = (target or DEFAULT_TARGET).snapshot_dir().parents[2]
        return root / ("models--" + name.replace("/", "--")) / "snapshots" / revision
    directory = Path(draft_dir)
    if not (directory / "config.json").is_file() or not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"draft directory {directory} needs config.json and model.safetensors")
    wanted = "DFlash2DraftModel"
    architectures = json.loads((directory / "config.json").read_text()).get("architectures")
    if architectures != [wanted]:
        raise ValueError(f"draft directory {directory} holds {architectures}, expected [{wanted!r}] for {proposer}")
    return directory


def create(proposer, k, model, context, num_pages, *, draft_precision='bf16', draft_dir=None, padding_rows=0):
    if proposer == "mtp":
        from sharar.model.mtp import MtpHead
        from sharar.proposers.mtp import MtpProposer
        from sharar.weights.safetensors import SnapshotWeights
        w = SnapshotWeights(model.target.snapshot_dir())
        try:
            if model.target.family == "flashnext":
                from sharar.model.flashnext.mtp import FlashNextMtpHead
                head = FlashNextMtpHead(w, context.device, model, num_pages, padding_rows=padding_rows)
            else:
                head = MtpHead(w, context.device, model, num_pages, padding_rows=padding_rows)
        finally:
            w.close()
        head.target = getattr(model, "full_target", model.target)  # a tensor-parallel draft is whole on every rank
        from sharar.engine.config import current
        cfg = current()
        vocab = cfg.draft_vocab  # reduced draft vocabulary rows (0: the full lm_head)
        fmt = cfg.draft_head  # the draft head's own precision: fp8 (the target's head) or w3
        if (vocab or fmt != "fp8") and not getattr(head, "DRAFT_HEADS", False):
            raise ValueError("SHARAR_DRAFT_VOCAB / SHARAR_DRAFT_HEAD need the Flash-Next MTP head")
        if vocab or fmt != "fp8":
            from sharar.proposers.draft_vocab import draft_vocab
            ids = draft_vocab(model.target.snapshot_dir(), vocab, head.lm_head.out_features) if vocab else None
            head.set_draft_head(ids, fmt, cfg.draft_head_rescore)  # shortlist rescored by the target's head
        adapter = MtpProposer(head, context, k)
    elif proposer == "dflash2":
        from sharar.model.parallel_draft import ParallelDraftModel
        from sharar.proposers.parallel import ParallelProposer
        snapshot = draft_snapshot(proposer, draft_dir, model.target)
        adapter = ParallelProposer(ParallelDraftModel(snapshot, model, context.max_context, precision=draft_precision), context, k, proposer)
    else:
        raise ValueError(f"unknown proposer {proposer!r}")
    return adapter
