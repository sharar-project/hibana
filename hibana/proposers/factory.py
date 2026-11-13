"""Checkpoint-specific construction at startup; the executor consumes contracts."""

from hibana.target import DEFAULT_TARGET

MAX_DEPTHS = {"mtp": 8, "dflash2": 15}


def draft_snapshot(proposer, draft_dir=None, target=None):
    """The target's pinned publisher snapshot, or a DFlash2 checkpoint directory (`--draft DIR`)."""
    import json
    from pathlib import Path
    target = target or DEFAULT_TARGET
    name, revision = target.draft_pin(proposer)
    if draft_dir is None:
        # In the target's cache or its hub/, where the launcher downloads it (launch/preflight.py checks both)
        root = target.cache_root()
        snapshots = [base / ("models--" + name.replace("/", "--")) / "snapshots" / revision for base in (root, root / "hub")]
        return next((snap for snap in snapshots if (snap / "config.json").is_file()), snapshots[0])
    directory = Path(draft_dir)
    if not (directory / "config.json").is_file() or not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"draft directory {directory} needs config.json and model.safetensors")
    wanted = ("DFlash2DraftModel", "LiLiCorrDraftModel")
    architectures = json.loads((directory / "config.json").read_text()).get("architectures")
    if not architectures or len(architectures) != 1 or architectures[0] not in wanted:
        raise ValueError(f"draft directory {directory} holds {architectures}, expected one of {wanted} for {proposer}")
    return directory


def create(proposer, k, model, context, num_pages, *, draft_precision='bf16', draft_dir=None, padding_rows=0):
    if proposer == "mtp":
        from hibana.model.mtp import MtpHead
        from hibana.proposers.mtp import MtpProposer
        from hibana.weights.safetensors import SnapshotWeights
        w = SnapshotWeights(model.target.snapshot_dir())
        try:
            if model.target.family == "flashnext":
                from hibana.model.flashnext.mtp import FlashNextMtpHead
                head = FlashNextMtpHead(w, context.device, model, num_pages, padding_rows=padding_rows)
            elif model.target.family == "nemotron":
                from hibana.model.nemotron.mtp import NemotronMtpHead
                head = NemotronMtpHead(w, context.device, model, num_pages, padding_rows=padding_rows)
            else:
                head = MtpHead(w, context.device, model, num_pages, padding_rows=padding_rows)
        finally:
            w.close()
        head.target = getattr(model, "full_target", model.target)  # a tensor-parallel draft is whole on every rank
        from hibana.engine.config import current
        cfg = current()
        vocab = cfg.draft_vocab  # reduced draft vocabulary rows (0: the full lm_head)
        fmt = cfg.draft_head  # the draft head's own precision: fp8 (the target's head), w3 or w2
        if (vocab or fmt != "fp8") and not getattr(head, "DRAFT_HEADS", False):
            raise ValueError("HIBANA_DRAFT_VOCAB / HIBANA_DRAFT_HEAD need the Flash-Next MTP head")
        if vocab or fmt != "fp8":
            from hibana.proposers.draft_vocab import draft_vocab
            ids = draft_vocab(model.target.snapshot_dir(), vocab, head.lm_head.out_features) if vocab else None
            head.set_draft_head(ids, fmt, cfg.draft_head_rescore)  # shortlist rescored by the target's head
        adapter = MtpProposer(head, context, k)
    elif proposer == "dflash2":
        import json
        from hibana.model.parallel_draft import ParallelDraftModel
        from hibana.proposers.parallel import ParallelProposer
        snapshot = draft_snapshot(proposer, draft_dir, model.target)
        cls = ParallelDraftModel
        if json.loads((snapshot / 'config.json').read_text())['architectures'] == ['LiLiCorrDraftModel']:
            from hibana.model.lilicorr import LiLiCorrDraftModel
            cls = LiLiCorrDraftModel
        adapter = ParallelProposer(cls(snapshot, model, context.max_context, precision=draft_precision), context, k, proposer)
        calibration = dict(model.target.draft_calibration).get(proposer)
        if draft_dir is None and calibration is not None:  # the pinned draft's own fit; --draft checkpoints score raw
            adapter.calibration = calibration
    else:
        raise ValueError(f"unknown proposer {proposer!r}")
    return adapter
