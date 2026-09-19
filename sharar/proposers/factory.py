"""Checkpoint-specific construction at startup; the executor consumes contracts."""
from sharar.target import DEFAULT_TARGET

MAX_DEPTHS = {"mtp": 4, "ngram": 4, "dspark": 7, "dflash2": 15}


def draft_snapshot(proposer, draft_dir=None, target=None):
    """The pinned publisher snapshot, or a trained checkpoint directory saved by `sharar train`."""
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
    wanted = "DFlash2DraftModel" if proposer == "dflash2" else "DSparkDraftModel"
    architectures = json.loads((directory / "config.json").read_text()).get("architectures")
    if architectures != [wanted]:
        raise ValueError(f"draft directory {directory} holds {architectures}, expected [{wanted!r}] for {proposer}")
    return directory


def create(proposer, k, model, context, num_pages, *, draft_precision='bf16', draft_dir=None):
    if proposer == "mtp":
        from sharar.model.mtp import MtpHead
        from sharar.proposers.mtp import MtpProposer
        from sharar.weights.safetensors import SnapshotWeights
        w = SnapshotWeights(model.target.snapshot_dir())
        try:
            if model.target.family == "flashnext":
                from sharar.model.flashnext.mtp import FlashNextMtpHead
                head = FlashNextMtpHead(w, context.device, model, num_pages)
            else:
                head = MtpHead(w, context.device, model.embed, model.lm_head, model.rope, num_pages)
        finally:
            w.close()
        head.target = model.target
        adapter = MtpProposer(head, context, k)
    elif proposer == "ngram":
        from sharar.proposers.ngram import NGramProposer
        adapter = NGramProposer(context, k)
    elif proposer in ("dspark", "dflash2"):
        from sharar.model.parallel_draft import ParallelDraftModel
        from sharar.proposers.parallel import ParallelProposer
        snapshot = draft_snapshot(proposer, draft_dir, model.target)
        adapter = ParallelProposer(ParallelDraftModel(snapshot, model, context.max_context, precision=draft_precision), context, k, proposer)
    else:
        raise ValueError(f"unknown proposer {proposer!r}")
    return adapter
