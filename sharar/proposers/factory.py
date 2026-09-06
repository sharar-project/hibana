"""Checkpoint-specific construction at startup; the executor consumes contracts."""
from sharar import config

MAX_DEPTHS = {"mtp": 4, "ngram": 4, "dspark": 7, "dflash2": 7}


def create(proposer, k, model, context, num_pages):
    if proposer == "mtp":
        from sharar.model.mtp import MtpHead
        from sharar.proposers.mtp import MtpProposer
        from sharar.weights.safetensors import SnapshotWeights
        w = SnapshotWeights(config.snapshot_dir())
        try:
            head = MtpHead(w, context.device, model.embed, model.lm_head, model.rope, num_pages)
        finally:
            w.close()
        adapter = MtpProposer(head, context, k)
    elif proposer == "ngram":
        from sharar.proposers.ngram import NGramProposer
        adapter = NGramProposer(context, k)
    elif proposer in ("dspark", "dflash2"):
        from sharar.model.parallel_draft import ParallelDraftModel
        from sharar.proposers.parallel import ParallelProposer, PINS
        name, revision = PINS[proposer]
        root = config.snapshot_dir().parents[2]
        snapshot = root / ("models--" + name.replace("/", "--")) / "snapshots" / revision
        adapter = ParallelProposer(ParallelDraftModel(snapshot, model, context.max_context), context, k, proposer)
    else:
        raise ValueError(f"unknown proposer {proposer!r}")
    return adapter
