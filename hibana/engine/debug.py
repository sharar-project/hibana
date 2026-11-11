"""HIBANA_DEBUG_SYNC=1: synchronize and name the phase, so an asynchronous CUDA
fault is reported by the phase that launched it, with the batch it served."""

import torch


def debug_sync(phase: str, reqs=None) -> None:
    from hibana.engine.config import current
    if not current().debug_sync or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        batch = [(r.request_id, len(r.prompt_ids), r.context_len, r.committed_len, r.prefill_pos, r.prefill_len) for r in reqs or []]
        raise RuntimeError(f"CUDA fault surfaced {phase}; batch (id, prompt, context, committed, prefill_pos, prefill_len): {batch}") from exc
