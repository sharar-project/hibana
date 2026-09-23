"""Safe re-planning of FlashInfer wrappers on the eager path.

plan() fills a pinned host buffer and copies it to the device asynchronously.
Planning the same wrapper again before the GPU has executed that copy rewrites
the buffer under the queued copy, so the earlier launch reads the later plan:
wrong page indices and an illegal memory access. Eager Flash-Next hit it under
load (2026-09-23): the MTP head re-plans once per draft step, and a cache-hit
prefill and a verification shared a wrapper in one scheduler step. CUDA graphs
plan into fixed buffers between replays and are not affected. Waiting on an
event recorded after the previous plan only blocks when the GPU has not yet
reached that copy.
"""

import torch


def plan(wrapper, *args, **kwargs) -> None:
    event = getattr(wrapper, "_sharar_plan_event", None)
    if event is not None:
        event.synchronize()
    wrapper.plan(*args, **kwargs)
    if torch.cuda.is_available():
        event = torch.cuda.Event()
        event.record()
        wrapper._sharar_plan_event = event
