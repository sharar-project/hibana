"""Host -> device uploads that do not stall the host.

`torch.tensor(values, device=cuda)` copies through pageable memory and waits
for the GPU stream to drain first; on the serving path that idles the GPU
between a decode graph and the next prefill. A pinned staging tensor with a
non-blocking copy queues the transfer instead (the pinned caching allocator
keeps the staging block alive until the copy completes).
"""

from __future__ import annotations

import torch


def upload(values, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    host = torch.tensor(values, dtype=dtype)
    if host.numel() == 0 or str(device) == "cpu":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)
