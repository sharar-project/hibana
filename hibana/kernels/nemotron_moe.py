"""Nemotron-3.5-Super's latent MoE block for decode and verify calls (at most 16 rows): `csrc/nemotron_moe.cu`.

`router` (the router|fc1 GEMV; it also routes each row as `model/nemotron/moe.py`'s route_topk does, bit for bit,
plans the call, and quantizes x and the latent), then `block` (up, down, out), or `w3_block` around the 3-bit
experts. Budgeted routing routes from the router's FP32 output itself and hands `block` its ids to plan. The NVFP4
tensors are the checkpoint's, as b12x's MoE and the NVFP4 GEMM read them for prefill. `NemotronMoe.small_weights`
lists what the kernels read. The block's launches are chained by programmatic dependent launch, and the router GEMV
runs on the CUDA cores up to 8 rows (58 against 64 us at M=1-4 on tensor-core fragments)."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

_SRC = Path(__file__).with_name("csrc") / "nemotron_moe.cu"
MAX_ROWS = 16


@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    from hibana.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    return load(name=f"sharar_nemotron_moe_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3"], verbose=False)


_COUNTERS: dict[torch.device, torch.Tensor] = {}


def counters(device: torch.device) -> torch.Tensor:
    """The kernels' arrival counters (the down kernel's tiles, the router's logit blocks): zero between calls, each
    call's last blocks reset theirs. One set per device: the block's calls run in stream order."""
    c = _COUNTERS.get(device)
    if c is None:
        c = _COUNTERS[device] = torch.zeros(64, dtype=torch.int32, device=device)
    return c


def router(x: torch.Tensor, weights: list[torch.Tensor], bias: torch.Tensor, k: int, slots: torch.Tensor | None = None,
           steps: int = 1, route: bool = True, pad_id: int = -1, latent: bool = False):
    """x [M, 4096] bf16 -> (fused FP32 [M, 1536], ids [M, k], weights [M, k], workspace[, BF16 latent [M, 1024] with
    `latent`]). With `route`, ids / weights are route_topk's (with `slots`: rows of requests on slot 0 padded: row 0's
    top expert at weight 0, or `pad_id` when >= 0, e.g. 512 = no expert for the 3-bit experts) and the call is
    planned; else unwritten."""
    slots = x.new_empty(0, dtype=torch.int32) if slots is None else slots.to(torch.int32).contiguous()
    r = _module().router(x, weights, bias, k, slots, steps, route, pad_id)
    return tuple(r) if latent else tuple(r[:4])


def block(x, ws, ids, wts, weights: list[torch.Tensor], plan: bool = False) -> torch.Tensor:
    """The rest of the block after `router` -> bf16 [M, 4096]; `plan` when the caller routed (ids not the router's)."""
    return _module().block(x, ws, ids, wts, weights, plan)


def w3_block(x, ws, lat, ids, wts, weights: list[torch.Tensor], experts) -> torch.Tensor:
    """The block after `router` (with `latent`) around 3-bit routed experts (`kernels/w3_moe.py`'s W3Experts):
    the shared expert's up, the routed sum through w3_moe from the router's BF16 latent and ids, then fc2_latent plus
    the shared expert's down -> bf16 [M, 4096]. `weights` may name no NVFP4 experts (empty tensors)."""
    from hibana.kernels.w3_moe import w3_moe
    m = _module()
    m.shared_up(x, ws, weights)
    return m.out(x, ws, w3_moe(experts, lat, ids, wts), weights)
