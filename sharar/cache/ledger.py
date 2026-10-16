"""One unified-memory budget for the whole process.

GB10 has a single physical pool shared by the OS, the host process and the
GPU allocator, so the only true constraint is host MemAvailable minus a hard
floor. The ledger records every large allocation category, plans the KV page
and state slot counts from what is available at startup, and can re-check the
floor at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from sharar.hardware.gb10 import HOST_FLOOR_KIB
from sharar.target import DEFAULT_TARGET, Target

GIB = 2**30
# The default target's per-token and per-slot bytes, for tools written against the 27B.
KV_BYTES_PER_TOKEN = DEFAULT_TARGET.kv_bytes_per_token
STATE_BYTES_PER_SLOT = DEFAULT_TARGET.state_bytes_per_slot
# Per-request context every running request is guaranteed at once; longer
# contexts are guaranteed for one request (the 32K profile's plan is unchanged).
# Above four running requests (the throughput profile, #59) the same total is
# guaranteed and split between them: 16 requests are guaranteed 8K each.
SHARED_CONTEXT = 32768
SHARED_REQUESTS = 4


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def check_load_headroom(peak_bytes: int) -> None:
    """Reject a second model load before transient allocations reach the floor."""
    available = mem_available_bytes()
    floor = HOST_FLOOR_KIB * 1024
    if available < floor + peak_bytes:
        raise MemoryError(f"model load needs {(floor + peak_bytes) / GIB:.1f} GiB MemAvailable including the host floor; have {available / GIB:.1f} GiB")


@dataclass
class Plan:
    kv_tokens: int
    state_slots: int
    reserved_bytes: int  # activation workspace kept free for the largest prefill


@dataclass
class Ledger:
    floor_bytes: int = 16 * GIB
    entries: dict[str, int] = field(default_factory=dict)
    start_available: int = field(default_factory=mem_available_bytes)
    target: Target = DEFAULT_TARGET

    def record(self, name: str, nbytes: int) -> None:
        self.entries[name] = self.entries.get(name, 0) + int(nbytes)

    def record_cuda_allocated(self, name: str) -> None:
        """Attribute the growth of torch's allocation since the last call."""
        cur = torch.cuda.memory_allocated()
        prev = getattr(self, "_last_cuda", 0)
        self.record(name, cur - prev)
        self._last_cuda = cur

    @property
    def total(self) -> int:
        return sum(self.entries.values())

    def plan(
        self,
        max_running: int = 4,
        context_len: int | None = None,
        prefill_reserve_bytes: int = 8 * GIB,
        state_slot_target: int = 96,
        kv_token_target: int = 1 << 20,
        margin_bytes: int = 4 * GIB,
        future_bytes: int = 0,
        extra_page_bytes: int = 0,
        state_bytes_per_slot: int | None = None,
    ) -> Plan:
        """Size the pools from what is available now.

        Guarantees KV pages for one request at `context_len` or `max_running`
        requests at `SHARED_CONTEXT` (above `SHARED_REQUESTS` requests, the
        same total split between them), whichever is more, and committed state
        slots for `max_running` requests, plus the prefill activation reserve
        and the floor; the remainder, less `margin_bytes` for what the engine
        allocates after the plan (CUDA graphs, workspaces) but never less than
        half of it, goes to prefix-cache capacity. Admission budgets each request's actual tokens,
        so a long request and shorter ones share the pool and a request that
        does not fit waits. A cached request costs about three state snapshots
        (441 MiB) against 256 MiB of KV for an 8K prompt, and a soak test
        showed slots, not pages, drive eviction, so slots take up to 60% of
        the spare memory and pages the rest. `state_bytes_per_slot` defaults
        to the target's FP32 state.
        """
        context_len = self.target.context_length if context_len is None else context_len
        kv_bytes_per_token = self.target.kv_bytes_per_token
        state_bytes_per_slot = state_bytes_per_slot or self.target.state_bytes_per_slot
        available = mem_available_bytes()
        budget = available - self.floor_bytes - prefill_reserve_bytes - future_bytes
        page_bytes = kv_bytes_per_token + extra_page_bytes
        shared = SHARED_CONTEXT * min(max_running, SHARED_REQUESTS) // max_running
        must_tokens = max(context_len, max_running * min(context_len, shared))
        must_kv = must_tokens * page_bytes
        must_state = (max_running + 1) * state_bytes_per_slot  # +1 dummy slot 0
        if budget < must_kv + must_state:
            raise MemoryError(
                f"only {available / GIB:.1f} GiB available; need {(must_kv + must_state + prefill_reserve_bytes + future_bytes + self.floor_bytes) / GIB:.1f} GiB"
            )
        leftover = max(0, budget - must_kv - must_state)
        # never less than half of it (the split before the margin): where little is left (Flash-Next `exact`, its
        # NVFP4 experts and the SSD staging), the margin alone would leave the pools a handful of slots
        spare = max(leftover - margin_bytes, leftover // 2)
        # the target never cuts the guaranteed slots: a batch of 128 needs 129, and room for cached sessions besides
        state_slots = min(max(state_slot_target, 2 * max_running + 1), max_running + 1 + int(spare * 0.6) // state_bytes_per_slot)
        spare -= (state_slots - max_running - 1) * state_bytes_per_slot  # what the slot cap leaves goes to pages
        kv_tokens = min(kv_token_target, must_tokens + spare // page_bytes)
        return Plan(kv_tokens=kv_tokens, state_slots=state_slots, reserved_bytes=prefill_reserve_bytes + future_bytes)

    def above_floor(self) -> bool:
        """Whether host MemAvailable is still above the floor (no exception)."""
        return mem_available_bytes() >= self.floor_bytes

    def check_floor(self) -> int:
        available = mem_available_bytes()
        if available < self.floor_bytes:
            raise MemoryError(f"MemAvailable {available / GIB:.1f} GiB is below the {self.floor_bytes / GIB:.0f} GiB floor")
        return available

    def describe(self) -> str:
        lines = [f"  {name:<28}{nbytes / GIB:8.2f} GiB" for name, nbytes in self.entries.items()]
        lines.append(f"  {'total recorded':<28}{self.total / GIB:8.2f} GiB")
        lines.append(f"  {'MemAvailable now':<28}{mem_available_bytes() / GIB:8.2f} GiB  (floor {self.floor_bytes / GIB:.0f} GiB)")
        return "\n".join(lines)
