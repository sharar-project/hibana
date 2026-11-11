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

from hibana.target import DEFAULT_TARGET, Target

GIB = 2**30
# The default target's per-token and per-slot bytes, for tools written against the 27B.
KV_BYTES_PER_TOKEN = DEFAULT_TARGET.kv_bytes_per_token
STATE_BYTES_PER_SLOT = DEFAULT_TARGET.state_bytes_per_slot
# Per-request context every running request is guaranteed at once; longer
# contexts are guaranteed for one request (the 32K profile's plan is unchanged).
# Above four running requests (the throughput profile) the same total is
# guaranteed and split between them: 16 requests are guaranteed 8K each.
SHARED_CONTEXT = 32768
# Snapshots the tree keeps for one cached conversation, which sizes the state slots' share of spare memory: its
# message boundary, its prompt's last chunk end and its reply end (the cache policy, docs/cache_design.md).
CACHED_SNAPSHOTS = 3
SHARED_REQUESTS = 4


def host_floor_bytes() -> int:
    """The host floor (HIBANA_HOST_FLOOR_GIB, default 6 GiB): the MemAvailable the pools leave for the OS and other
    programs once everything Sharar allocates is counted."""
    from hibana.engine.config import current
    return int(current().host_floor_gib * GIB)


# What Sharar allocates after the pools are planned, counted so that the plan leaves the floor free. Measured on
# Flash-Next, one Spark, HIBANA_MAX_BATCH=64: the first prefill takes 2.12 GiB of GPU memory outside PyTorch (the
# driver's per-process count; never freed), and PyTorch keeps 0.87 GiB of activation blocks cached beyond what is
# allocated. Neither grew with requests, context or prefill.
FIRST_USE_BYTES = int(2.25 * GIB)
ACTIVATION_BYTES = 1 * GIB
# MemAvailable moves between the plan's read and the checks after each allocation against a fresh read (the page
# cache; on two Sparks each rank reads its own and every rank takes the smallest plan): left unplanned so those checks
# do not fail on a few MB of drift (Flash-Next --throughput --tp 2: "needs 55.1 GiB, have 55.1 GiB" on rank 1)
PLAN_SLACK_BYTES = GIB // 4
# What a byte of the pools takes from MemAvailable: on a Spark's unified memory a CUDA allocation lowers MemAvailable by
# more than its size (Flash-Next, one Spark, no cached blocks to reuse: the KV pool's 19.78 GiB took 20.29, the state
# pool's 10.87 took 11.06, the workspaces' 0.53 took 0.92; 3.5% over the three). Planned at face value, the pools left
# the drafter 1.1 GiB short of the headroom the plan had kept for it ("needs 12.6 GiB, have 11.7").
POOL_COST = 1.04


def graph_bytes(max_batch: int) -> int:
    """The decode, verify and draft CUDA graphs for the batch buckets up to max_batch (1.77 GiB measured at 64)."""
    return max(GIB // 2, int(1.77 * GIB * max_batch / 64))


def workspace_bytes(max_batch: int) -> int:
    """FlashInfer's decode, prefill and verify workspaces (`Executor.__init__`); the prefill and verify ones grow with
    each 64 rows."""
    wide = max(1, -(-max_batch // 64))
    return (128 + 256 * wide + 128 * wide) * 2**20


def after_plan_bytes(max_batch: int) -> dict[str, int]:
    """Sharar's own allocations after the pool plan, by part."""
    return {"workspaces": workspace_bytes(max_batch), "graphs": graph_bytes(max_batch), "first_use": FIRST_USE_BYTES,
            "activations": ACTIVATION_BYTES}


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def check_load_headroom(peak_bytes: int) -> None:
    """Reject a load (weights, pools) whose allocations would leave less than the floor."""
    available = mem_available_bytes()
    floor = host_floor_bytes()
    if available < floor + peak_bytes:
        raise MemoryError(f"model load needs {(floor + peak_bytes) / GIB:.1f} GiB MemAvailable including the host floor; have {available / GIB:.1f} GiB")


@dataclass
class Plan:
    kv_tokens: int
    state_slots: int
    after_pools_bytes: int  # what Sharar allocates after the pools: drafter, SSD staging, workspaces, graphs, first use, activations


@dataclass
class Ledger:
    floor_bytes: int = field(default_factory=host_floor_bytes)
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
        state_slot_target: int = 96,
        kv_token_target: int | None = None,
        future_bytes: int = 0,
        extra_page_bytes: int = 0,
        state_bytes_per_slot: int | None = None,
        cached_prompt_tokens: int = SHARED_CONTEXT,
        restore_slots: int = 0,
    ) -> Plan:
        """Size the pools from what is available now: everything but the floor and `future_bytes`, what Sharar
        allocates after the pools (the drafter, the SSD tier's staging, and `after_plan_bytes`: workspaces, CUDA
        graphs, the first prefill's allocations outside PyTorch, cached activations).

        Guarantees KV pages for one request at `context_len` or `max_running` requests at `SHARED_CONTEXT` (above
        `SHARED_REQUESTS` requests, the same total split between them), whichever is more, and the startup slot floor:
        two state slots per running request (its own slot and the snapshot its lock keeps), the dummy slot 0, and
        `restore_slots`, what an SSD restore can lock beyond those (`ssd.RESTORE_SLOTS`). With that floor admission
        never waits for a slot a running request holds, and private snapshots, which live only inside a prefill step,
        take what is left. The rest goes to
        prefix-cache capacity. Admission budgets each request's actual tokens, so a long request and shorter ones share
        the pool and a request that does not fit waits.

        Prefix-cache capacity is split by what one cached conversation costs: its KV pages and the snapshots the tree
        keeps for it (`CACHED_SNAPSHOTS`: its message boundary, its prompt's last chunk end and its reply end), at
        `cached_prompt_tokens` (32K)
        a conversation. With the 3-bit experts' INT8 state that is about 17% state slots and 83% KV pages; with FP32
        state about 34% state slots. `state_slot_target` stays a minimum (the 4-request profiles keep 96
        slots), and slots never take more than 60% of the spare memory. The KV pool has no token cap unless
        `kv_token_target` sets one. `state_bytes_per_slot` defaults to the target's FP32 state.
        """
        context_len = self.target.context_length if context_len is None else context_len
        kv_bytes_per_token = self.target.kv_bytes_per_token
        state_bytes_per_slot = state_bytes_per_slot or self.target.state_bytes_per_slot
        available = mem_available_bytes()
        budget = int((available - self.floor_bytes - future_bytes - PLAN_SLACK_BYTES) / POOL_COST)  # in pool bytes
        page_bytes = kv_bytes_per_token + extra_page_bytes
        shared = SHARED_CONTEXT * min(max_running, SHARED_REQUESTS) // max_running
        must_tokens = max(context_len, max_running * min(context_len, shared))
        must_kv = must_tokens * page_bytes
        # the slot floor: per running request its own slot and the snapshot its lock keeps, the dummy slot 0, and what an
        # SSD restore can lock beyond those
        must_slots = 2 * max_running + 1 + restore_slots
        must_state = must_slots * state_bytes_per_slot
        if budget < must_kv + must_state:
            raise MemoryError(
                f"only {available / GIB:.1f} GiB available; need {(must_kv + must_state + future_bytes + self.floor_bytes) / GIB:.1f} GiB "
                f"(pools for {max_running} requests, {future_bytes / GIB:.1f} GiB allocated after them, the "
                f"{self.floor_bytes / GIB:.0f} GiB floor): lower HIBANA_MAX_BATCH or HIBANA_HOST_FLOOR_GIB"
            )
        spare = budget - must_kv - must_state
        snapshots = CACHED_SNAPSHOTS * state_bytes_per_slot  # what the tree keeps for one cached conversation
        share = snapshots / (snapshots + cached_prompt_tokens * page_bytes)
        state_slots = min(must_slots + int(spare * 0.6) // state_bytes_per_slot,
                          max(state_slot_target, must_slots + int(spare * share) // state_bytes_per_slot))
        spare -= (state_slots - must_slots) * state_bytes_per_slot  # what the slots leave goes to pages
        kv_tokens = must_tokens + spare // page_bytes
        if kv_token_target is not None:
            kv_tokens = min(kv_token_target, kv_tokens)
        return Plan(kv_tokens=kv_tokens, state_slots=state_slots, after_pools_bytes=future_bytes)

    def above_floor(self) -> bool:
        """Whether host MemAvailable is above half the floor; under it admission holds (and the memory guard acts)."""
        return mem_available_bytes() >= self.floor_bytes // 2

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
