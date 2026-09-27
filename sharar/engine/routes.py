"""Router traces (#32): the experts a mixture-of-experts target routes every decode and verify row to.

Opt-in with ``SHARAR_TRACE_ROUTER=1`` beside the trace sink (``engine/traces.py``). The model reads
the setting when it is built and hands its router kernels the buffers of one `RouteTap`, so the
CUDA graphs capture the recording; with the setting off nothing is allocated and the kernels are
the unrecorded ones.

- `RouteTap` (device): per MoE layer the top-k expert ids and their softmax probabilities (before
  the renormalization the combine uses) of the first `ROWS` rows of the latest forward, plus the
  draft layer's routes for each draft step of the current round. After a decode or verify forward
  the engine stages the rows into pinned memory without a synchronization; the step's own token
  download completes the copy, and `take` returns host arrays.
- `RouteRecorder` (host, one per recorded request): keeps the rows of each forward in order, with
  their kind (decode or verify), round and block offset; folds them into compact aggregates
  (`fold`) on the trace writer thread, or on the engine thread in batches when the request does
  not keep full rows.

The model contract: ``model.route_tap()`` returns the tap or None (dense targets, or recording
off). Forwards of more than `ROWS` rows (prefill chunks) are not recorded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROWS = 64  # the fused router kernel's row limit (MoeLayer.route); decode and verify batches fit
DECODE, VERIFY = 0, 1  # row kinds
FOLD_ROWS = 4096  # aggregate-only requests fold their pending rows on the engine thread past this many


class RouteTap:
    """Device route buffers one model's router kernels write into, and their pinned host staging."""

    def __init__(self, layers: int, k: int, experts: int, device: torch.device | str, draft_depth: int = 8):
        self.layers, self.k, self.experts, self.draft_depth = layers, k, experts, draft_depth
        self.device = torch.device(device)
        d = self.device
        self.ids = torch.zeros((ROWS, layers, k), dtype=torch.int32, device=d)
        self.probs = torch.zeros((ROWS, layers, k), dtype=torch.float32, device=d)
        # The draft layer (an MTP head's router) writes one slot per step; `keep_draft` files it under the step.
        self.draft_slot = (torch.zeros((ROWS, k), dtype=torch.int32, device=d), torch.zeros((ROWS, k), dtype=torch.float32, device=d))
        self.draft_ids = torch.zeros((ROWS, draft_depth, k), dtype=torch.int32, device=d)
        self.draft_probs = torch.zeros((ROWS, draft_depth, k), dtype=torch.float32, device=d)
        pin = d.type == "cuda"
        self._host = {name: torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
                      for name, t in (("ids", self.ids), ("probs", self.probs), ("draft_ids", self.draft_ids), ("draft_probs", self.draft_probs))}
        self._staged = (0, 0)
        self.draft_steps = 0  # draft steps filed this round

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.ids, self.probs, *self.draft_slot, self.draft_ids, self.draft_probs))

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (ids, probs) record buffers of MoE layer `index`: [ROWS, k] views, row stride layers * k."""
        return self.ids[:, index], self.probs[:, index]

    def keep_draft(self, step: int, rows: int) -> None:
        """File the draft layer's routes of draft step `step` (device copies, no synchronization)."""
        if step >= self.draft_depth or rows > ROWS:
            return
        self.draft_ids[:rows, step].copy_(self.draft_slot[0][:rows])
        self.draft_probs[:rows, step].copy_(self.draft_slot[1][:rows])
        self.draft_steps = step + 1

    def stage(self, rows: int, draft_rows: int = 0, forward_rows: int | None = None) -> bool:
        """Queue the device-to-host copy of the latest forward's first `rows` rows (and `draft_rows` rows of
        draft routes) behind the work already on the stream. `forward_rows`: the rows the forward actually ran
        (a graph's padded size). False when the forward was too large to record."""
        if max(rows, forward_rows or 0, draft_rows) > ROWS:
            self._staged = (0, 0)
            return False
        h = self._host
        h["ids"][:rows].copy_(self.ids[:rows], non_blocking=True)
        h["probs"][:rows].copy_(self.probs[:rows], non_blocking=True)
        if draft_rows:
            h["draft_ids"][:draft_rows].copy_(self.draft_ids[:draft_rows], non_blocking=True)
            h["draft_probs"][:draft_rows].copy_(self.draft_probs[:draft_rows], non_blocking=True)
        self._staged = (rows, draft_rows)
        return True

    def take(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
        """The staged rows as host arrays: ids uint16 [rows, layers, k], probs float32, and the draft routes
        [draft_rows, steps, k] (None without). Call after the step's own download synchronized the stream."""
        rows, draft_rows = self._staged
        if not rows:
            return None
        h = self._host
        ids = h["ids"][:rows].numpy().astype(np.uint16)
        probs = h["probs"][:rows].numpy().copy()
        draft_ids = draft_probs = None
        if draft_rows and self.draft_steps:
            draft_ids = h["draft_ids"][:draft_rows, :self.draft_steps].numpy().astype(np.uint16)
            draft_probs = h["draft_probs"][:draft_rows, :self.draft_steps].numpy().copy()
        self._staged = (0, 0)
        return ids, probs, draft_ids, draft_probs


def empty_aggregate(layers: int, k: int, experts: int) -> dict:
    return {
        "rows": np.zeros(2, dtype=np.int64),                      # [decode, verify] rows folded
        "counts": np.zeros((layers, experts), dtype=np.int64),    # rows that routed to each expert
        "mass": np.zeros((layers, experts), dtype=np.float64),    # summed combine weights (renormalized top-k)
        "rank_mass": np.zeros((layers, k), dtype=np.float64),     # summed combine weight at each rank
        "topk_mass": np.zeros(layers, dtype=np.float64),          # summed softmax mass the top-k carry
        "block_rounds": np.zeros(ROWS + 1, dtype=np.int64),       # verify rounds by block size (rows)
        "block_distinct": np.zeros((ROWS + 1, layers), dtype=np.int64),  # summed distinct experts per layer by block size
    }


def fold(agg: dict, ids: np.ndarray, probs: np.ndarray, kind: np.ndarray, blocks: list[tuple[int, int]] = ()) -> dict:
    """Add rows to an aggregate: ids [N, L, k], probs [N, L, k] (softmax, not renormalized), kind [N];
    `blocks` are (first row, rows) of the verify blocks among them."""
    n, layers, k = ids.shape
    if n == 0:
        return agg
    experts = agg["counts"].shape[1]
    agg["rows"] += np.bincount(kind.astype(np.int64), minlength=2)[:2]
    probs = probs.astype(np.float64)
    total = probs.sum(-1, keepdims=True)
    weights = np.divide(probs, total, out=np.zeros_like(probs), where=total > 0)
    flat = (np.arange(layers)[None, :, None] * experts + ids.astype(np.int64)).reshape(-1)
    agg["counts"] += np.bincount(flat, minlength=layers * experts).reshape(layers, experts)
    agg["mass"] += np.bincount(flat, weights=weights.reshape(-1), minlength=layers * experts).reshape(layers, experts)
    agg["rank_mass"] += weights.sum(0)
    agg["topk_mass"] += total[..., 0].sum(0)
    for first, size in blocks:
        agg["block_rounds"][size] += 1
        agg["block_distinct"][size] += distinct_per_layer(ids[first:first + size], experts)
    return agg


def distinct_per_layer(ids: np.ndarray, experts: int) -> np.ndarray:
    """ids [rows, L, k] -> [L] number of distinct experts the rows touch in each layer."""
    rows, layers, k = ids.shape
    seen = np.zeros((layers, experts), dtype=bool)
    seen[np.repeat(np.arange(layers), rows * k), ids.transpose(1, 0, 2).reshape(-1)] = True
    return seen.sum(1)


class RouteRecorder:
    """The route rows of one request, in forward order; aggregates always, every row when `full`."""

    def __init__(self, layers: int, k: int, experts: int, full: bool):
        self.layers, self.k, self.experts, self.full = layers, k, experts, full
        self.agg = empty_aggregate(layers, k, experts)
        self._parts: list[tuple] = []  # (kind, round, position, ids, probs, draft_ids, draft_probs) not yet folded
        self._pending = 0
        self.rows = 0
        self.valid = True

    def add(self, kind: int, round_index: int, position: int, ids: np.ndarray, probs: np.ndarray,
            draft_ids: np.ndarray | None = None, draft_probs: np.ndarray | None = None) -> None:
        """Rows of one forward for this request: a decode row, or a verify block (pending token then drafts)
        starting at `position`, with the draft layer's routes of the round's draft steps."""
        if ids.shape[1:] != (self.layers, self.k):
            self.valid = False
            return
        self._parts.append((kind, round_index, position, ids, probs, draft_ids, draft_probs))
        self._pending += ids.shape[0]
        self.rows += ids.shape[0]
        if not self.full and self._pending >= FOLD_ROWS:
            self._fold_pending()

    def _fold_pending(self) -> None:
        if not self._parts:
            return
        ids = np.concatenate([p[3] for p in self._parts])
        probs = np.concatenate([p[4] for p in self._parts])
        kind = np.concatenate([np.full(p[3].shape[0], p[0], dtype=np.int8) for p in self._parts])
        blocks, first = [], 0
        for p in self._parts:
            if p[0] == VERIFY:
                blocks.append((first, p[3].shape[0]))
            first += p[3].shape[0]
        fold(self.agg, ids, probs, kind, blocks)
        self._parts, self._pending = [], 0

    def arrays(self) -> dict:
        """Every array the trace file stores (writer thread): the aggregate, and the full rows when kept."""
        out = {}
        if self.full and self._parts:
            parts = self._parts
            out["ids"] = np.concatenate([p[3] for p in parts])
            out["probs"] = np.concatenate([p[4] for p in parts]).astype(np.float16)
            out["kind"] = np.concatenate([np.full(p[3].shape[0], p[0], dtype=np.int8) for p in parts])
            out["round"] = np.concatenate([np.full(p[3].shape[0], p[1], dtype=np.int32) for p in parts])
            out["offset"] = np.concatenate([np.arange(p[3].shape[0], dtype=np.int16) for p in parts])
            out["position"] = np.concatenate([p[2] + np.arange(p[3].shape[0], dtype=np.int32) for p in parts])
            drafted = [p for p in parts if p[5] is not None]
            if drafted:
                depth = max(p[5].shape[0] for p in drafted)
                di = np.zeros((len(drafted), depth, self.k), dtype=np.uint16)
                dp = np.zeros((len(drafted), depth, self.k), dtype=np.float16)
                for j, p in enumerate(drafted):
                    di[j, :p[5].shape[0]] = p[5]
                    dp[j, :p[5].shape[0]] = p[6]
                out["draft_ids"], out["draft_probs"] = di, dp
                out["draft_round"] = np.array([p[1] for p in drafted], dtype=np.int32)
                out["draft_steps"] = np.array([p[5].shape[0] for p in drafted], dtype=np.int16)
        self._fold_pending()
        for name, value in self.agg.items():
            out[f"agg_{name}"] = value
        return out

    def write(self, path: Path) -> dict | None:
        """Writer thread: one .npz with the aggregate and, when kept, every row."""
        if not self.valid or self.rows == 0:
            return None
        arrays = self.arrays()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)
        return {"rows": self.rows, "full": "ids" in arrays, "layers": self.layers, "k": self.k, "experts": self.experts,
                "bytes": path.stat().st_size}

    def _release(self) -> None:
        self._parts, self._pending = [], 0


def load_routes(path: str | Path) -> dict:
    """A trace's route file as a dict of arrays (`agg_*` always; `ids`, `probs`, ... when full)."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def summarize_aggregate(agg: dict, top: int = 3) -> dict:
    """The few numbers `sharar traces` prints from summed aggregates (`agg_*` arrays without the prefix)."""
    rows = int(agg["rows"].sum())
    if rows == 0:
        return {"rows": 0}
    counts, rank = agg["counts"], agg["rank_mass"]
    per_layer_rows = counts.sum(1) / rank.shape[1]
    rank_share = rank.sum(0) / rank.sum()
    mass = agg["mass"]
    need = [experts_for_share(mass[layer], 0.5) for layer in range(mass.shape[0])], [experts_for_share(mass[layer], 0.9) for layer in range(mass.shape[0])]
    blocks = {int(s): round(float(agg["block_distinct"][s].mean() / agg["block_rounds"][s]), 1)
              for s in np.nonzero(agg["block_rounds"])[0]}
    return {
        "rows": {"decode": int(agg["rows"][0]), "verify": int(agg["rows"][1])},
        "rank_share": [round(float(x), 4) for x in rank_share],
        "ranks_7_10_share": round(float(rank_share[6:].sum()), 4),
        "topk_softmax_mass": round(float(agg["topk_mass"].sum() / per_layer_rows.sum()), 4),
        "experts_for_50pct_mass": {"median": float(np.median(need[0])), "min": int(min(need[0])), "max": int(max(need[0]))},
        "experts_for_90pct_mass": {"median": float(np.median(need[1])), "min": int(min(need[1])), "max": int(max(need[1]))},
        "distinct_experts_per_verify_block": blocks,
    }


def experts_for_share(mass: np.ndarray, share: float) -> int:
    """How many experts, largest first, carry `share` of this mass vector."""
    total = mass.sum()
    if total <= 0:
        return 0
    ordered = np.sort(mass)[::-1].cumsum() / total
    return int(np.searchsorted(ordered, share - 1e-12) + 1)
