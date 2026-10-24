"""A recurrent-state snapshot and what depends on it.

A hybrid model resumes a cached prefix only from a GDN state snapshot at its end: attention pages alone cannot resume
the recurrent layers. A `Snapshot` is one state slot (the conv/SSM state after `position` tokens) together with the
drafters' metadata for that position (`auxiliary`; MTP: the final-norm row of `position - 1`; DFlash2: a
feature-window handle bound to `slot`). It has exactly one owner at a time: a prefilling request (one of its private
snapshots, `RequestState.private`, inside one prefill step) or a radix node. Moving it from one to the other moves the
slot and the metadata together, so drafter metadata can neither outlive its slot nor lose it. Freed, it is dead:
`slot` is -1 and `auxiliary` empty.

`spare` marks a snapshot no request is known to need yet: a reply end, and a chunk end of a long prompt that is not its
last. It is cleared the first time a request resumes from the snapshot. Slot pressure takes spare snapshots first and
the rest least recently used (`get_priority`): SGLang's SLRU strategy (evict_policy.SLRUStrategy), whose probationary
segment is the spare one here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

GRID = 64  # the GDN kernels' chunk: a snapshot inside a prefill forward sits a multiple of 64 tokens past its start

_ticks = itertools.count(1)


def tick() -> int:
    """The recency clock for snapshots and KV: a count, not the time, so every tensor-parallel rank (each takes the same
    steps in the same order) orders its snapshots and leaves alike, and no two snapshots tie."""
    return next(_ticks)


@dataclass(eq=False)
class Snapshot:
    slot: int  # the state slot (StatePool) holding the state after `position` tokens; -1 once freed
    position: int  # tokens the state has consumed; in the tree, the depth of the node that owns it
    auxiliary: dict = field(default_factory=dict)  # draft identity -> metadata for `position` (written by the drafters)
    # Set (`tick`) when the snapshot is made and when a request resumes from it (`RadixCache.match` with touch), never
    # when a request merely walks past its node.
    last_access: int = field(default_factory=tick)
    spare: bool = False  # no known reader yet (a reply end, an earlier chunk end): slot pressure takes it first


def get_priority(snapshot: Snapshot) -> tuple[bool, int]:
    """Slot pressure's order (SGLang's EvictionStrategy.get_priority): the lowest value goes first, so spare snapshots
    before the rest, and the least recently used first within each."""
    return not snapshot.spare, snapshot.last_access
