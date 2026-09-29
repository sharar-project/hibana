"""Draft depth policies (#45): how many drafted tokens a round verifies.

`SpecDecoder` asks its policy once per round; the answer must be one of the
depths whose verify graphs were captured (`SpecDecoder.verify_depths`).
`FixedDepth` is the served behavior: the proposer's k every round.
`BatchDepth` chooses by the batch bucket (#59): the throughput profile verifies
shallower blocks, or none (depth 0: plain decode), when many requests share a
round. Adaptive depth (#43) plugs in here.
"""

from __future__ import annotations

from sharar.engine import graph_set


class FixedDepth:
    depths = None  # one depth at every batch size (`BatchDepth.depths` maps bucket -> depth)

    def __init__(self, depth: int):
        if type(depth) is not int or depth < 1:
            raise ValueError("draft depth must be a positive integer")
        self.depth = depth

    def choose(self, requests) -> int:
        return self.depth

    def sizes_for(self, depth: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
        """The batch buckets that verify at `depth`: all of them (benchmarks switch the depth at run time)."""
        return sizes


class BatchDepth:
    """The depth of the smallest batch bucket that holds the batch; 0 runs the batch as plain decode."""

    def __init__(self, depths: dict[int, int], k: int):
        if not depths or any(type(d) is not int or not 0 <= d <= k for d in depths.values()):
            raise ValueError(f"depths per batch bucket must be in [0, {k}]")
        self.depths = dict(sorted(depths.items()))
        self.sizes = tuple(self.depths)
        self.depth = self.depths[self.sizes[0]]  # the depth a single request verifies

    def choose(self, requests) -> int:
        return self.depths[graph_set.pick(self.sizes, len(requests))]

    def sizes_for(self, depth: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(b for b in sizes if self.depths.get(b) == depth)

    @property
    def verify_depths(self) -> tuple[int, ...]:
        return tuple(sorted({d for d in self.depths.values() if d}))


def parse_depths(text: str | None, sizes: tuple[int, ...], k: int) -> BatchDepth | None:
    """`SHARAR_SPEC_DEPTHS` as "BUCKET:DEPTH,..." (e.g. "8:1,16:0"); buckets not named verify the proposer's k.
    None when unset or when every bucket keeps k."""
    if not text:
        return None
    depths = dict.fromkeys(sizes, k)
    for item in text.split(","):
        bucket, sep, depth = item.strip().partition(":")
        if not sep or not bucket.strip().isdigit() or not depth.strip().isdigit():
            raise ValueError(f"SHARAR_SPEC_DEPTHS must be BUCKET:DEPTH pairs, got {text!r}")
        if int(bucket) not in depths:
            raise ValueError(f"SHARAR_SPEC_DEPTHS bucket {bucket} is not a captured batch size {sizes}")
        depths[int(bucket)] = int(depth)
    if all(d == k for d in depths.values()):
        return None
    return BatchDepth(depths, k)
