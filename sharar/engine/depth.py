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



class AcceptanceDepth:
    """Adaptive depth (#43): each request drafts as deep as its own recent acceptance pays for.

    Per request, a moving average (weight `alpha` per round) of the drafted tokens accepted per position tested, q.
    With a draft accepted position by position at rate q, a round of depth d emits (1 - q^(d+1)) / (1 - q) tokens on
    average and costs 1 + step_cost * (d - 1) depth-1 rounds (one more draft step and verify row each; Flash-Next on
    one Spark: 31.5 ms at depth 1, about 4.3 ms per token more, so step_cost 0.14). The request's depth maximizes their
    ratio up to the base policy's depth for the batch, and a batch verifies the deepest depth any of its requests
    wants. Code and math then draft deep and prose or multilingual text shallow, where any fixed depth loses one or the
    other. Batches above `ADAPTIVE_BATCH` requests verify the bucket's depth: one of so many requests nearly always wants
    the cap, and graphs at every depth of the wide buckets would cost the throughput profile its spare memory (on one
    Spark under 1 GiB was left, and the memory guard stopped the server at 64 requests). The choice is made before drafting from what the host already has, so drafting is not held up, and every
    rank of a tensor-parallel pair makes the same one."""

    ADAPTIVE_BATCH = 4

    def __init__(self, base, step_cost: float, alpha: float = 0.1, prior: float = 0.8):
        if step_cost <= 0:
            raise ValueError("SHARAR_SPEC_STEP_COST must be positive")
        self.base, self.step_cost, self.alpha, self.prior = base, step_cost, alpha, prior

    @property
    def depth(self) -> int:
        return self.base.depth

    @property
    def depths(self):
        return self.base.depths

    def wanted(self, request, cap: int) -> int:
        accepted, tested = request.spec_rate or (self.prior, 1.0)
        q = min(accepted / tested, 0.999)
        return max(range(1, cap + 1), key=lambda d: (1 - q ** (d + 1)) / (1 - q) / (1 + self.step_cost * (d - 1)))

    def choose(self, requests) -> int:
        cap = self.base.choose(requests)
        if not cap or len(requests) > self.ADAPTIVE_BATCH:
            return cap
        return max(self.wanted(r, cap) for r in requests)

    def observe(self, request, depth: int, accepted: int) -> None:
        """After a round: `accepted` of `depth` drafted tokens; the positions tested are those plus the first rejected."""
        a, t = request.spec_rate or (self.prior, 1.0)
        tested = accepted + (accepted < depth)
        request.spec_rate = ((1 - self.alpha) * a + self.alpha * accepted, (1 - self.alpha) * t + self.alpha * tested)

    def sizes_for(self, depth: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
        """The batch buckets verifying at `depth`: up to ADAPTIVE_BATCH those whose cap reaches it, above it those whose
        cap is it."""
        cap = (lambda b: self.base.depth) if self.base.depths is None else (lambda b: self.base.depths.get(b, 0))
        return tuple(b for b in sizes if (cap(b) >= depth if b <= self.ADAPTIVE_BATCH else cap(b) == depth))

    @property
    def verify_depths(self) -> tuple[int, ...]:
        caps = self.base.depths or {1: self.base.depth}
        small = max((d for b, d in caps.items() if b <= self.ADAPTIVE_BATCH), default=0)
        return tuple(sorted(set(range(1, small + 1)) | {d for d in caps.values() if d}))
