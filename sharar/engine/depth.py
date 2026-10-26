"""Draft depth policies: how many drafted tokens a round verifies.

`SpecDecoder` asks its policy once per round; the answer must be one of the
depths whose verify graphs were captured (`SpecDecoder.verify_depths`).
`FixedDepth` is the served behavior: the proposer's k every round.
`BatchDepth` chooses by the batch bucket: the throughput profile verifies
shallower blocks, or none (depth 0: plain decode), when many requests share a
round. `AcceptanceDepth` (SHARAR_SPEC_STEP_COST) drafts each request as deep as its recent acceptance pays for.
`RowBudget` (SHARAR_SPEC_POLICY=budget, DFlash2 in `fast` and `throughput`) verifies the drafts whose scores pay for
their measured cost.
"""

from __future__ import annotations

from collections import Counter

import torch

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
    """Adaptive depth: each request drafts as deep as its own recent acceptance pays for.

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


def ragged_budgets(sizes, steps: int) -> dict[int, tuple[int, ...]]:
    """The row totals packed verify graphs are captured for, per batch bucket: multiples of the bucket from one row a
    sequence (plain decode) to `steps`, and 128 (the 27B's 3-bit copies' limit) where it falls between."""
    out = {}
    for b in sizes:
        rows = {b * m for m in (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16) if m <= steps} | {b * steps}
        if b < 128 < b * steps:
            rows.add(128)
        out[b] = tuple(sorted(rows))
    return out


class RowBudget:
    """Adaptive verification for a drafter that scores its drafts (DFlash2, SHARAR_SPEC_POLICY=budget): a round
    verifies the drafted tokens whose expected acceptance pays for their measured cost.

    - Scores: `Proposal.acceptance`, each draft's calibrated probability of acceptance given the drafts before it,
      with a log-odds shift per position learned from the positions the target tested. Request i's draft j is worth
      its survival S_ij = q_i1 ... q_ij: acceptance stops at the first rejection.
    - Cost, per batch bucket and verified rows: the verify graphs timed after capture (`SpecDecoder.measure_costs`),
      plus what they do not cover (drafting the block, acceptance, feedback, longer contexts), a moving mean of each
      round's GPU time beyond its graph.
    - Rule: verify the seats that maximize sum(S) - rate * cost(rows), rate the bucket's measured tokens per ms (each
      round's tokens / cost until one is measured).

    The round verifies packed (`packed_rows`, spec_graphs.RaggedVerifyGraphs): seats go to each request in order of S,
    at the captured row total that maximizes the rule, so a confident request verifies deep and an unsure one shallow.
    One scored round in `EXPLORE` verifies every draft, so deep positions keep being tested.

    Sampled batches verify one depth chosen before drafting (`choose`) from the drafts' survival per position over past
    rounds (accepted / tested; an untested position is not a rejection; greedy and sampled kept apart), so rejection
    sampling stays exact; greedy batches with a constrained request verify every request at the cap. A depth chosen
    from scores reads only the drafts before each position. Until the costs are measured every round verifies the cap."""

    depths = None
    EXPLORE = 16

    verify_depths = ()  # packed graphs hold every depth

    def __init__(self, cap: int, sizes, *, packed_rows: dict[int, tuple[int, ...]], caps: dict[int, int] | None = None,
                 keep: float = 0.999, prior: float = 0.8, prior_weight: float = 4.0):
        self.depth = cap
        self.steps = cap + 1
        self.sizes = tuple(sizes)
        self.caps = {b: max(1, min(cap, (caps or {}).get(b, cap))) for b in self.sizes}  # a ceiling per bucket (SHARAR_SPEC_DEPTHS)
        self.budgets = packed_rows  # bucket -> captured row totals
        self.candidates = tuple(range(1, cap + 1))
        self.keep = keep
        self.costs: dict[tuple[int, int], float] = {}  # (bucket, rows) -> ms of the verify graph
        self.overhead: dict[tuple[int, bool], float] = {}  # (bucket, sampled) -> ms a round takes beyond its verify graph
        self._fit: dict[tuple[int, bool], tuple[float, float]] = {}
        self.rate: dict[tuple[int, bool], float] = {}  # (bucket, sampled) -> achieved tokens per ms
        self.survival = {s: ([prior * prior_weight] * cap, [prior_weight] * cap) for s in (False, True)}
        self.shift = [0.0] * cap
        self.chosen: Counter = Counter()  # (bucket, depth) -> one-depth rounds (sampled or constrained) at that depth
        self.chosen_rows: Counter = Counter()  # (bucket, rows) -> scored rounds that verified that many rows
        self._scored = 0
        self._round = None  # the last scored round's {id(request): its positions' probabilities}

    def sizes_for(self, depth: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
        return sizes

    def rows(self, bucket: int, b: int, depth: int) -> int:
        """The rows a round of b requests at `depth` verifies in a graph of `bucket`."""
        needed = b * (depth + 1) + bucket - b  # the bucket's padded sequences take a row each
        return next(r for r in self.budgets[bucket] if r >= needed)

    # A saving smaller than this (ms, or this share of the full round) is timing noise: one request on Flash-Next verifies
    # 1 or 9 drafts in 16.6-18.1 ms (not even monotone), and verifying fewer there lost 4% of the tokens for no shorter round
    NOISE_MS, NOISE_SHARE = 1.0, 0.03

    def cost(self, bucket: int, rows: int, sampled: bool = False) -> float:
        """Expected ms of a round: the verify graph and the fitted rest. Rows saving less than the noise floor against
        the bucket's full verification cost as much as it, so the rule verifies every draft where rows are free."""
        rest = self.overhead.get((bucket, sampled), 0.0)
        full = self.costs.get((bucket, self.full_rows(bucket)))
        own = self.costs[bucket, rows]
        if full is not None and full - own < max(self.NOISE_MS, self.NOISE_SHARE * (full + rest)):
            own = max(own, full)
        return own + rest

    def full_rows(self, bucket: int) -> int:
        """The rows a bucket's full verification takes: every request at the cap."""
        return self.rows(bucket, bucket, self.caps[bucket])

    def expected_tokens(self, depth: int, sampled: bool = False) -> float:
        """Tokens a request emits per round at `depth` from past rounds: 1 + sum_{j<=depth} P(at least j accepted)."""
        accepted, tested = self.survival[sampled]
        tokens, survive = 1.0, 1.0
        for j in range(depth):
            survive *= accepted[j] / tested[j]
            tokens += survive
        return tokens

    def choose(self, requests) -> int:
        """Before drafting: the depth to draft. A greedy batch drafts the whole block, scored and allotted after (one
        forward costs about the same at any depth); a sampled one verifies one depth from past rounds. Idempotent: the
        executor asks too (a depth of 0 would decode plainly; this policy never returns one)."""
        self._round = None
        b = len(requests)
        bucket = graph_set.pick(self.sizes, b)
        cap = self.caps[bucket]
        if not self.costs or all(r.params.greedy for r in requests):
            return cap
        return max((d for d in self.candidates if d <= cap), key=lambda d: b * self.expected_tokens(d, True)
                   / self.cost(bucket, self.rows(bucket, b, d), True))

    def allot(self, requests, acceptance, depth: int) -> tuple[list[int], int]:
        """After drafting `depth`: each request's limit and the rows the round verifies (acceptance None: every request
        at `depth`, as for sampled or constrained batches)."""
        b = len(requests)
        bucket = graph_set.pick(self.sizes, b)
        depth = min(depth, self.caps[bucket])
        if acceptance is None or not self.costs:
            if acceptance is None:
                self.chosen[bucket, depth] += 1
            return [depth] * b, self.rows(bucket, b, depth)
        shift = torch.tensor(self.shift[:depth], device=acceptance.device)
        q = torch.sigmoid(torch.logit(acceptance[:, :depth].float().clamp(1e-6, 1 - 1e-6)) + shift)
        rows_q = q.tolist()  # one wait for the draft
        self._round = {id(r): row for r, row in zip(requests, rows_q)}
        self._scored += 1
        if self._scored % self.EXPLORE == 0:
            return [depth] * b, self.rows(bucket, b, depth)
        rate = self.rate.get((bucket, False))
        value = lambda tokens, rows: tokens / self.cost(bucket, rows) if rate is None else tokens - rate * self.cost(bucket, rows)  # noqa: E731
        survival = []
        for row in rows_q:
            run, out = 1.0, []
            for p in row:
                run *= p
                out.append(run)
            survival.append(out)
        gains = sorted(((g, -j, i) for i, out in enumerate(survival) for j, g in enumerate(out)), reverse=True)
        prefix = [0.0]  # within a request values never increase: ties keep positions in order
        for g, _, _ in gains:
            prefix.append(prefix[-1] + g)
        best, best_value = None, None
        for rows in self.budgets[bucket]:
            if rows < bucket:
                continue
            extra = min(rows - bucket, len(gains))
            v = value(b + prefix[extra], rows)  # the draft is spent: verify rows only
            if best_value is None or v > best_value:
                best, best_value = (rows, extra), v
        rows, extra = best
        self.chosen_rows[bucket, rows] += 1
        limits = [0] * b
        for _, _, i in gains[:extra]:
            limits[i] += 1
        return limits, rows

    def observe(self, request, depth: int, accepted: int, rate: float = 0.02, limit: float = 4.0) -> None:
        """A request's round: `accepted` of its `depth` drafts; it tested positions up to the first rejected."""
        sampled = request is not None and not request.params.greedy
        acc, tested = self.survival[sampled]
        for j in range(len(tested)):
            acc[j] *= self.keep
            tested[j] *= self.keep
        for j in range(min(accepted + 1, depth, len(tested))):
            tested[j] += 1
            if accepted > j:
                acc[j] += 1
        row = self._round.pop(id(request), None) if self._round is not None else None
        if row is None:
            return
        for j in range(min(accepted + 1, depth, len(row))):
            outcome = 1.0 if accepted > j else 0.0
            self.shift[j] = max(-limit, min(limit, self.shift[j] + rate * (outcome - row[j])))

    def observe_round(self, batch: int, rows: int, ms: float, tokens: int | None = None, *, sampled: bool = False,
                      keep: float = 0.95, weight: float = 0.05) -> None:
        """A finished round's GPU time (events, read once the round completed), its rows and tokens: the bucket's rate
        and the time beyond its verify graph, a moving mean (rows follow the drafts' scores, so the round time cannot
        tell a cost per row from a cost per token; the timed graphs carry the rows)."""
        bucket = graph_set.pick(self.sizes, batch)
        key = (bucket, sampled)
        if tokens is not None and ms > 0:
            previous = self.rate.get(key)
            self.rate[key] = tokens / ms if previous is None else (1 - weight) * previous + weight * tokens / ms
        if (bucket, rows) not in self.costs:
            return
        n, total = (keep * v for v in self._fit.get(key, (0.0, 0.0)))
        self._fit[key] = (n + 1, total + ms - self.costs[bucket, rows])
        self.overhead[key] = self._fit[key][1] / self._fit[key][0]
