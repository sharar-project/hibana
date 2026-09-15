# Prompt-lookup second verification branch: serving measurement

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-14, source `1feb8c6` plus the working-tree lookup branch (opt-in
`--lookup-branch N` / `SHARAR_LOOKUP_BRANCH`, default off). This is step (b)
of roadmap item 1 after the [offline screen](ngram-branch-screen.md). The
result is negative for the 15-row form: served throughput moves by less than
the window-to-window noise, although the mechanism works exactly as designed.
**Decision: not adopted; the implementation is parked on branch `feat/lookup-branch`.**

## What was built

- `sharar/proposers/lookup.py`: a host-side prompt-lookup proposer with an
  incremental n-gram index per request (n from 12 down to 3, most recent
  earlier occurrence, continuation of k tokens).
- `SpecDecoder.prepare` adds one extra verification block per request when a
  match exists and the batch leaves a block free (at most `max_batch` blocks
  per forward, so a full batch of four runs no branch). The block repeats the
  pending token and continues with the looked-up tokens; it reads the same
  committed KV, recurrent state and conv state, and writes its own reserved
  pages (`StateManager.reserve_branch`, `branch_pages`, `resolve_branch`).
- Acceptance picks, per request, the block with the strictly longer accepted
  prefix (ties keep the draft chain). The winning block's features feed the
  drafter, its recurrence inputs reconstruct the committed state
  (`AcceptedState.run(rows=...)`), its conv window and pages are committed;
  the loser's pages return to the pool. Sampled batches never branch.
- Traces record `an`/`g` (the block's acceptance and tokens) whenever a block
  ran and `ad`/`src` when it won; `a` remains the emitted count.
- Seven CPU tests in `tests/unit/test_lookup_branch.py`; the whole CPU suite
  passes (526).

## Protocol

Exclusive GPU windows through `serve_window.py` (production and the memory
guard stopped and restored around each window), the working-tree source
mounted read-only in the runtime image, FP8 pinned DFlash2 at k=15, prefix
cache flushed by a fresh process per window. `bench/draft_ab.py` replayed
three sets at batch one with the recorded output budgets, greedy: the 29
repository-coding test prompts (`traces-test`, all), the 24 newest held-out
ShareGPT prompts (`traces`), and the five real-traffic requests with features
(`$SHARAR_RESULTS/traces`). Arms: `base` (branch off), `branch` (on),
`branch2` (on, traces recorded for the audit). The branch arm reproduced
itself exactly between its two runs (same rounds, same counters), so the TPS
spread between them (about 1%) is the window noise.

## Results

Aggregate decode throughput is total tokens over total wall time; tokens per
round come from the server's counters over each set.

| Set | Requests | Base TPS | Branch TPS (run 1 / run 2) | Change | Base tok/round | Branch tok/round | Change | Identical outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Repository-coding test | 29 | 45.44 | 45.64 / 45.27 | +0.5% / −0.4% | 4.884 | 4.954 | +1.4% | 3 / 29 |
| ShareGPT held-out | 24 | 35.99 | 36.88 / 36.53 | +2.5% / +1.5% | 3.714 | 3.816 | +2.7% | 5 / 24 |
| Real traffic | 5 | 37.08 | 38.12 / 37.34 | +2.8% / +0.7% | 3.843 | 3.988 | +3.8% | 2 / 5 |

Round cost is unchanged within noise: 102.5 ms mean per round in the base
arm against 102.6 and 103.2 ms in the branch arms, with the branch firing in
11.3% of rounds (a firing round verifies 32 rows instead of 16).

Per set, the branch fired in 16.0% / 5.5% / 15.5% of rounds and won 6.6% /
0.8% / 2.0% of rounds, adding 1,615 / 71 / 88 accepted tokens. Outputs
differ from the base arm in most requests (median first divergence at token
94 / 159 / 42): a firing round changes the verify row count, and the target's
arithmetic depends on it ([performance limits](../../performance.md#limits)),
so comparisons are equal-length, not equal-output.

## Why the offline screen was wrong by an order of magnitude

The audit (`audit.py` in the evidence directory) replays the serving lookup
offline on the served outputs at the served anchors: it fires in the same
1,028 of 9,106 rounds, proposes the same tokens in every one of them, and
matches the served block acceptance (6.19 versus 6.12 mean). The serving
path is correct. Two things inflated the screen:

1. **The recorded histories are not the served histories.** For the same
   prompts, recorded and served greedy outputs diverge at a median of token
   25. In copy regions (an offline match of eight or more tokens) the served
   draft chain accepts a mean of 10.5 tokens (median 11) against 8.4 in the
   recorded rounds, and served rounds land in copy regions less often (8.8%
   versus 12.2% of rounds). A 15-wide branch can only beat a chain that is
   already near the 16-token cap by a few tokens.
2. **A token gained in a copy region is a cheap token.** The per-round union
   values every extra accepted token at the average round (about 4.9
   tokens). But the branch wins inside copy spans, where the draft chain
   would have accepted the same tokens in the following round at 11 to 15
   per round. In the repository set the branch added 1,615 accepted tokens,
   which the oracle values at about 330 saved rounds; the served run saved
   57. The requests with the largest gains (whole-file rewrites, up to 478
   extra tokens each) already ran at 15.7 tokens per round, the cap.

## What would change the answer

The branch cannot add much while both blocks share the 16-row cap. A wider
branch (31 or 63 rows, with the draft block padded to the same width or a
non-uniform verify shape) lifts the cap only in copy regions: on the served
repository histories copy-region rounds are about 9% of rounds and carry
about 20% of tokens, so even an unbounded branch saves at most about 7 to 8%
of rounds at batch one before its cost (a 64-row forward in firing rounds,
roughly +20 ms each). Whole-file reproduction requests would speed up
several-fold, everything else not at all. That is a workload bet, not an
engine improvement, and it is not worth the non-uniform verify shapes on the
evidence here. The idea should be revisited only if real traffic turns out to
be dominated by verbatim reproduction.

## Decision

Not adopted. The implementation (opt-in flag, default off, seven CPU tests)
is parked on branch `feat/lookup-branch` from the gate commit; the roadmap
item is closed with this result. The offline
screen protocol (`bench/ngram_branch_screen.py`) remains useful for fire
rates but must not be read as a throughput forecast: any per-round oracle on
recorded histories over-values tokens that fall in regions the draft chain
already handles.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-14-ngram-branch/`: `serve_window.py`,
`ab_client.sh`, `compare.py`, `audit.py`, the three arms' `ab/*.json` (per
request tokens, seconds, tokens per round, output hashes and IDs),
`*-final-server-info.json`, `traces-branch2/` (served rounds with `an`/`g`),
`audit-branch2.txt`, and the combined `results-serving.json` named in the
[manifest](lookup-branch-results.json).
