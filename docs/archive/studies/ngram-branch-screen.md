# Prompt-lookup n-gram branch coverage screen

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-14, source `1feb8c6` plus `bench/ngram_branch_screen.py`. CPU only;
no model, GPU or serving change. This is the offline step (a) of roadmap
item 1: would a second, independent proposal branch built by prompt lookup,
placed in the verify rows that DFlash2 leaves unused, add accepted tokens per
round?

## Protocol

Recorded traces hold the prompt IDs, the output IDs and every verification
round (accepted count, proposals, target argmax row). For each round the
screen rebuilds the context at the round's anchor (prompt plus output so
far), finds the most recent earlier occurrence of the context's longest
suffix n-gram (12 down to 3 tokens) and takes the tokens that followed it as
the n-gram branch. The branch's match is its longest prefix equal to the
continuation the target actually produced. Two branches rooted at the same
pending token accept `max(a_dflash, a_ngram)` tokens plus one correction:
the prefix-plus-correction protocol of the
[tree coverage study](tree-coverage-study.md). Branch widths 15, 31 and 63
rows were evaluated on the same pass.

This is a coverage oracle on fixed histories, not a serving measurement. Real
rounds re-anchor after every union, the target's numerics change with the row
count ([performance limits](../../performance.md#limits)), and the branch
costs verify rows only in the rounds where a match of at least three tokens
exists. The earlier two-path screen translated an 11.6% coverage gain into a
6.4% completion estimate; expect a similar discount here.

## Results

Tokens per round, mean over recorded rounds; "fires" is the fraction of
rounds with an n-gram match.

| Corpus | Requests | Rounds | Fires | DFlash2 alone | Union, 15 rows | Union, 31 rows | Union, 63 rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repository-coding proxy (`traces-repo2`, pinned draft) | 399 | 52,151 | 28.7% | 5.162 | 5.889 (+14.1%) | 7.743 (+50.0%) | 10.850 (+110%) |
| Serve-train loop (`traces-loop`, drafts across four sessions) | 747 | 89,511 | 29.4% | 5.481 | 6.117 (+11.6%) | 7.911 (+44.3%) | 10.937 (+99.5%) |
| ShareGPT first turns (`traces`) | 599 | 92,769 | 12.6% | 3.815 | 3.835 (+0.5%) | 3.860 (+1.2%) | 3.873 (+1.5%) |
| First real traffic (`$SHARAR_RESULTS/traces`) | 7 | 1,025 | 31.4% | 4.141 | 4.192 (+1.2%) | 4.192 (+1.2%) | 4.192 (+1.2%) |

The n-gram branch beats the DFlash2 chain in 11 to 14% of coding rounds and
in about 1% of chat rounds. When it wins on coding it wins by a lot: the
31-row branch alone averages 5.0 tokens per round over all rounds, which
means matches of twenty or more tokens are common. The proxy tasks are
copy-heavy (two of the five ask for the complete updated file or quoted
lines), so real coding traffic should sit between the proxy and the chat
rows; the seven real requests are too few to place it.

## Decision

Advance to a serving prototype (roadmap item 1, step b): DFlash2 chain plus
the n-gram continuation as a second branch from the pending token, 32 rows
only when a match exists, on top of the tree verifier and causal-bounded
tree attention kept on the `bench/tree-verifier` and `bench/tree-native-mask`
branches, reusing `proposers/ngram.py`. Judge it on the 18-family set and on
served throughput over the repo proxy and the newest real traces with the
stable-numerics baseline, not on the three-prompt benchmark. A 31-row branch
is the first width to try because rows are nearly free and the coding gain
triples between 15 and 31 rows; a 63-row branch needs a wider verify graph
and should wait for the 31-row result.

## Outcome

The serving prototype was built and measured the same day
([lookup branch study](lookup-branch.md)): +1.4% tokens per round and
throughput within window noise on the repository test set. The per-round
union above over-values tokens in copy regions, where the draft chain is
already near the 16-token cap and would accept the same tokens in the next
round; read this screen as a fire-rate estimate, not a throughput forecast.

## Evidence

Raw per-corpus results and the combined file live under
`$SHARAR_RESULTS/reviews/2026-09-14-ngram-branch/` (`real-traffic.json`,
`repo-coding.json`, `serve-train-loop.json`, `sharegpt.json`,
`results.json`); the compact manifest is
[ngram-branch-screen-results.json](ngram-branch-screen-results.json).
Reproduce with:

```bash
python3 bench/ngram_branch_screen.py $SHARAR_RESULTS/reviews/2026-09-11-draft-traces/traces-repo2 --widths 15 31 63
```
