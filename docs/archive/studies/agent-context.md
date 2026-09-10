# Long-session cache work on GB10

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

This starts Phase 8 with useful history reuse and sustained generation. The
supported profile remains Qwen3.8-27B on one GB10, at a maximum total context of
32,768 tokens. DFlash2 default promotion remains blocked by the earlier frozen
quality evaluation; this work does not change the production proposer.

## Repairs

- Reserve each prefill row's latest interior checkpoint before earlier interval
  checkpoints. The 12-slot pool previously filled from the start of a long
  prompt, dropping the near-end boundary needed by an identical retry.
- Recycle only older, completed, request-owned checkpoints. Pending destinations
  in the same forward cannot alias, and an exact prompt-end copy cannot replace
  the last interior checkpoint. Batched reservations give every row priority
  for its endpoint before filling earlier intervals.
- Budget scheduler admission from actual prompt IDs and the uncached suffix,
  including decode reserve. The previous length-only check evicted reusable
  history before admission: the baseline scheduler retained only 2,048 tokens
  on a repeated 32,766-token prompt.
- Reclaim an unlocked interior recurrent snapshot when only state slots are
  exhausted, preserving KV pages and newer resume points. Obtain a private slot
  before locking the matched path. The baseline direct executor raised
  `MemoryError: state pool exhausted` on the fourth growing-history turn.
- Attach matching draft metadata when adding a snapshot to an existing radix
  node. Preserve existing metadata when discarding a duplicate snapshot.

These changes use the existing pools and introduce no serving policy flags.
StateManager owns reservations; RadixCache owns cached snapshots; proposers own
draft features. Snapshot reclamation returns allocator capacity, not physical
DRAM from the monolithic pool. Physical reclamation remains separate Phase 8
work. [Contributor contracts](../../interfaces.md) describe the ownership rules.
Runtime size changes from 15,100 to 15,154 Python lines; `executor.py` shrinks
from 493 to 487 lines. The additional allocation policy stays with its owner.

## Protocol and scope of the measurements

The baseline is runtime commit `395feb1`, unchanged through `71e017f`, in clean
image `2addcbfc71a4`. Both runs use DFlash2 depth 7, accepted-state reconstruction,
CUDA graphs, FP8 target KV, FP32 recurrent state, 35,000 KV entries, 12 state
slots, concurrency one, eight CPU cores and four OpenMP threads. GPU work is
exclusive, with a 64 GiB container/physical-drop budget and a 16 GiB host floor.
The guard restores and checks the original production container after each job.

`bench/agent_context.py` fixes 8,192 / 16,384 / 32,256 input tokens and 512
generated tokens per request, with two cold/retry pairs per length. It also
checks 32,766 input plus two output tokens, full and 7,000-token chunked prefill,
and four append-only turns starting at 8,192 tokens. Each agent turn appends the
actual 512-token response and 128 further input tokens. A separate pair of
repeats runs through the same Scheduler used by serving.

The input is a repeated paragraph, and EOS is ignored to measure fixed work.
This is a cache/lifecycle and sustained-generation diagnostic, not an agent
task-quality benchmark. Append-only histories contain each profile's actual
outputs and are recorded individually. Decode TPS is **per request**, counts
the 511 tokens after the prefill token, and excludes prefill time. Aggregate
TPS includes prefill and is also recorded. Timings exclude model startup.
Direct cache-hit TTFT includes the diagnostic restored-state equality checks;
the Scheduler timings have no such observer checks and are the serving-path
measurement. Neither includes HTTP transport overhead.

Inside the pinned GPU environment, with exclusive device access and the stated
memory guard, reproduce the candidate study with:

```bash
python3 -m bench.agent_context --expect-retention \
  --output "$SHARAR_RESULTS/agent-context-repeat"
python3 -m bench.prefill_partitions \
  --inputs "$SHARAR_RESULTS/reviews/2026-09-06/promotion/inputs.json" \
  --output "$SHARAR_RESULTS/prefill-partition-repeat"
```

Omit `--expect-retention` for the old runtime. `--scheduler-only` runs the
separate serving-admission comparison. Exact container commands and source
hashes are archived with the evidence.

## Results and validation

Implementation: `ba77a5f`. The clean image is `4a7206b66ead`; all 95 runtime
source hashes match. It passes **584 CPU/GPU tests**, including exact reordered
GDN checkpoint/state comparisons, four-row long-prefill endpoint retention,
cache loans, eviction, cancellation, replay and speculative lifecycle checks.
The standalone CPU run passes 260 tests with six GPU-only skips.

The candidate completes 24 measured requests. The baseline completes 23 across
its direct and scheduler runs, and fails its fourth direct agent turn. All 19
comparable completed direct requests have identical input IDs and output IDs.

Each fixed-work row below is a two-repeat median, with **512 output tokens per
request, concurrency one**. Decode rates exclude prefill and its first token.

| Input tokens | Cache state | Before TTFT (s) | After TTFT (s) | Before decode tok/s | After decode tok/s |
|---:|---|---:|---:|---:|---:|
| 8,192 | cold | 3.793 | 3.811 | 70.86 | 70.73 |
| 8,192 | retry | 0.129 | 0.142 | 71.40 | 71.09 |
| 16,384 | cold | 7.811 | 7.839 | 67.18 | 67.69 |
| 16,384 | retry | 0.145 | 0.132 | 70.23 | 70.07 |
| 32,256 | cold | 20.940 | 21.072 | 67.49 | 67.56 |
| 32,256 | retry | 7.535 | 0.154 | 67.96 | 68.50 |

For the 32,256-token repeat, recomputation falls from **11,776 to 64 tokens**.
TTFT improves 48.99×, and total completion falls from **15.05 to 7.61 seconds**
for the same 512 output tokens. Cold prefill and generation rates are broadly
unchanged. The short cached TTFTs move in both directions (8K rises about 13 ms;
16K falls about 13 ms); these two-repeat diagnostics do not establish a general
small-context latency improvement.

The uninstrumented **serving Scheduler** test uses **32,766 input / 2 output
tokens, concurrency one**. Before: 14.457 / 14.499 seconds to first token,
retaining only 2,048 tokens. After: **0.146 / 0.145 seconds**, retaining
**32,704 tokens** and completing one speculative round. This roughly 99× TTFT
improvement measures prefix admission and retention, not an arithmetic-kernel
speedup or a new comparison against SGLang.

Both full and chunked near-32K retries restore aligned draft state and recompute
at most 64 tokens (62 for full prefill, 30 for 7,000-token chunks). All four
append-only agent turns complete with draft decoding active and no missing-draft
fallbacks. After the initial turn, TTFT is **0.150–0.160 seconds**, recomputing
129 tokens per turn. The baseline raises `MemoryError` at turn four; the failed
run and its original traceback remain in the evidence.

The guarded candidate window reaches a minimum **75.02 GiB MemAvailable**,
within the 64 GiB test budget and above the 16 GiB floor. This does not validate
external CPU/tool contention or physical cache reclamation. The production
image and default remain subject to the separate promotion gates.
The original image `b893cdc2e317` was restored; the memory guard is active and
health, model identity and canary checks passed. The [compact evidence
manifest](agent-context-results.json) indexes 90 sealed artifacts outside Git.

## Target-prefill trace

`bench/prefill_partitions.py` preserves production arithmetic while observing
fixed-token last-row projections, layer outputs, recurrent state, selected
draft features and logits. Observer copies synchronize, so trace wall times
are not performance measurements. Five partitions of the same four 128-token
prompts are repeated twice; a separate control uses paged attention even for
one-row prefills. A common one-row vocabulary projection separates target
hidden-state drift from the vocabulary projection's batch shape.

The native partitions reproduce the earlier first-token changes. Their first
observed layer divergence occurs in layers 0–2, before the first full-attention
layer. A fixed-operand probe finds identical full first-layer FP8 QKV output
at 512 versus 128 rows, but different BF16 b/a gating projections: maximum
absolute difference 0.015625, relative RMS about 0.011%. Both shapes have
similar error against an FP32 projection reference. The small early difference
can be amplified through the model; final greedy token changes are not confined
to tiny final-logit margins.

All five corrected paged-attention partitions reproduce their observed tensors
exactly on repetition. Forcing paged attention changes some recorded tensors in
single-row groups, but preserves every partition's first-token IDs. Changing
the attention path alone therefore does not remove this first-token variation.

This identifies a concrete numerical sensitivity, not proof that one partition
is more correct or that it alone explains every acceptance tail. No batching
delay, preferred partition, width policy or alternative numerical kernel is
added to serving. A bounded fixed-b/a numerical control is the next follow-up.

The initial exploratory trace's `paged-*` labels did not force attention
dispatch: installing a wrapper alone leaves a cold single request on ragged
attention. Those artifacts are retained and explicitly excluded as an attention
control; the corrected run sets one-row batched context metadata as well.

## Remaining work

Extend from fixed paragraphs to held-out agent tasks with repeated tool turns,
branching, pauses, cache churn and concurrent agents, then run the same profiles
against SGLang. Profile sustained generation at 8K/16K/32K, screening attention/KV
traffic, recurrent-state movement and accepted tokens per target traversal.
Require measured quality, memory and latency results before adding an algorithm
or runtime option. Validate 64K/128K profiles explicitly before claiming support.
Internal pressure/admission control, reclaimable physical cache slabs and shared
CPU/tool-load experiments follow; targeted SM121 kernels can proceed alongside
them, with final AOT packaging after the profiles stabilize.

Raw tensors, outputs, failed runs and resource logs stay under
`$SHARAR_RESULTS/reviews/2026-09-06/agent-context`. The accompanying compact
manifest links the sealed evidence bundle.
