# Chat controls and the single-agent two-path study

The chat contract repair is `3c20397`. The standalone prospective cost/coverage
study is `f4ceb83`. Target, executor, proposer, memory and kernel sources match the
previously validated runtime image; only chat request validation changes.
The API now rejects unsupported output controls. The two-path rule fails its
frozen speed screen: estimated completion reductions are 6.39% and 6.47%,
below the required 10%. No two-path serving mechanism is added.

## Chat API repair

Previously `/v1/chat/completions` silently ignored `tool_choice` and
`response_format`. It now validates both before rendering, tokenization or
engine submission, for streaming and non-streaming requests.

| Control | Accepted values | Other values |
|---|---|---|
| `tool_choice` | Omitted, `null`, `"auto"` | HTTP 400, including `"none"`, `"required"` and named-function choices |
| `response_format` | Omitted, `null`, exactly `{"type":"text"}` | HTTP 400, including JSON object/schema constraints and extra keys |

These accepted values request unconstrained generation. Supplying tools makes
their schema available to the model; automatic selection still does not force a
call or guarantee a particular format. Enforcement requires a separate design
for constraint state, speculative acceptance and rollback. The prior agent-task
format failures are preserved; this validation repair does not turn them into
passes. The prior native `/generate` benchmark did not send these controls.

Forty-two new HTTP tests prove rejection before rendering/submission and valid
streaming/non-streaming output with and without tools. Three additional study
checks cover conditional branch history, correction/bonus accounting and late
continuation/terminal exclusions. The complete CPU suite passes 313 tests with
six GPU-only skips. A first network-isolated suite attempt stalled and was
terminated; the complete run with localhost access passes in 6.07 seconds.

## Results and decision

Both GPU windows complete all twelve control/capture pairs. The target cost
below is measured, with medians over fourteen samples per context/batch across
the two windows. One path uses eight target rows; two paths use sixteen.

| Input tokens | One-path target cost | Two-path target cost | Extra cost |
|---:|---:|---:|---:|
| 8,192 | 88.11 ms | 90.20 ms | 2.09 ms |
| 16,384 | 89.16 ms | 91.63 ms | 2.46 ms |
| 31,616 | 91.46 ms | 94.41 ms | 2.95 ms |

The following coverage and completion figures combine both tasks and windows
at each input/output size, always at **single-request concurrency**. Completion
reduction is an optimistic projection using the frozen formula below; there is
no alternative-rollout TPS measurement.

| Input tokens | Output tokens/request | Extra coverage | Estimated completion reduction |
|---:|---:|---:|---:|
| 8,192 | 512 | 15.19% | 9.94% |
| 8,192 | 1,024 | 12.70% | 8.83% |
| 16,384 | 512 | 15.05% | 8.24% |
| 16,384 | 1,024 | 12.72% | 7.89% |
| 31,616 | 512 | 8.20% | 2.65% |
| 31,616 | 1,024 | 7.38% | 3.01% |


Across each full window, coverage rises from 9,143 to 10,200 potential tokens
(**+11.56%**). After output token 512, it rises from 3,024 to 3,300 (**+9.13%**).
The earlier +23.15% short-screen observation does not persist over these full
continuations. Summed native completion is 393.809 / 394.056 seconds across the
twelve controls in the two windows; estimated two-path completion is 368.630 /
368.565 seconds. The resulting **6.39% / 6.47%** reductions fail the 10% gate.
Near 32K, estimated completion improvement is only about 2.6–3.0%.

The calculation already omits extra branch selection, winning-row handling and
changed-rollout effects. Its failure is sufficient to withhold a serving
implementation; even a projected pass would not establish an actual speedup.
The previous SGLang comparison and DFlash2 promotion status are unchanged.

At 16K, changing the physical batch size changes two target argmax positions
after the first rejection, in both windows. The calculated first-chain emission
is the same in these six cost controls. This limited check does not establish
batch-shape parity or validate the hypothetical alternative continuation.

Forty-eight full generations emit **36,864 tokens** and execute **10,304 native
speculative request-rounds**. All 24 captured runs exactly match their uninstrumented
controls; all twelve unique outputs match across the two reload windows.
The 512-token outputs match the corresponding 1,024-token prefixes in all twelve
budget-pair checks. Recomputing every saved lattice selection and coverage count
passes: **5,112 included observations**, with only 40 terminal rounds excluded
for missing lookahead. All generation/cost releases return 34,999 usable KV
entries and eleven usable state slots. Six cost cells run 252 measured and 72
warmup graph calls and preserve all hashed committed history.

Peak host `MemAvailable` drop is **31.31 / 31.21 GiB**; minima are **81.99 / 82.03
GiB**. These are whole-job host changes, including loading, caches and diagnostic
copies, rather than reconciled process footprints or physical DRAM traffic.
The model image is the prior validated image with a read-only source mount;
this is not a new image build. Both windows restore production's original
image, memory guard and passing health/model/canary checks. The API repair is
committed source; this study does not deploy it to that original image.

## Next work

Keep the failed two-complete-path rule out of serving. The next speed hypothesis
is a fixed **sixteen-target-row tree: one anchor, seven greedy nodes and eight
alternative nodes**. Spread the alternative budget across likely corrections
and share prefix nodes. Freeze selection and evaluate new held-out full
continuations before building a verifier. The low measured incremental target
cost motivates this experiment; tree attention and recurrent-state cost remain
unmeasured.

A viable prototype must preserve GPU-side selection, reusable bounded buffers,
winner-only StateManager publication, correct draft feedback and the existing
concurrency-four path. It must demonstrate actual full-rollout latency and
quality/state/memory gates before serving integration. Output/tool-choice
constraint-state enforcement and a validated image for the API repair are the
separate agent-readiness work; existing task-format failures remain failures.

## Prospective protocol

Two new deterministic repository-review tasks cover memory-capacity incidents
and target-verification ownership. They use different source fixtures and
telemetry from the previous short screen. Inputs, code checksums, selection rule
and decision thresholds are frozen before model outputs. Contexts are 8,192,
16,384 and 31,616 input tokens; output budgets are 512 and 1,024, greedy with EOS
ignored. These are fixed-work reviews without an answer-quality score.

Two separately loaded GPU windows reverse cell order and the order of native
control versus captured continuation. Each window contains twelve pairs. The
captured run saves every draft lattice and native output; it must exactly match
the paired uninstrumented run's output IDs and round counts. Capture timing is
excluded from performance estimates. This is a Sharar mechanism study; SGLang
is not rerun and the previous competitive results are unchanged.

The alternative policy retains the complete seven-token greedy chain. At one
of its first six positions it chooses the highest draft path-mass alternative,
then follows conditional greedy choices through the remaining positions. This
is the same policy as the earlier short two-chain screen. Selection occurs
before target verification, without access to target outputs.

For each native round with eight known subsequent output tokens, count matching
prefix coverage plus one target correction/bonus for each path, and credit the
better path. Native greedy-path coverage must equal actual native emission.
Exclude only terminal observations without the full eight-token lookahead.
Record observations after token 512 separately. Even across a full native
continuation, this is a counterfactual at native states: an alternative rollout
would visit different states and can have different finite-precision outputs.

## Cost and ownership checks

At each context, prepare one real native proposal and its selected alternative.
Measure the existing eight-row verification graph at physical batch one and two.
Both paths read the same committed recurrent slot and immutable prefix KV; the
second path gets eight distinct writable target KV entries. Existing verify
scratch/reconstruction inputs use distinct rows. Hash all committed SSM/conv and
prefix KV bytes before and after the repeated graph runs.

Two warmups and seven paired samples per context/window alternate batch order.
Each timing sample performs three `VerifyGraphs.run()` calls and records wall
time plus separate CUDA event spans. Wall timing includes input uploads, attention
planning and graph replay. Record target argmax differences between batch shapes
instead of assuming numerical equivalence.

The target graphs and maximum-batch-four workspaces already exist. Eight extra
KV entries account for the additional candidate's writable target suffix; no
second committed recurrent slot is allocated for these read-only verification
runs. This does not prove a winner-only commit path. Current StateManager commit
requires unique private destination slots and consumes verifier rows in request
order. A real two-path verifier would need explicit winning-row selection,
accepted-state reconstruction, retained-KV publication, loser release and matching
draft-feature feedback. No duplicated-slot commit is attempted.

## Decision rule

For each cell/window estimate alternative decode time as:

```
(native decode seconds + native rounds * extra target seconds per round)
/ observed candidate-to-native coverage ratio
```

Add measured cold setup/prefill time to estimate completion. The frozen advancement screen
requires at least 10% lower summed completion time in **each** window and no
cell more than 5% worse. It is optimistic: extra selection, metadata, winning-row
handling, draft feedback and changed rollout are not measured. A screen pass
would authorize an isolated verifier prototype, with actual full-rollout speed,
quality/state correctness, memory and protected concurrency-four behavior still
required before any serving integration. A failed screen adds no runtime policy.

Both windows use the pinned DFlash2/target revisions, FP8 target KV, FP32 recurrent
state, seven draft tokens, 35,000 KV entries, twelve state slots, maximum batch
four and a 32,768 context limit. The guard enforces a 64 GiB cgroup/host-memory-drop
budget and 16 GiB host free-memory floor and restores the original production
container and memory guard after each window. Clock/power/temperature samples
remain in the raw evidence; clocks are not locked and physical DRAM traffic is
not measured.

## Reproduction

The small [frozen protocol](../bench/workloads/two_path_v1.json) is committed;
raw inputs, outputs and lattice records stay outside Git. With the pinned model
cache available, freeze before starting either exclusive GPU window:

```bash
python3 -m bench.two_path_study --protocol bench/workloads/two_path_v1.json \
  --freeze /path/to/new-inputs.json
python3 -m bench.two_path_study --inputs /path/to/new-inputs.json \
  --output /path/to/new-window-1 --window 1
python3 -m bench.two_path_study --inputs /path/to/new-inputs.json \
  --output /path/to/new-window-2 --window 2
```

Run each window under the exclusive GPU/memory guard and restore production
between windows. Source snapshots, exact container commands, compressed raw
continuations, all cost samples, memory/clock logs and assertion results are
retained in the external evidence bundle.

The [compact evidence manifest](two-path-study-results.json) identifies the sealed
external bundle by path, byte count and SHA-256.
