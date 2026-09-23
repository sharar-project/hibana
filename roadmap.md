# Sharar roadmap

Updated 2026-09-17. This is the current work order. Historical milestones and
their original gates are in the [archived roadmap](docs/archive/roadmap-2026-09-11.md).

## Supported baseline

One GB10, the pinned Qwen3.8-27B NVFP4 target, 32K context, maximum batch four.
DFlash2 with 15 proposals and accepted-state reconstruction; FP8 draft by
default since 2026-09-17 (BF16 is the publisher's precision); relaxed acceptance at top-2 within
1.0 logit for greedy requests that do not choose; 3-bit decode copies of the
layers and the draft on by default in the launcher (the lm_head copy opt-in).
Activation-aware calibrated target copies are available through
`SHARAR_W3_CHECKPOINT`; paired accuracy and the family performance gate pass.
The NVFP4/FP8 prefill weights, vocabulary head and verifier stay fixed.
The DFlash2 drafter is trainable on recorded
traffic and serving follows the promoted checkpoint; the publisher's draft is
the fallback. See [operations](docs/operations.md), [performance](docs/performance.md),
the [optimization inventory](docs/optimizations.md) and [draft adaptation](docs/draft-adaptation.md).

## Current state

- W3 calibration: GPTQ fitting from the pinned FP8 release improves full
  GSM8K from 92.04% to 94.47% and family decode throughput by 8.6%
  [5.1%, 12.3%] against the current nearest-rounded W3 profile. The kernel
  and stored weight size are unchanged. See the
  [calibration and adaptation study](docs/archive/studies/w3-calibration-adaptation.md).
- Calibrated-target drafter adaptation: 300 steps over 240 coding-proxy
  requests, with separate validation and test files, selected step 200.
  Actual FP8/W3 serving improves by 3.60% at concurrency one and 4.43%
  at concurrency four. The updated runtime, calibrated target and selected
  drafter are active on the local GB10 since 2026-09-17; live generation and
  artifact identities were checked, and the previous container is retained
  for rollback.
- Engine: exact BS1 QKV/MLP integration (+1.43% family decode), k=15 with
  2,048-token prefill chunks and a 4,096-token prefill budget, checkpoint-owned
  draft features, shared GDN prefill planning, accepted-state reconstruction,
  prefix caching, managed-memory cold storage and the host memory guard.
  Narrower verification, mixed forwarding, numerical kernel replacements and
  scheduler variants failed their gates ([decision index](docs/archive/README.md)).
- Draft adaptation: opt-in traces with serving-exact features, `sharar train`
  (checked against the serving proposer, the publisher's model and SpecForge),
  trained-draft serving, `sharar-ctl` and the start/stop launcher. On a
  repository-coding workload a 1,000-request serve-then-train loop raised
  serving throughput 12.4% ([report](docs/archive/studies/gb10-draft-training.md)).
  Continuation sessions use a smaller rate and a fixed regression set in the
  promotion gate; on-policy selector supervision is available (`--on-policy`).
- API: chat with auto/null/none tool choice, thinking, streaming; forced tool
  choice and constrained JSON output remain unsupported.
- Known property: the target's prefill numerics depend on sequence length and
  chunking, so greedy continuations can change with batch shape, verification
  width or the drafter ([performance](docs/performance.md#limits)).

## Harness-native serving (2026-09-23), ahead of the list below

Production served no requests in three days and the trace sink holds two real
files, so the adaptation loop has no traffic to learn from. Agent harnesses
(Claude Code, Codex, OpenCode) are the way to daily use; they need long
context, their own APIs, fast sampled decoding and structured output. In order:

1. **Done 2026-09-23: long context, 262,144 tokens on the 27B, the launcher
   default.** `--context-length` / `SHARAR_MAX_CONTEXT`; the plan guarantees one
   request at that length or four at 32K. Measured with `bench/long_context.py`
   (evidence `$SHARAR_RESULTS/reviews/2026-09-23-long-context/`): needle
   retrieval correct at three depths at 10K, 32K, 64K, 128K and 262K; cold
   prefill 3.8 / 15 / 39 / 112 / 351 s; exact decode 50 / 54 / 41 / 39 / 30
   tok/s with rounds of 70 / 75 / 82 / 94 / 119 ms (the 10K round unchanged).
   Found and fixed on the way: chunked prefill (2,048-token chunks, equal to
   the snapshot interval) never snapshotted interior boundaries, so a prompt
   edited more than 64 tokens before its end re-prefilled from scratch (113 s
   at 128K); chunk ends are now snapshotted, at most 16 per prompt plus every
   chunk end in the last 8K. Next: the NVMe session tier, since a cold 262K
   prefill takes six minutes.
2. **Done 2026-09-23: Anthropic Messages API.** `/v1/messages` and
   `count_tokens` over the chat route, `tools/claude-sharar.sh`; a headless
   Claude Code session fixed a two-bug repository in 16 s (6 turns, 7.7 tokens
   per round, first-token median 0.37 s after the first 16.5K-token prompt).
   Harness fixes in the same change: system messages inside the conversation,
   OpenAI tool-call arguments as JSON strings, cached tokens in chat usage.
   Open: map Claude Code's `output_config.effort` onto the template's reasoning
   effort; `/v1/responses` only if Codex is wanted.
3. **Done 2026-09-23: sampled decoding, measured** (evidence
   `$SHARAR_RESULTS/reviews/2026-09-23-sampled/`). The top-p nucleus now ranks
   leading candidates instead of sorting the vocabulary (identical output):
   the sampled round's overhead over greedy fell from 9.1 to 5.6 ms (74.3 vs
   68.7 ms at batch one); the rest is the full-vocabulary softmax and the
   per-request verifier. Relaxed acceptance for sampled requests (a drafted
   token among the request's own top-k within the margin) is built, opt-in
   per request and off by default: at temperature 0.6 it gave +4% throughput
   and lost quality, GSM8K-300 96.0 to 94.0% (2 better, 8 worse) and HumanEval
   94.5 to 91.5% (2/7), pooled 4 better against 15 worse (p about 0.02). A
   sampled request's drafts are the draft's own samples, so relaxing accepts
   tokens the target would not have chosen; a faster sampled path has to come
   from the draft side (proposing the target's likely tokens) or from the
   remaining 5.6 ms. Not pursued further now.
4. **Done 2026-09-23: NVMe session store** (`sharar/memory/session_disk.py`,
   evidence `$SHARAR_RESULTS/reviews/2026-09-23-sessions/`). Parked sessions
   span slabs (long contexts no longer rejected), are written through to
   `$SHARAR_RESULTS/sessions`, and are persisted at shutdown. Restart test on a
   54.8K-token conversation: turn one cold 31.1 s to first token; after a
   server restart turn two 1.68 s, with 54,795 prompt tokens restored from
   disk (1.4 s load of 1.84 GiB, 24 ms restore). Turn two of a chat reuses
   exactly turn one's prompt plus output (checked with and without thinking),
   which is the snapshot a record carries. Restores now read straight into
   pool pages (first writes into fresh managed slabs ran at 1.5 GB/s): the same
   restart gives 0.57 s to first token, 321 ms of disk restore. Follow-ups: records
   carry only their end snapshot, so an edit earlier in a stored conversation
   still re-prefills (add the prompt-end and last chunk-end snapshots); first
   prefills are unchanged (attention at long context is the lever: about 250
   of the 351 s at 262K).
5. **Test-time scaling, first result (2026-09-23).** `n` choices share one
   prefill (a one-token primer publishes the prompt; 4 choices of a 6.4K
   prompt prefilled 6,635 tokens in total). `bench/tts_eval.py` on 200 MATH-500
   problems, thinking off (evidence `$SHARAR_RESULTS/reviews/2026-09-23-tts/`):
   greedy 77.0% in 594 s (four problems in flight, 352 tok/s aggregate);
   self-consistency n=4 at T 0.7 83.0% (16 better, 4 worse) in 3,451 s;
   n=8 86.5% (21/2) in 6,860 s; level 5 58.6 -> 72.4 -> 75.9%. pass@8 is
   87.5%, one point above the vote: at this scale a verifier or reward-weighted
   particles could add little; the lever is cost. The runs used 244 tok/s
   against greedy's 352 because one problem at a time leaves slots idle while
   its slowest choice finishes, and 19% of greedy answers had no boxed answer
   (mostly repetition to the 2,048-token cap), part of what sampling fixes.
   Next on this track, in order: keep four sequences busy (two problems in
   flight), stop a problem's remaining choices once a majority is settled
   (adaptive consistency), then thinking-on runs, where single-sample accuracy
   and cost both rise; SMC-style particles only if pass@n pulls away from the
   vote on harder sets.
6. **Forced tool choice and structured output with speculation.** Grammar
   masks on the verify rows and on the drafter's candidates, and grammar-forced
   tokens committed without drafting (jump-forward). Order against item 3 by
   what the harness logs show: the Messages route counts rejected forced
   tool choices.

## Next work, in order

Reviewed 2026-09-14 against the whole archive; the order is by expected gain
per unit of work. Work the items one at a time, top to bottom. Each item names
its first step and its gate, so a session can pick it up cold. Batch-one decode
is at about 87% of the measured 239 GB/s weight-read ceiling, so the multipliers
left are accepted tokens per round, the eager accept path at batch two to four,
and hardware; kernel work at batch one is capped near 10%.

1. **Done, negative: a second proposal branch from prompt lookup.** Built as
   an opt-in (`--lookup-branch`, parked on branch `feat/lookup-branch`): a
   host-side n-gram continuation verified as a second 16-row block beside
   the DFlash2 chain, whichever block accepts more wins. The offline screen
   ([study](docs/archive/studies/ngram-branch-screen.md)) predicted +14% tokens
   per round on repository coding; served at batch one the branch gave
   +1.4% tokens per round and throughput within the 1% window noise
   (repository test set 45.4 to 45.6/45.3 TPS; ShareGPT +2.5/+1.5%; five
   real requests +2.8/+0.7%), with the mechanism audited correct
   ([study](docs/archive/studies/lookup-branch.md)). The branch wins inside
   copy spans where the chain is already near the 16-token cap and would
   have accepted the same tokens next round, so gained tokens save few
   rounds. A 31- or 63-row branch would lift the cap only for verbatim
   reproduction (at most 7 to 8% of rounds on these histories); not
   pursued. Lesson for every future screen: a per-round union oracle on
   recorded histories over-values tokens the chain already handles.
2. **Done: the family gate replaces the bit-exact gate and the three-prompt
   benchmark as the decision tool** ([study](docs/archive/studies/family-gate.md)).
   `tools/family-gate.py` runs paired 22-task passes in exclusive windows
   with the noise floor; `bench/family_inputs.py` regenerates the frozen
   inputs byte-identically; the code benchmark prompt no longer answers with
   a tool call at token 98 (`code` is the tool-free `gb10-coding-v2`, the old
   prompt is `code_v1`); the shape-stable kernels are in main behind
   `SHARAR_STABLE_NUMERICS=1`. Measured: two native passes agree to 0.1% with
   identical outputs; round cost resolves to 0.05%; accepted tokens per round
   have a ±7% interval on 18 families because every arithmetic change moves
   every continuation (about 100 families for ±3%, 1,000 for ±1%). Decision
   rule: judge cost on the round-cost interval, treat acceptance as unknown
   unless its interval excludes 1, never require identical outputs. Re-judged:
   `fp8_m16` is a clean −1.5% round-cost win (integrate next), the NVFP4
   draft body −2.9% against the FP8 draft with acceptance unresolved, stable
   numerics cost-neutral. Not re-run: the 16-row tree and the SiLU boundary
   need porting to current main first. Follow-ups: integrate `fp8_m16`;
   grow the family set to about 100 tasks with the generator in
   `bench/family_inputs.py` so acceptance claims of a few percent become
   testable. Done 2026-09-16: `family-eval-v3`, 101 families, ±3.0% measured
   ([study](docs/archive/studies/family-set-v3.md)).
3. **Half done: the accept path is device-resident, the server does not
   gain yet** ([study](docs/archive/studies/device-accept.md)). One captured
   graph per batch bucket decides acceptance (exact or relaxed, budget, stop
   tokens), reconstructs the state, commits the convolution windows and
   feeds the draft rings before the single host wait; ownership stays with
   StateManager and the proposer. Opt-in `SHARAR_DEVICE_ACCEPT=1`, exact
   (19 GPU tests, bitwise state while round boundaries agree). Engine-loop
   round cost x0.944 at batch four and 10K context, x0.883 at 1K, x0.983 at
   batch one; HTTP replay +2% at concurrency one, unchanged at concurrency
   four, where the engine thread's mean step is 117 ms on both paths
   because everything it does between rounds under the HTTP thread
   (publication, stats, reservations, plans, uploads) is now the exposed
   critical path. Next: complete rounds lazily (return after the replay,
   wait at the next step) and take an `nsys` timeline of the serving process
   at concurrency four; then the family gate before a default flip.
4. **Train on real traffic, then train while serving.** Record a few hundred
   requests of actual use and repeat the held-out and served measurements;
   the repository proxy overfits after one session. Then let the trainer run
   beside the server when the machine is idle: both fit in unified memory,
   so bound the interference instead, with anchor chunks of 32 and a check
   of the engine's active flag between chunks so a first token never waits
   more than a few hundred milliseconds. Reload is already in place.
5. **Spend memory on draft capacity.** The draft is 12 ms of a 107 ms round
   and the oracle ceiling on code is 7.85 tokens per round against 4.7. A
   draft twice as deep costs about 10 ms and breaks even at +0.45 tokens per
   round; tapping more than five target layers is the cheaper variant. Both
   change the rings, so they are restart changes distilled with `sharar
   train` from the pinned draft, judged as in item 1(c).
6. **Tensor parallel across two Sparks, if a second machine exists.** Halving
   weight bytes per GPU takes the round from 107 to roughly 55 to 60 ms; the
   128 all-reduces of 160 KB per round over the ConnectX-7 link cost about
   3 ms. It is the only path near 2x because the floor is bytes, not compute.
   Steps: NCCL over RoCE between the two containers, column/row split of the
   projections, replicated draft, a two-node launcher.
7. **Target bytes, only if the target is reopened.** Calibrated NVFP4 for the
   7.2 GB of FP8 projections removes about 12 ms per round. The W4A16 attempt
   failed because 5% NRMSE was the selection objective; re-quantize from the
   BF16 originals with Model Optimizer and judge on the family set. 2:4
   sparsity halves weight bytes at the kernel level but needs sparse
   fine-tuning of an already 4-bit model; a research project, not a task.
8. **Latency for one local user.** Prefill while the user types: the chat
   client sends prefill-only requests for the partial prompt and the prefix
   cache absorbs the rest, so the first token after submit is nearly
   immediate. Then the request cache policy for agent sessions from the
   [proposal](docs/request-cache-policy.md), and bounded admission with
   backpressure and live memory reporting (admission already holds under the
   host floor since 2026-09-18; queue-depth and per-session limits remain).
9. **Restore CI, cold-restore overlap, output controls, release work.**
   GitHub refused the previous run on account billing limits; keep the CPU,
   distribution and documentation checks. Event-owned restore lifetimes,
   request-owned constraint state before forced tool choice or structured
   output, prepared-weight caching and AOT kernels once profiles stabilize.

10. **Qwen3.8-Flash-Next (second target, `docs/flashnext.md`), next steps in
   order.** Served today through `config/flashnext.env` with MTP k=3: plain
   decode 24.9 tok/s at batch one (SGLang 16.7), 92 tok/s at batch four,
   MTP 1.99x on the JSON prompt, GSM8K-300 96.0% and HumanEval 94.5% at 1.8
   to 1.9x SGLang's wall time. What remains, each a fresh multi-hour build:
   1. Sparse-attention indexer for contexts past 2,048 tokens (dense is exact
      below): compressed-key cache per attention layer (mean over groups of
      four, Gemma norm, rope), top-512 block selection, expansion to token
      indices with the pending tail, a vendored sparse attention kernel; SGLang
      ships torch oracles for every step (`layers/attention/qsa/{mqa,kernel}.py`).
      Gate: reference parity on a prompt longer than the budget.
   2. Calibrated 3.25-bit trunk. Nearest-rounded copies give 31 ms per step
      (32 tok/s) but fail the gate (GSM8K 92.0, HumanEval 89.6) and are no
      faster at concurrency four, so `SHARAR_FLASHNEXT_TRUNK=w3` stays off;
      a GPTQ pass over the BF16 trunk following `docs/w3-calibration.md` is
      the fix. Gate: served GSM8K/HumanEval within noise of FP8.
   3. MTP draft-step fusion: 11.6 ms per round for three launch-bound steps
      (about 100 MB of weights each); fuse the head's small ops and its
      expert path the way the trunk's were. Gate: tok/s at batch one on the
      JSON and needle prompts.
   4. Family-gate inputs regenerated for the target so its changes are
      judged like the 27B's; per-row FP8 activation scales for the trunk
      (batch-one parity on the 512-token code prompt is 84% top-1 at FP8
      versus 91% at BF16).
   5. If a deeper pipeline is wanted: `sharar/kernels/wait_flag.py` is a
      graph-capturable device wait on a host flag, so the next decode graph
      can be launched before its PLE rows are gathered; the measured host gap
      after the parallel table reads is about 3 ms of the 40 ms step.

Done since the last order, kept for reference: promotion hardening (top-16
traces, KL distillation, attention subset, compiled trainer, continuation at
3e-5, 0.5% regression gate, optional on-policy selector), hot draft reload
(`POST /reload_draft`), and the target numerics study (shape sensitivity is
cuBLAS's BF16 b/a projection, cuBLASLt FP8 at 27 to 32 rows and the gated
norm's rows-per-block heuristic; M-invariant kernels opt-in in main behind
`SHARAR_STABLE_NUMERICS=1`, [study](docs/archive/studies/target-numerics.md)).

## Research directions (2026-09-15)

Ranked by expected multiplier at batch one, from the byte budget: one round
streams about 20 GB (9.6 GB NVFP4 MLPs, 7.2 GB FP8 projections, 0.7 GB head,
2.5 GB FP8 draft) in about 85 of 107 ms, at roughly 83% of the 239 GB/s
ceiling, and accepts 4.7 tokens per round against an oracle near 8. Only
fewer bytes, more accepted tokens, or a second GPU move the number.

- **Relaxed acceptance, opt-in per request: built and measured**
  ([study](docs/archive/studies/relaxed-acceptance.md)). `accept_top_k` and
  `accept_margin` accept a drafted token the target ranks within its top-k
  and within a logit margin of the argmax. Coding replays: top-2 within 1.0
  logit +9.8% at batch one and +5.7% at batch four with 4% of emitted tokens
  non-argmax; top-3 within 2.0 logits +21.6% at batch one with 7%; ShareGPT
  +14.8%. Round cost unchanged. Judged on 53 prompts by the target model in
  both orders: no setting distinguishable from exact (top-2/1.0: 13 relaxed
  wins, 14 exact, 26 ties; GSM8K 94.7% vs 95.0%, HumanEval 91.5% vs 90.9%,
  paired, not significant). Decision: merge as opt-in; top-2 within 1.0
  logit is the recommended interactive default, exact stays for benchmarks
  and training traces; a few hundred prompts per setting would be needed to
  see a 55/45 preference.
- **Three-bit decode weights beside NVFP4 prefill weights: built, tensor-core
  kernel at the memory ceiling for batch one to four** ([kernel study](docs/archive/studies/w3-cuda.md),
  [first study](docs/archive/studies/w3-decode.md)). `sharar/kernels/w3.py` +
  `csrc/w3_gemm.cu`: 3.25 bits per weight (3-bit codes into an eight-level
  E4M3 codebook, E4M3 scale per 32, FP32 row scale), bit planes stored in
  mma-fragment order so a lane expands four codes with three rotates, three
  masked ORs and one PRMT; `mma.sync` m16n8k32 E4M3 against the engine's
  static-scale FP8 projection inputs or per-row FP8 MLP inputs; split-K with
  an in-kernel reduction for the narrow shapes. 218 to 243 GB/s at 16 rows
  (ceiling about 239), 156 to 228 at 64. Opt-in `SHARAR_W3_DECODE=1` for calls
  of at most 64 rows; `SHARAR_W3_HEAD=1` and `SHARAR_W3_DRAFT=1` extend it to
  the lm_head and the draft. Round profile: batch one 104.7 to 68.8 ms
  (x0.66), batch four 139.1 to 106.3 ms (x0.76). 101-family gate with the
  layers, head and draft: round cost x0.649, tokens per round x0.965
  [0.937, 0.994], decode x1.488 [1.444, 1.532]; coding replays x1.58 batch
  one and x1.27 batch four exact (x1.63 and x1.40 with the relaxed
  default). Quality: round-to-nearest 3-bit is not free. Measured at
  concurrency four (the 2026-09-16 quality runs bypassed the then 16-row
  path and are void), the Lloyd-Max codebook with per-group FP8 activations
  is GSM8K 92.3% against 94.0% exact on the full test set (1,319, paired,
  p < 0.001) with HumanEval within noise for the layers; the lm_head copy
  costs HumanEval and stays opt-in; the uniform grid loses five points on
  both; per-row activation scales lose more. The launcher default flipped
  to layers and draft on (a05457d). GPTQ calibration from the FP8 release
  now passes paired accuracy and the full family gate; see the
  [new study](docs/archive/studies/w3-calibration-adaptation.md) and
  [artifact guide](docs/w3-calibration.md). Drafter adaptation on fresh
  calibrated-target traces also passes held-out serving checks. The remaining kernel opportunity
  is 64-row occupancy (attention qkv at 156 GB/s).

## Engine to-dos

- **Relaxed acceptance for sampled requests.** Today `accept_top_k` /
  `accept_margin` require temperature 0 (HTTP 400 otherwise) because sampled
  verification is rejection sampling, which keeps the exact target
  distribution and has no argmax to relax. The sampled analogue is typical
  acceptance (accept a drafted token whose target probability clears an
  entropy-scaled threshold, as Medusa and EAGLE do), or a biased ratio
  `min(1, p_target / (q_draft * factor))`. Add typical acceptance behind the
  same two parameters when a sampling user wants the speed; judge it the
  same way as the greedy mode.

## Maintenance fixes

Small, independent, any session; none changes numerics.

Completed 2026-09-17: trace storage counters are maintained by the writer,
rotation includes feature bytes, and TTFT percentiles are computed on the
HTTP side from an immutable snapshot. W3 dequantization also imports its
helpers from the installed runtime instead of the benchmark package. See the
[runtime and calibration study](docs/archive/studies/w3-calibration-adaptation.md).

Completed 2026-09-18: the detokenizer decodes a bounded window (the emitted
tokens since the last stable boundary plus the new ones) instead of the whole
output; `run-sharar.sh` forwards `SHARAR_COLD_TIER_GIB`; admission holds while
host `MemAvailable` is under the 16 GiB floor (`floor_holds` in the scheduler
stats) instead of leaving that to the external guard. The page-16 measurement
promised in `kv_pool.py` was made with `bench/kv_page_size.py` (FlashInfer
decode and 16-row verify attention over FP8 pages, 10K context, 16 layers,
random page order): page 16 leaves decode attention unchanged (x0.99) and
takes verify attention from 2.52 to 2.08 ms per round at batch one and 8.49
to 7.42 ms at batch four, under 1% of the round either way; plan() cost is
sub-millisecond at both sizes. Not worth the token-granular cache, cold-tier
and reservation refactor; parked. `/generate` streaming still re-sends the
cumulative output ID list per chunk because SGLang clients expect it.

- `can_admit` and `admit` each run the cold-tier scan, eviction and radix
  match; the cold-tier `find` materializes the prefix per parked entry.
- `gemma_rmsnorm_nvfp4` (fused norm and NVFP4 quantizer, bit-identical per
  its tests) is dead behind `SHARAR_FUSED_NORM_NVFP4`; keep or remove.
- The MTP head runs entirely in BF16 with unfused gate and SiLU; irrelevant
  while DFlash2 is the default, remove or port if MTP is kept.

## Contribution gates

- Preserve [request ownership](docs/phase7a.md), private KV tails, recurrent
  commits, graph-buffer lifetimes and stop-truncated draft feedback.
- Identify source, image, precision, input/output work and memory geometry for
  every performance claim. Retain failed runs and exact comparison limits.
- Run CPU contracts for control changes, the affected GPU tests for execution
  changes, and paired full-workload runs for performance changes; an engine
  change that alters arithmetic passes the family gate (`tools/family-gate.py`,
  18-family paired intervals plus the noise floor) before merging, and the
  standard benchmark is reported, not used to decide.
- A larger redesign should establish at least 5% aggregate improvement with
  protected BS1 and latency, and target 10% or more to justify complexity.
- Drafter changes are judged on held-out tokens per round and served
  throughput on the user's own newest traffic, never on a public corpus.
- Keep raw dumps outside Git and compact evidence manifests linked from dated
  reports. Keep licensing, attribution and numerical reference paths intact.
- Coherent commits with validation evidence and no attribution trailers.
  Validation windows restore the original service and memory guard.

Do not repeat unchanged rejected selectors, numerical kernel replacements,
mixed-forward policies or scheduler experiments. Consult the
[decision index](docs/archive/README.md) before starting another screen.
