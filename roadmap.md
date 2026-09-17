# Sharar roadmap

Updated 2026-09-14. This is the current work order. Historical milestones and
their original gates are in the [archived roadmap](docs/archive/roadmap-2026-09-11.md).

## Supported baseline

One GB10, the pinned Qwen3.8-27B NVFP4 target, 32K context, maximum batch four.
DFlash2 with 15 proposals and accepted-state reconstruction; BF16 draft by
default, FP8 as the measured serving profile; relaxed acceptance at top-2 within
1.0 logit for greedy requests that do not choose; 3-bit decode copies of the
layers and the draft on by default in the launcher (the lm_head copy opt-in),
accepted at about two GSM8K points until calibration lands. The target, its vocabulary head
and the verifier are fixed. The DFlash2 drafter is trainable on recorded
traffic and serving follows the promoted checkpoint; the publisher's draft is
the fallback. See [operations](docs/operations.md), [performance](docs/performance.md),
the [optimization inventory](docs/optimizations.md) and [draft adaptation](docs/draft-adaptation.md).

## Current state

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
3. **Make the accept path device-resident for batch two to four.** After the
   single `.tolist()` in `SpecDecoder._accept` the round is eager: the
   reconstruction launch, per-request conv copies, feature feedback, trace
   top-k, two FlashInfer plans and the stop scan. At batch four short context
   that is about 16 ms of a 136 ms round; at batch one about 1%. Steps:
   device-side acceptance and stop detection with a `max_new_tokens` clamp;
   one captured graph per batch size for accept, reconstruction and feedback;
   tokens read back through an event and a pinned buffer off the critical
   path; keep the plan inputs exact. Gate: batch-four HTTP throughput and
   unchanged output IDs at fixed shape.
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
   backpressure and live memory reporting; `Ledger.check_floor()` is not yet
   in the serving loop.
9. **Restore CI, cold-restore overlap, output controls, release work.**
   GitHub refused the previous run on account billing limits; keep the CPU,
   distribution and documentation checks. Event-owned restore lifetimes,
   request-owned constraint state before forced tool choice or structured
   output, prepared-weight caching and AOT kernels once profiles stabilize.

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
  is GSM8K 93.0% and HumanEval 90.2% against 95.0% and 90.9%; the uniform
  grid loses five points on both; per-row activation scales lose more. A
  full-size GSM8K decides whether the launcher default flips. Next: GPTQ
  or QuIP#-style calibration from the FP8 release in the local cache (the
  path to a lossless default); retrain the drafter on 3-bit traces; the
  64-row occupancy (attention qkv at 156 GB/s).

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

- `TraceSink.stats()` globs the trace directory and stats every trace file and
  every feature blob on every engine-loop iteration through
  `EngineLoop._run_inference` and `Scheduler.stats()`, idle or not. Keep
  counters in the sink, updated by the writer thread.
- `Metrics.summary()` sorts the 2,048-entry TTFT window on every iteration for
  the same reason; compute it on `/metrics` requests only.
- The detokenizer re-decodes the whole output every round, and `/generate`
  streaming re-sends the full output ID list in every chunk; both are
  quadratic in output length.
- `SHARAR_COLD_TIER_GIB` is read inside the container but `run-sharar.sh`
  never forwards it.
- `can_admit` and `admit` each run the cold-tier scan, eviction and radix
  match; the cold-tier `find` materializes the prefix per parked entry.
- KV page size is fixed at 1, so the per-round FlashInfer plan concatenates
  an index list as long as the context; the page-16 measurement promised in
  `kv_pool.py` never happened.
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
