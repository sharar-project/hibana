# Sharar roadmap

The work order. Every item is a GitHub issue on `abdelfattah-lab/sharar` with its
context, first step and gate; this page only orders them. History is in git and in
the tag `archive/pre-cleanup-2026-09-26` (the old roadmap and every dated study).
Updated 2026-09-27.

## Where we are

One GB10, two targets, one per process, no default between them.

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| Serving profile | calibrated 3-bit decode weights, FP8 DFlash2 k=15 (traffic-adapted drafter), relaxed acceptance | FP8 trunk, NVFP4 experts, MTP k=4 with the 3-bit draft head, sparse-attention graphs |
| Single-stream decode (h2h, greedy / t=0.7 median) | 93.7 / 76.7 tok/s | about 56 / 52 tok/s |
| Against SGLang's public setups on this box | 1.8-2.1x | 1.3x greedy, 1.3x sampled |
| Context | 262K | 262K (sparse attention past 2,048) |

Measurements and their evidence: [optimizations](docs/optimizations.md),
[performance](docs/performance.md), [Flash-Next](docs/flashnext.md), the h2h
benchmark (#26). Architecture and the capability matrix: [architecture](docs/architecture.md).

## Order

1. **Cleanup** (#23): lighter tree (done), the HTTP evals in `sharar-evals` (#53, done); test suite audit (#54).
2. **Architecture refactors**, one at a time, each bit-identical on both targets
   with round cost and host idle not higher (#24):
   done 2026-09-26/27: #44 EngineConfig, #45 speculator parts, #46 attention planner and
   graph sets, #47 model contract, #48 one Linear, #52 serving profiles (fast is the default).
   Left, each paired with the optimization it unblocks: #49 feature taps (-> #7),
   #50 one MTP body (-> 27B MTP), #51 Trainable interface (-> #40).
3. **Capability gaps and speed**, on the refactored seams:
   - #43 adaptive draft depth, both targets.
   - #36 Flash-Next round idle gaps (PLE reads, plans before the draft sync, device acceptance for MTP, #18).
   - done 2026-09-27: #6 calibrated 3-bit Flash-Next trunk (+24-26% single-stream, in the fast profile); #61 recovers its HumanEval cost. #37 small-M MoE kernel is parked on `feat/moe-small-m` until verifies of 1-2 rows are common.
   - #59 throughput profile (batch 8-16) for faster evals and many agents.
   - #40 MTP trained on traffic; #33 self-play traces; #12 27B drafter on real traffic.
   - #7 DFlash2 for Flash-Next (after #49; screen the public PixelML drafter first).
   - #29 draft vocabulary ranked from traffic.
4. **Research** (measure before building): #38 fewer active experts, #39 expert pruning,
   #34 on-policy distillation (separate repository when it starts), #41 expert-budgeted
   verification, #42 sub-3-bit 27B MLPs, #25 megakernels.
5. **Harness and quality**: #11 Flash-Next family gate, #28 harder evals, #31 lossy-change
   gate, #16 constrained output with speculation, #19 `/v1/responses`, #15 test-time scaling,
   #13 session store snapshots, #14 long-context prefill.
6. **Infrastructure**: #21 CI. Hardware-gated: #20 and #30 (a second Spark).

## Rules

The contribution gates are in [CONTRIBUTING](CONTRIBUTING.md). In short: a change that
alters arithmetic passes the family gate against its noise floor; lossy options are
judged by the lossy-change gate and land in a named profile, not the exact one; drafter
changes are judged on the user's own newest traffic; every performance claim names its
source, image, precision and geometry and keeps its evidence outside git.
