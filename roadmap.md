# Sharar roadmap

The work order. Every item is a GitHub issue on `abdelfattah-lab/sharar` with its
context, first step and gate; this page only orders them. History is in git and in
the tag `archive/pre-cleanup-2026-09-26` (the old roadmap and every dated study).
Updated 2026-10-01.

## Where we are

One GB10, two targets, one per process, no default between them.

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---|---|
| Serving profile (`fast`) | calibrated 3-bit decode weights, FP8 DFlash2 k=15 (traffic-adapted drafter), relaxed acceptance | FP8 trunk with calibrated 3-bit decode copies, NVFP4 experts, MTP k=4 with the 3-bit draft head, device-resident acceptance, sparse-attention graphs |
| Single-stream decode (h2h, greedy / t=0.7 median) | 92.5 / 74.5 tok/s | 72.9 / 67.9 tok/s before #68; greedy x1.16 with it (about 84) |
| Against SGLang's public setups on this box | 2.1x greedy, 1.7x sampled | 1.7x greedy, 1.7x sampled (before #68) |
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
   - done 2026-09-27/29: #6 calibrated 3-bit Flash-Next trunk (+24-26% single-stream, in `fast`); #61 its
     HumanEval cost is within the FP8 trunk's own spread, the checkpoint stays; #36 Flash-Next round idle gaps
     and device-resident acceptance for MTP (+16% single-stream in `fast`); #59 `throughput` profile (batch 16,
     GSM8K x1.84 Flash-Next, x1.74 27B). #37 small-M MoE kernel is parked on `feat/moe-small-m` until verifies
     of 1-2 rows are common.
   - Screened 2026-09-29, no-go: #7 DFlash2 for Flash-Next. The verify slope is the routed experts (16 rows 68 ms
     against ~42 ms for MTP's 5, with the 3-bit trunk); a drafter needs +18% tokens per round over MTP at block 7,
     +53% at block 15, and the public PixelML drafter accepts less than MTP ([numbers](https://github.com/abdelfattah-lab/sharar/issues/7)).
     Reopen when wide verification is cheaper or a drafter clears that bar on our traffic.
   - Built and parked 2026-10-01: #73, 3-bit routed experts as the only resident expert copy (`park/experts-3bit`).
     x1.19 single-stream over `fast` and -17.2 GiB, but with the cheap calibration it loses knowledge: lossy gate
     rejected (-0.8 pp, p 0.044), MMLU-Pro -2.1 pp against `fast` (law, business, engineering), more over-long
     deliberation on GPQA ([numbers](https://github.com/abdelfattah-lab/sharar/issues/73)). Reopen with GPTQ on routed
     activations, judged on MMLU-Pro and GPQA (sharar-evals `evals.hard_eval`) before the full gate.
   - #72: the `fast` profile fetches the calibrated 3-bit checkpoint on first serve (#75). Flash-Next's is public
     (`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3`, Qwen Community License); the 27B needs recalibrating and gating
     before it has one.
   - Next screens: #33 self-play traces for the drafter work; the #55 oracle (re-capture routes from
     `park/router-traces`, then simulate how much expert loading an ideal scheduler saves at depth 4 and 8 and how
     much a confidence rule keeps; rejected rows first-use 16% of expert bytes at k=4, 44% at k=8). If the oracle shows
     a big gap: an online scheduler that defers unlikely rows' experts (#41, in a persistent kernel, #25).
   - Drafter learning: #40 MTP trained on traffic (after #51), #12 27B drafter on real traffic, #29 draft
     vocabulary ranked from traffic.
   - Later: #43 adaptive draft depth (the draft confidence it needs is recorded with `SHARAR_TRACE_CONFIDENCE`).
4. **Research** (measure before building): #34 on-policy distillation (separate repository when it starts), #41 expert-budgeted
   verification, #42 sub-3-bit 27B MLPs, #25 megakernels. Deprioritized by the #32 router data:
   #38 fewer active experts (flat router mass), #39 expert pruning (domains share few experts).
5. **Harness and quality**: #11 Flash-Next family gate, #28 harder evals (sharar-evals `evals.hard_eval`: MMLU-Pro, GPQA, IFEval, WMT24++, 2026-10-01; Terminal-Bench 2.1 next) (#31 lossy-change gate
   done 2026-09-29: `tools/lossy-gate.py`), #16 constrained output with speculation, #19 `/v1/responses`, #15 test-time scaling,
   #13 session store snapshots, #14 long-context prefill.
6. **Infrastructure**: #21 CI (GitHub Actions jobs do not start: the organization's billing; a self-hosted
   runner on the GB10 would also run the GPU suite). Hardware-gated: #20 and #30 (a second Spark).

## Rules

The contribution gates are in [CONTRIBUTING](CONTRIBUTING.md). In short: a change that
alters arithmetic passes the family gate against its noise floor; lossy options are
judged by the lossy-change gate and land in a named profile, not the exact one; drafter
changes are judged on the user's own newest traffic; every performance claim names its
source, image, precision and geometry and keeps its evidence outside git.
