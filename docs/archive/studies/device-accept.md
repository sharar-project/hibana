# Device-resident acceptance: the round is faster, the server is not yet

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities.

2026-09-18, main `268fee2` plus the working tree of this change. Evidence
root `${SHARAR_RESULTS}/reviews/2026-09-18-device-accept/`, compact
[manifest](device-accept-results.json). Roadmap item 3 asked for the accept
path to be device-resident for batch two to four; this builds it, measures
it in the engine loop and in HTTP serving, and records why the two disagree.

## What was built

`sharar/engine/accept.py` decides acceptance as static-shaped tensor ops
(exact or relaxed match, cumulative prefix, budget clamp, first stop token;
`tests/unit/test_device_accept.py` checks it against the host rule over 300
random blocks). `sharar/engine/accept_graphs.py` captures one graph per batch
bucket that runs that decision on the verify graph's static outputs, the
accepted-state reconstruction and convolution commit for exactly those
counts, and the proposer's new `feedback_device` (one projection over the
bucket's rows, accepted rows scattered into the private rings, the rest into
the dummy ring). The host reads one packed result through a pinned buffer
and an event and only publishes. `StateManager.authorize_reconstruction` and
`publish_reconstructed` split the host commit so ownership rules are
unchanged ([interfaces](../../interfaces.md#device-resident-acceptance)).
`SHARAR_DEVICE_ACCEPT=1` enables it; the graphs are always captured so
benchmarks can switch at run time.

## Correctness

`tests/gpu/test_device_accept.py` (19 tests, production W3 profile, promoted
drafter): outputs identical to the host path for batch one to four, exact
and relaxed, 2 and 12 rounds; stop tokens and budgets cut identically;
pools clean afterwards; the device feedback bit-identical to a host
projection of the same rows. Round-by-round comparison from identical starts
(`diag.log`): while the two paths emit the same per-round counts, the
committed recurrent and convolution state is bitwise identical. The device
path projects every verify row of the bucket in one GEMM while the host path
projects exactly the accepted rows, and cuBLAS rounds bf16 differently at
one row, so after a round with count one the drafter's rings differ at bf16
precision and its proposals can move a round boundary; from then on the
target's own shape sensitivity ([performance](../../performance.md#limits))
separates the continuations, as it does for any arithmetic change.

## Round cost in the engine loop

`bench/gb10_round_profile.py`, 60-round windows after 8 warm rounds, the
frozen family prompts, exact acceptance, W3 layers and draft with the
calibrated target, the promoted FP8 drafter (`profile-*.json`):

| Context | Batch | Host path ms | Device path ms | Ratio |
|---:|---:|---:|---:|---:|
| 10,000 | 1 | 71.15 | 69.92 | 0.983 |
| 10,000 | 4 | 110.51 | 104.34 | 0.944 |
| 1,000 | 1 | 69.28 | 68.05 | 0.982 |
| 1,000 | 4 | 110.20 | 97.32 | 0.883 |

The batch-four saving (6.2 ms at 10K, 12.9 ms at 1K) is the eager section
the item named: reconstruction launch, convolution copies, feedback,
metadata uploads and the stop scan, now inside one graph replay before the
single wait.

## HTTP serving

`bench/draft_ab.py` on the 40-request, source-file-disjoint test corpus of
the [calibration study](w3-calibration-adaptation.md), exact acceptance,
server from the runtime image with the working tree mounted, arms
interleaved host, device, host, device (`ab-*.json`):

| Concurrency | Host aggregate TPS | Device aggregate TPS | Rounds | Accepted |
|---:|---|---|---|---|
| 1 | 68.3, 68.8 | 69.7, 69.7 | 2,793 vs 2,781 | 11,823 vs 11,826 |
| 4 | 153.6 (first, cold arm), 162.0, 162.6 | 162.2, 162.1, 163.5 | 730 to 749 vs 739 | 11,746 to 11,822 vs 11,785 to 11,807 |

Batch one gains about 2% and acceptance is unchanged. At concurrency four
the warm arms agree to 0.3%. The scheduler's own counters explain it
(`timing-*.json`, one warm replay per arm): the engine thread is busy for the
whole wall time in both arms (91.1 of 89.1 s and 90.7 of 88.5 s, including
42 prefill chunks), and its mean step takes 117.3 ms on the host path and
118.2 ms on the device path over 777 and 767 steps. So in serving the
round's host critical path is not the accept section: it is everything the
engine thread does between rounds under the HTTP thread (token publication,
per-step stats, reservations, two FlashInfer plans, uploads), which the host
path overlapped with the reconstruction and feedback kernels and the device
path now exposes. The engine-loop gain and the serving result are both real;
they measure different critical paths.

## Decision

Keep the path opt-in (`SHARAR_DEVICE_ACCEPT=1`, launcher passthrough), exact
and tested; do not flip the default on a 2% batch-one gain without the
family gate. The serving gain needs the second half of the item: complete a
round lazily (return after the replay, wait at the start of the next step so
publication and stats overlap the GPU), and measure the serving process's
host timeline under load (an `nsys` capture of the server at concurrency
four with the HTTP thread active) before touching the scheduler. The
per-step host cost under load, not the accept section, is now the number to
reduce at batch four.
