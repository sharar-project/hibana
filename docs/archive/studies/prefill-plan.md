# One GDN prefill plan per forward

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The GB10 prefill path now prepares sequence metadata once and shares it across
all 48 GDN layers. `model/gdn_plan.py` owns the plan; `ForwardContext` owns its
lifetime. CPU lengths produce sequence starts, initial-state flags, chunk
indices/offsets and checkpoint locations without reading CUDA tensors.

The plan is independent of `ForwardContext.batched`. Cold single-sequence
attention keeps its BF16 ragged path; cached and batched attention keep their
existing paged path. Recurrent arithmetic, FP32 committed state, checkpoint
boundaries and decode/verification graphs are unchanged. Create a new context
when its request metadata changes. Standalone chunk kernels retain optional
metadata construction for callers without a prepared plan.

## Measured ordinary prefill

Measured on the same GB10 on 2026-09-05, target
`RadixArk/Qwen3.8-27B-NVFP4@319f741cce68d7914884900c138a1fbb70a42f30`,
DFlash2 k=7, FP32 state reconstruction, 35,000 target pages, 12 state slots,
maximum four active requests, mixed execution off. The control loads the
unmodified GDN core and chunk helpers from `d319896`. Both implementations
share the other runtime code, weights, allocations and HTTP server.

One warmup ABBA block and three measured ABBA blocks give six measured samples
per implementation per cell. These are small, workload-specific measurements.
All measured output token IDs match between control and candidate.

| Natural request | Control TTFT | Planned TTFT | TTFT reduction | Completion reduction |
|---|---:|---:|---:|---:|
| Chat, cold | 144.7 ms | 103.3 ms | 28.6% | 0.9% |
| Coding, cold | 155.3 ms | 118.8 ms | 23.5% | 4.3% |
| Session, cold (2,172 input tokens) | 1,016.3 ms | 833.6 ms | 18.0% | 5.3% |
| Session, cached prefix | 150.0 ms | 118.5 ms | 21.0% | 2.9% |

A separate warmed 129-token prefill trace reduces `cudaStreamSynchronize`
calls from **290 to 7**. GPU activity union remains approximately 117 ms;
profiled idle fraction falls from 43.6% to 7.1%. The trace span falls from
206.7 to 126.7 ms. Instrumented spans are diagnostic, not client latency, and
neither allocator sizes nor kernel timings measure physical DRAM traffic.

An independent model load and ABBA window reproduced the improvement later
the same day. It again passed all 288 full-model layer comparisons, and every
measured output matched both the control and the initial run. The four cells
again have six samples per implementation:

| Request | Control TTFT | Planned TTFT | TTFT reduction | Completion reduction |
|---|---:|---:|---:|---:|
| chat-cold | 141.0 ms | 104.3 ms | 26.0% | 0.7% |
| coding-cold | 163.6 ms | 117.7 ms | 28.0% | 6.4% |
| session-cold | 1045.5 ms | 837.4 ms | 19.9% | 7.1% |
| session-session | 168.9 ms | 120.3 ms | 28.8% | 2.9% |

The second trace again records 290 versus seven stream synchronizations.
Host `MemAvailable` stayed above 81.53 GiB during this confirmation window.

## Correctness evidence

- Ten GPU chunk cases compare prepared metadata against the standalone path,
  including lengths 1, 3, 16, 17, 63, 64, 65, 129 and uneven batches. Outputs,
  entering-chunk states and committed FP32 state are bit-exact; unowned state
  remains untouched. Tests forbid fallback metadata construction in every
  helper when a prepared plan is supplied.
- Six full-model cold/continued/batched cases compare every GDN core and owned
  convolution/recurrent/checkpoint state with the frozen old implementation:
  all 288 layer comparisons are exact, as are first output token IDs.
- CPU checks cover metadata without tensor reads, slot order, uneven lengths,
  checkpoint alignment, private scratch slot zero and preserved attention mode.

The full GPU/runtime suite passes **675 tests with one expected skip**. After
archiving the public mixed-serving option and completing the benchmark client,
the final CPU suite passes **315 tests with six GPU-only skips**. The candidate
runtime passes all **16 historical parity comparisons** with the existing
criteria. The first parity launcher was a no-op; its empty log is explicitly
excluded and the corrected CLI run requires a 16-result summary.

Both mixed modes complete the full 78-request local-v1 protection suite without
request or automatic-check failures. Manual review gives 22 passes and four
existing sampled-chat rubric misses per mode. The earlier additional mixed-chat
misses do not recur in this run. Arrival TTFT nevertheless regresses **16.9–24.4%**
in every selected mixed case, against a 5% limit. The public serving option is
removed. The remaining mixed runtime hooks were subsequently archived at
commit 56605f0; current serving retains the ordinary prefill improvement. See [the archive decision](mixed-forward.md).

The [evidence manifest](prefill-plan-results.json) records the full external bundle
location and SHA-256; detailed samples, sources and outputs stay outside Git.
Raw traces and attempts are under
`$SHARAR_RESULTS/reviews/2026-09-05/prefill-plan/`; the corrected parity run is
also under the adjacent `competitive/` directory. The lowest sampled available
host memory during validation is 67.42 GiB. Runtime dependencies are unchanged;
the candidate image copies the source onto the validated mixed-forward base.

## Reproduction

Export `sharar/model/gdn.py` and `sharar/kernels/fla/{chunk,chunk_o,chunk_delta_h}.py`
from `d319896` to a directory as `legacy_gdn.py`, `legacy_chunk.py`,
`legacy_chunk_o.py` and `legacy_chunk_delta_h.py`. In the pinned GB10 runtime,
with exclusive model memory and the host guard active:

```bash
python3 bench/prefill_plan.py --reference /path/to/control --output /path/to/new-run
python3 -m pytest tests/unit tests/gpu -q -p no:cacheprovider
```

For the archived mixed gate, check out commit `56605f0` and run
`python3 bench/mixed_forward.py --output /path/to/mixed-gate --protected`.

The benchmark writes every sample, frozen prompt IDs, reference file hashes,
per-layer state hashes and separate profiler traces. The host harness records
memory/power conditions and restores the original serving container afterward.
Initial benchmark-client failures (an import path and EOS text handling) are
retained in the raw evidence; they did not change model arithmetic or gates.
