# Fresh GB10 comparison: Sharar and SGLang

This experiment compares plain decoding and DFlash2 on one GB10 using the
same pinned target, draft, token inputs, requested sampling parameters and
total memory budget.
It measures the current Sharar source after shared GDN prefill planning.
[The older MTP comparison](comparison.md) remains historical evidence.


Sharar is competitive, but this pilot does not establish a broad substantial
lead. Across **345 measured requests in five profiles**, every request completes
with valid token accounting and every profile stays within the common budget.
Against the faster measured SGLang DFlash2 setting per cell, Sharar's fixed-work
median throughput is **7–17% higher in the four 2K/8K cells**, **29% lower at
128 tokens / concurrency four**, and **3.2× at 128 tokens / concurrency one**.
The latter comes mostly from different continuations and verification counts;
see the interpretation below. Batched measurements vary widely.

The coding and tool answers are token-identical across all three DFlash2
profiles and pass their automatic checks. Sharar completes them **6.5% and
7.8% sooner**, respectively, than the faster SGLang setting. Chat, reasoning
and the cached session complete 10.7%, 6.7% and 2.3% sooner with different
answers; the cold session takes 3.3% longer. JSON remains invalid on SGLang
because it emits Markdown fences against the frozen bare-JSON requirement.
Sampled chat misses the frozen computation rubric on every profile.

Plain decoding is 1–3% faster on SGLang in five cells. Sharar's 8K / four-request
throughput is 51.5% higher than the measured SGLang plain configuration, while
its median TTFT is worse. SGLang plain uses 2K prefill chunks here; plain 8K was
not measured. This isolated result does not establish a tuned plain-decode lead.

## Protocol

- Target: `RadixArk/Qwen3.8-27B-NVFP4` at
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Draft: `incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, block size eight (seven proposals),
  with the checkpoint's 2,048-token sliding attention window.
- Both use FP8 target KV, FP32 recurrent state, 35,000 target KV tokens,
  32,768 context capacity and at most four active requests. Sharar uses accepted
  state reconstruction and keeps mixed execution off.
- Internal state-slot counts differ: Sharar uses 12; SGLang uses 24. SGLang
  requires five resident slots per active request in this configuration. The
  excluded 12-slot smoke test was capped to two active requests. The final
  launcher requires SGLang's runtime log to confirm capacity four. Equal
  internal slot counts would have imposed unequal serving capacity.
- Each engine runs alone, with a CPU quota of eight cores and four OpenMP threads.
  Client and server are separate processes for both engines, on container
  loopback. Model loading, compilation and warmups are outside request timings.
- Each profile has the same 64 GiB cgroup cap and host `MemAvailable`-drop budget
  relative to its prelaunch baseline, plus a 16 GiB absolute host floor. The
  host guard samples every 250 ms. This is a common enforced budget, not a claim
  of identical allocation or precise process-attributed physical usage. Host
  deltas include cache reclamation and other processes; cgroup accounting alone
  does not contain CUDA allocations on this machine. Do not add overlapping
  CUDA and RSS figures.

The fixed-work matrix has 128/2,048/8,192 input tokens, concurrency one/four,
256 output tokens per request, one warmup and three measured batches per cell.
Every batch starts after a cache flush; request prefixes differ. `ignore_eos`
is true. These repetitive, forced-output requests diagnose throughput; their
acceptance rates and speedups do not establish useful-answer speed.

The natural suite keeps the frozen `local-v1` prompts, stopping rules and
sampling for chat, coding, tools, JSON, reasoning, sampled chat and a session.
The session is measured both cold and after priming its prefix. Each cell has
one warmup and three measured requests. All output IDs, token counts, cache
hits, errors and automatic/manual quality checks are retained.

Aggregate throughput is total emitted output tokens divided by client batch
makespan. TTFT is the first received token-bearing SSE event. Decode throughput
excludes the first event's token burst; no timestamps are invented for tokens
within a speculative burst. Natural completion times include differences in
answer length; fixed-work and exact-output ablations answer different questions.
No tail-latency or quality-equivalence claim follows from three repetitions.
The default SGLang FlashInfer sampler ignores per-request seeds unless its
separate deterministic sampling mode is enabled; Sharar uses the requested
seed. SGLang plain produced different sampled answers across repetitions.
The source branch and raw outputs are archived. Treat the sampled-chat cell
as qualitative evidence, not a matched-seed speed or distribution comparison;
no deterministic-mode slowdown is imposed on SGLang for the throughput matrix.
Profiles run sequentially; independent repeated comparison windows remain a
promotion requirement. Some batched greedy outputs and acceptance counts also
vary between repetitions/settings: these are fixed token budgets, not fixed
continuations or identical amounts of verification work. Such variation can
change the last request and batch makespan substantially. The artifact retains
output hashes, per-request completion and native acceptance counters; the next
controlled cost study must hold target inputs and accepted lengths fixed.

SGLang is image `lmsysorg/sglang:dev-qwen38-27b-dflash2` at
`sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`,
source `5f55db35e926d50676f75b812640ea2410b0fe0e`. The benchmark uses its native
DFlash2 selector, FlashInfer attention, CUDA graphs and compiled speculative
path. The plain control uses 2,048-token prefill chunks. For DFlash2, both
2,048- and 8,192-token prefill settings are measured and reported;
a per-cell best of these settings must be explicitly labeled. The image runs
with `--json-model-override-args '{"language_model_only":true}'`, which skips
the unused vision tower through the Qwen model config, and without engine source
patches. This build restricts the similarly named CLI switch to another model;
that rejected launch produced no timings. `--language-only` alone controls encoder routing; the
partial run that still loaded vision weights is retained and excluded. Exact local snapshot paths and
`SGLANG_HOST_IP=127.0.0.1` allow offline loopback startup.

Sharar's source overlay image is
`sha256:412c4c2e9a4e1179cad974d72b5f3a1a943c33075a0b39801f0dd774af66751b`.
It reuses validated runtime dependencies; this is not a clean dependency
rebuild. Serving source hashes, external frozen benchmark inputs, full launch
commands, source manifests, request samples and memory/power records identify
what was actually tested. The original production image is restored and
checked after each guarded window.

## Fixed-work results

Median aggregate output tokens/s, including prefill and client delivery; higher is faster. The ratio compares Sharar DFlash2 with the faster SGLang DFlash2 setting in each cell. Both SGLang settings are retained below.

| Input / concurrency | Sharar plain | SGLang plain | Sharar DFlash2 | SGLang DFlash2 / 2K | SGLang DFlash2 / 8K | Sharar / best SGLang DFlash2 |
|---|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 12.07 | 12.39 | 64.30 | 19.60 | 20.06 | 3.205× |
| 128 / 4 | 44.00 | 45.37 | 95.65 | 135.56 | 56.91 | 0.706× |
| 2,048 / 1 | 11.65 | 11.92 | 56.94 | 52.08 | 53.05 | 1.073× |
| 2,048 / 4 | 38.78 | 39.93 | 112.23 | 59.63 | 98.45 | 1.140× |
| 8,192 / 1 | 10.43 | 10.57 | 29.96 | 23.27 | 25.65 | 1.168× |
| 8,192 / 4 | 27.23 | 17.97 | 48.03 | 41.42 | 38.17 | 1.159× |

Observed aggregate-throughput ranges across the three measured DFlash2 batches (not confidence intervals):

| Input / concurrency | Sharar DFlash2 | SGLang DFlash2 / 2K | SGLang DFlash2 / 8K |
|---|---:|---:|---:|
| 128 / 1 | 64.16–64.50 | 19.54–19.73 | 19.99–20.10 |
| 128 / 4 | 94.14–215.43 | 135.24–143.02 | 55.07–57.11 |
| 2,048 / 1 | 55.65–57.00 | 51.99–52.30 | 53.00–53.12 |
| 2,048 / 4 | 56.55–132.26 | 56.47–73.05 | 78.19–98.78 |
| 8,192 / 1 | 29.76–29.98 | 23.13–23.31 | 25.59–25.68 |
| 8,192 / 4 | 38.13–49.22 | 40.22–42.57 | 35.39–40.83 |

Median client TTFT in milliseconds; lower is faster. At concurrency four this is the median across the twelve measured requests.

| Input / concurrency | Sharar plain | SGLang plain | Sharar DFlash2 | SGLang DFlash2 / 2K | SGLang DFlash2 / 8K |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 122.1 | 127.1 | 125.0 | 164.3 | 159.1 |
| 128 / 4 | 223.5 | 310.0 | 237.0 | 338.5 | 337.3 |
| 2,048 / 1 | 706.3 | 796.4 | 719.1 | 821.3 | 822.3 |
| 2,048 / 4 | 2294.1 | 2321.2 | 2336.5 | 2715.2 | 3564.2 |
| 8,192 / 1 | 3010.4 | 3333.8 | 3061.6 | 3356.2 | 4010.7 |
| 8,192 / 4 | 10032.2 | 8467.4 | 10219.3 | 8684.6 | 13753.7 |

## Naturally stopping requests

Each cell shows **output tokens / TTFT ms / completion seconds**, using the median of three requests. Token ranges appear when the answers differ. Read these with the quality review: answer lengths and generated text can differ across engines.

| Request | Sharar DFlash2 | SGLang DFlash2 / 2K | SGLang DFlash2 / 8K |
|---|---:|---:|---:|
| chat-cold | 70 / 108.1 / 3.684 | 79 / 136.5 / 4.125 | 79 / 134.4 / 4.140 |
| coding-cold | 35 / 119.4 / 0.677 | 35 / 145.0 / 0.731 | 35 / 140.9 / 0.725 |
| tool-cold | 39 / 157.4 / 0.826 | 39 / 188.8 / 0.899 | 39 / 188.1 / 0.895 |
| json-cold | 30 / 114.8 / 0.566 | 35 / 136.6 / 0.732 | 35 / 135.1 / 0.726 |
| reasoning-cold | 49 / 121.8 / 1.120 | 48 / 147.3 / 1.211 | 48 / 148.9 / 1.200 |
| sampled_chat-cold | 62 / 128.6 / 2.757 | 54–70 / 139.2 / 2.625 | 57–76 / 130.1 / 3.189 |
| session-cold | 68 / 842.7 / 3.088 | 69 / 947.2 / 3.310 | 68 / 864.9 / 2.988 |
| session-session | 71 / 118.1 / 2.367 | 70 / 174.5 / 2.422 | 70 / 178.5 / 2.548 |

## Frozen quality checks

Counts below cover 24 measured natural requests per profile. A rubric pass covers its stated points, not every possible factual claim in the answer. The sampled-chat rubric specifically requires recomputation; several otherwise coherent retrieval analogies miss that requirement. This narrow rubric is retained without relaxing it after seeing results. Sampled speed is not used for a matched-seed claim.

| Profile | Automatic pass / fail | Manual rubric pass / fail |
|---|---:|---:|
| sharar-plain | 12 / 0 | 9 / 3 |
| sglang-plain | 9 / 3 | 9 / 3 |
| sharar-dflash2 | 12 / 0 | 9 / 3 |
| sglang-dflash2 | 9 / 3 | 9 / 3 |
| sglang-dflash2-prefill8192 | 9 / 3 | 9 / 3 |

Failed natural cells remain visible above; their completion times are not quality-qualified speed wins. The portable output review identifies every manual decision and preserves automatic failures.

## Observed memory

The shared budget is a limit, not an assertion that engines allocate the same amount. Peak host deltas include loading/compilation and may include other host activity. They are sampled every 250 ms and are not exact allocation attribution.

| Profile | Largest sampled host MemAvailable drop | Minimum host MemAvailable |
|---|---:|---:|
| sharar-plain | 26.12 GiB | 87.08 GiB |
| sglang-plain | 32.45 GiB | 80.77 GiB |
| sharar-dflash2 | 32.18 GiB | 81.07 GiB |
| sglang-dflash2 | 42.18 GiB | 70.36 GiB |
| sglang-dflash2-prefill8192 | 44.50 GiB | 69.07 GiB |

## Interpreting the speed gap

The 128-token single-request diagnostic illustrates why throughput alone
cannot establish a kernel advantage. Sharar emits a repetitive continuation in
35 verification rounds; both SGLang settings emit a different continuation in
108 rounds. Both return the required 256 tokens. The round-count ratio already
explains most of that cell's throughput gap. Preserve this as a measured
end-to-end observation; do not describe it as a threefold faster target kernel.
Native acceptance field definitions also differ; the actual verification counts
and output IDs are retained for inspection.

The next experiment holds target inputs and acceptance lengths fixed to split
verification, draft work, reconstruction and convolution-state cost. A compact
accepted convolution window is the first candidate only if that measured cost
supports a material client-visible gain. Broader natural sessions, cache
pressure and independent comparison windows remain necessary before proposer
promotion or a claim of a general lead over SGLang.

## Reproduction

Freeze inputs once in the pinned tokenizer environment, then measure each
exclusive server with the same file:

```bash
python3 -m bench.competitive --freeze /path/to/inputs.json
python3 -m bench.competitive --inputs /path/to/inputs.json \
  --url http://127.0.0.1:30000 --profile sglang-dflash2 \
  --output /path/to/new-results
```

The [manifest](../bench/workloads/competitive_v1.json) defines workload policy.
The [portable evidence](competitive-results.json) includes the guarded matrix
runner and per-profile launcher, so the exact flags and restoration procedure
are reviewable. Adapt host paths and device mounts to the local installation.
Raw results and excluded attempts are under
`$SHARAR_RESULTS/reviews/2026-09-05/competitive/`.

The state-capacity and text-only loading clarifications followed diagnostic
runs, before final SGLang measurements. The already completed Sharar plain profile kept its
unchanged resources and frozen work. Artifact generation verifies identical
prompt IDs, natural cases, sampling and repetition policy across all final
profiles; historical resource metadata is retained in each original report.
