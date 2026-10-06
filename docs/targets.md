# Qwen3.8-27B or Qwen3.8-Flash-Next

Sharar serves both and neither is the default (`sharar-ctl serve 27b|flashnext`). This page compares
them under the default `fast` profile on one GB10, measured 2026-10-05/06 for #11. Evidence:
`$SHARAR_RESULTS/reviews/2026-10-05-issue11/` (sessions, long context, RULER), `2026-10-06-evaluate-issue11-*`,
`2026-10-06-h2h-27b-*` and `2026-09-30-harder-evals` (Flash-Next's hard battery).

**In short.** Flash-Next is the stronger model and the better agent: higher on every hard eval, a long
Claude Code session in 19 minutes against 31, and faster at long context. The 27B decodes greedy
output faster on short prompts, downloads and loads faster (26 GB against 135 GB, about 30 s against
2.5 min) and holds 28 GiB of weights against 70.

## A long Claude Code session

One conversation of five turns on [click](https://github.com/pallets/click) at d036881: explain the help
system, then three upstream changes (f67c2bb, f25f697 + 05f6fd0, 9f0bcdf), then run the whole suite, add
changelog entries and commit. Graded by the upstream commits' tests over the agent's code: 30 tests fail
before. Claude Code `-p` with edits and test commands allowed, through `sharar-ctl launch claude`; the
serving image of main at d3aee34 plus #115/#116 (an omitted temperature is 1.0; the 27B samples at the
2:1.0 threshold).

| | 27B | Flash-Next |
|---|---:|---:|
| Fail-to-pass tests | 30/30 | 30/30 |
| Upstream suite (2,262 tests) | clean | clean |
| Wall time, 5 turns | 31 min | 19 min |
| Requests / tool calls | 112 / 131 | 77 / 97 |
| Output tokens | 96.6K | 69.3K |
| Decode rate in the session | 58 tok/s | 67 tok/s |
| Uncached prefill rate | 724 tok/s | 939 tok/s |
| Largest prompt | 194K tokens | 131K tokens |

Both models finished every turn. The 27B took more requests and wrote more to reach the same result; its
longest turn (the `Parameter.spec` change) was 15 minutes against 9.

Two server bugs this found, both fixed:
- **#115.** Claude Code sends no sampling fields, and the Messages API treated that as greedy. The 27B's
  greedy thinking looped to the 32K `max_tokens` in three of five turns (4/30 tests, 85 minutes). An
  omitted temperature is now Anthropic's 1.0.
- **#115, #116.** At temperature 1, the `fast` profile's relaxed acceptance for sampled requests (3:2.0)
  still made the 27B loop (a looped request replayed: 4/6 seeds; the session: 60 minutes, one turn timed
  out). At 2:1.0 it did not (0/6; the session above). Flash-Next did not loop in any configuration.

## Speed

One request at a time, h2h decode workload (22 prompts, thinking off, median tok/s; README):

| | 27B | Flash-Next |
|---|---:|---:|
| Greedy | 92.3 | 85.4 |
| Temperature 0.7 | 66.5 | 74.4 |
| Time to first token, short prompt | 0.09 s | 0.17 s |

Past 32K tokens Flash-Next is the faster one (`evals.long_context`, greedy, fast profile; prefill is the
cold time to first token of a needle prompt):

| Context | Decode, 27B | Decode, Flash-Next | Prefill, 27B | Prefill, Flash-Next |
|---|---:|---:|---:|---:|
| 8K | 78.2 | 76.8 | 3.0 s | 5.7 s |
| 32K | 76.0 | 73.8 | 15.4 s | 19.1 s |
| 128K | 54.2 | 65.5 | 112 s | 79 s |
| 256K | 46.4 | 57.2 | 362 s | 178 s |

Flash-Next attends sparsely past 2,048 tokens ([flashnext.md](flashnext.md#sparse-attention-past-2048-tokens-2026-09-23)),
so its prefill and decode fall off more slowly with context.

## Quality

| | 27B | Flash-Next |
|---|---:|---:|
| Needle at 8K-256K, three depths | 12/12 | 12/12 |
| RULER v2 32K (120) | 95.4 | 94.0 |
| RULER v2 128K (120) | 82.2 | 85.6 |
| MMLU-Pro (1,400) | 75.4 | 81.6 |
| GPQA Diamond (198) | 59.6 | 73.7 |
| IFEval strict (541) | 78.7 | 84.8 |
| WMT24++ chrF++ (1,000) | 46.4 | 49.3 |

RULER ran at four streams on Flash-Next; the 27B's 128K half ran at one, because at four a request
waited past the client's 900 s timeout behind the others' prefill. The hard battery is greedy with thinking off, four streams, token caps MMLU-Pro 4,096 and GPQA 8,192.
The 27B's ran on 2026-10-06 (sharar-eval 7ae2182); Flash-Next's is the `fast` arm of 2026-09-30
(the same tasks and caps on the harness before sharar-eval). The 27B hit the cap more often (MMLU-Pro 136
against 109, GPQA 52 against 28).

## Footprint

| | 27B | Flash-Next |
|---|---:|---:|
| Download | 26 GB (plus the 3-bit checkpoint) | 135 GB |
| Load to healthy, warm page cache | about 30 s | about 2.5 min |
| Weights in memory (`/get_server_info` ledger) | 28 GiB | 70 GiB |
| KV and state pools at 262K context | 28 GiB | 6 GiB |
