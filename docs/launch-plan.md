# Launch plan

Started 2026-10-04. Goal: make Sharar the engine GB10 owners install first. It is already the
fastest at batch one on both targets (#26), so the remaining work is mostly distribution, harness
reliability and agent-shaped latency, with a few speed and drafter items that users notice.
Items that have an issue link to it; the rest were agreed in discussion and have no issue yet.
[roadmap.md](../roadmap.md) stays the engine work order; this page tracks what stands between
the engine and its users.

Status: `todo`, `doing`, `done`, `parked`, `decide` (needs a decision before work starts).

## 1. One command to install and serve

| Item | Issue | Status | Notes |
|---|---|---|---|
| Decide: public repository | none | decide | Everything public below depends on it. A public image exposes the source too (it is inside the image). |
| Prebuilt image on `ghcr.io/abdelfattah-lab/sharar` | none | done | Merged in #101 (2026-10-04): the launchers default to `ghcr.io/abdelfattah-lab/sharar:latest` and pull it when missing, `sharar-ctl pull`, images carry their commit, the nightly check publishes main after the GPU suite and serve checks pass (CI token push access verified). Private with the repository; container storage is free on GitHub. Make it public with the repository. |
| Smaller image | none | todo | The runtime uses the CUDA devel base because FlashInfer SM121 modules and the GEMV extension JIT-compile; building them ahead of time allows a runtime base. |
| `sharar serve 27b` / `sharar serve flashnext` | none | done | Merged in #103 (2026-10-05): `sharar-ctl serve 27b|flashnext` (stop/status/logs/chat follow the serving target); settings in `config/targets/<target>.env`; `config/gb10.env` optional; `config/flashnext.env` removed. |
| Automatic weight download | none | done | Merged in #103: missing pinned snapshots (model and drafter) are downloaded after a free-disk check; tested from an empty cache with no config (27B: 26 GB plus the 3-bit checkpoint, 19 min to launch). Flash-Next is 135 GB. |
| Preflight for any GB10 machine | none | done | The launcher's preflight covers the blocking checks (GPU, driver, memory, disk before downloads, image login hint) and, with #107, Docker access and user systemd with their fixes. A full one-report `sharar-ctl doctor` (machine vendor, every check at once) is parked on `feat/doctor` (#106, closed) for the public launch. |
| `sharar launch claude\|codex\|opencode\|aider\|...` | none | doing | PR #109: `sharar-ctl launch claude|opencode|aider|show`, each agent with its own config for the serving target; all three fixed a bug end to end on the 27B (8-14 s). Codex waits for #19. Found #108: the model retries a denied tool call instead of stopping. |

## 2. Agent harnesses that work every time

| Item | Issue | Status | Notes |
|---|---|---|---|
| Forced tool choice and structured output with speculation | #16 | todo | |
| OpenAI Responses API (Codex) | #19 | todo | Codex cannot use Sharar without it. |
| Claude Code session on Flash-Next end to end | #11 | todo | |
| Tool-call parsing hardened on real agent runs | relates to #28 | todo | Use sharar-eval `agentic` (Terminal-Bench) transcripts as the regression set. Also #108: retries of a denied tool call (753 in one Claude Code run). |
| Tested harness matrix in the README | none | todo | Claude Code, Codex, OpenCode, Aider, Cline, Continue, Zed, Pi: works / known issues. |

## 3. Agent-shaped speed: long prompts, short answers

| Item | Issue | Status | Notes |
|---|---|---|---|
| Sparse prefill | #67, PR #80 | doing | Agent turns start with 20-60K-token prompts; prefill dominates the first turn. |
| Faster first prefill at long context | #14 | todo | |
| Choose the profile from load | none | todo | Switch between `fast` and `throughput` (#59) by the number of live streams, so parallel subagents do not need a restart. Related: #43 adaptive draft depth. |
| Time to finish an agent task as the headline metric | none | todo | Measured with sharar-eval `agentic`, next to tok/s. |

## 4. Drafters and thinking

| Item | Issue | Status | Notes |
|---|---|---|---|
| Published drafter packs (coder, agent, chat) | relates to #12, #40, #29, #33 | todo | Today the trained-drafter gain (+12.4% on repository coding, [draft adaptation](draft-adaptation.md)) needs the user's own traffic first. Ship drafters trained per domain (self-play traces, #33) so a new install starts with it, then keep adapting on the user's traffic. Lossless: the drafter never changes outputs under exact acceptance. Pack selection by `sharar-ctl draft use <pack>`. |
| Thinking budget | none | todo | Map `reasoning_effort` / a token budget to the thinking length. Fewer tokens is the cheapest end-to-end speedup (the idea behind Strata's "Swift" fine-tune). |
| Serve fine-tunes of a supported architecture | relates to #40, #12, #51 | todo | Load any Qwen3.8-27B / Flash-Next fine-tune; when its MTP or drafter acceptance collapses, retrain the drafter with `sharar train`. |

## 5. Honest numbers and the launch

| Item | Issue | Status | Notes |
|---|---|---|---|
| Re-run h2h on current main and update the README | #26 | done | Merged in #105 (2026-10-05, published image, fast profile): 27B 92.3 / 71.5, Flash-Next 85.4 / 74.4 tok/s greedy / t=0.7. The 27B sampled number is 3 tok/s under the old README; repeat before reading into it. |
| Add llama.cpp and Strata to the h2h | #26 | todo | Strata (github.com/Niko1221/Strata) serves Flash-Next on consumer GPUs and claims 94 tok/s at Q2_0 on an RTX 5070; users will compare headline numbers, so publish quality next to speed. |
| Benchmark page and demo | none | todo | Reproducible commands, image tags, precisions; a short screen recording of a coding agent on Sharar. |
| GB10 machines from other vendors | none | todo | ASUS Ascent GX10, Dell Pro Max with GB10, HP ZGX Nano G1n, Lenovo ThinkStation PGX, MSI EdgeXpert, Gigabyte AI TOP Atom and Acer Veriton GN100 all use the same GB10 and 128 GB LPDDR5X at 273 GB/s, so Sharar should run unchanged. Untested: OS image, NVMe size (1 TB models are tight with both targets, about 200 GB of weights and the PLE table) and power or thermal limits that could move the numbers. Ask early users for `sharar-ctl` preflight reports. |

## 6. Research tracks (after the launch work, or for a paper)

| Item | Issue | Status | Notes |
|---|---|---|---|
| Domain-pruned experts (Strata's "Coder": half the experts removed, 91% of SWE-bench Verified) | #39, #34 | parked | On the GB10 this buys memory, not speed: decode reads only the active experts (10 per layer), and all of Flash-Next is resident. The router data (#32) shows domains share few experts (code vs prose Jaccard about 0.01) and new prompts of one domain overlap only 0.26-0.44, so a fixed cut has to be calibrated on the user's own traffic. Worth it for co-residency of both targets, paired with on-policy distillation (#34). |
| Lazy weight loading from storage | relates to PR #78, #39, #10 | parked | Not a speed win while everything fits: a token at batch one reads about 1.3 GB of experts (48 layers x 10 x 2.76 MB) and NVMe is single-digit GB/s against 273 GB/s memory, so every miss is ~50x slower. It helps (a) start-up: serve while the rest loads (Flash-Next loads in 115 s), (b) keeping both targets loaded with cold experts on disk, (c) future models above 128 GB. Sharar already does it where it fits: the 48 GB Flash-Next PLE table is read from NVMe by row, because the rows are known from the token ids (#10 prefetches them). PR #78 is KV/session tiers on disk, not weights. |
| Local learning of the target model (RL / on-policy distillation) | #34, #33 | parked | The drafter learning on traffic is the local-learning feature users get today. Training the target is a separate repository when it starts; its first use is recovering quality lost to pruning or 3-bit experts (#73). |
