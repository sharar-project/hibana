# September 11 repository cleanup

The pre-cleanup tree is preserved by `archive/pre-cleanup-2026-09-11` at
`3655d7f7d57f68c905f373281a76a9c56368705a`. Current documentation is indexed
in [docs/README.md](../README.md); original studies and decisions remain
[searchable in the archive](README.md).

## Completed

- Include the runtime CUDA source in source distributions and wheels. CI now
  builds an sdist and its wheel, verifies packaged assets and probes installed
  imports and the benchmark CLI outside the source checkout.
- Remove the unused seven-proposal limit from ServingProfile. Startup continues
  to validate proposer-specific limits, including DFlash2 at 15.
- Cover `tool_choice: none` in streaming and non-streaming tests and describe
  its actual schema-omission behavior in the current API guide.
- Extract engine-thread/ticket ownership into `engine/loop.py` and HTTP input
  normalization into `api/validation.py`. The moved definitions are AST-identical
  to their previous implementations; lifecycle regression coverage is retained.
- Extract shared evidence, recording, HTTP, retained-batch and CUDA-timing
  helpers into `bench/support/`. Source identity now includes helper modules.
  Ten completed experimental implementations moved to `bench/archive/` with
  their imports and reference tests updated.
- Retire the misleading prefix-capacity comparison with a migration message.
  Its original duplicate-save protocol remains reproducible at `ea6a9de`.
- Consolidate the frozen source fixture in the installed package. All current
  readers use the same content, and the original manifest/checksum is preserved.
- Separate current operational/API/performance guides from dated reports,
  keeping all 29 JSON evidence manifests byte-identical beside their reports.
  The checker now discovers manifests recursively and validates local links.
- Reduce README from 321 to 92 lines, roadmap from 1,616 to 78, and the main
  benchmark guide from 477 to 109. Historical guidance is explicitly labeled.

## Validation and scope

The cleanup passes 454 CPU tests with eight GPU-only skips. Distribution
checks pass for both the sdist and its wheel. Every archived evidence manifest
matches its pre-cleanup bytes. Local Markdown file links and Python parsing
pass; external URLs and heading fragments are outside that link check.

Model/kernel arithmetic, scheduling policy and feature-retention behavior are
unchanged. No new throughput gain is claimed, and production is not redeployed.
Memory-pressure admission, asynchronous restore, reclaimable cache slabs and
constrained output remain separate engineering tasks in the
[roadmap](../../roadmap.md). The existing experimental fused-norm switch remains
off by default, with its numerical reference tests retained.

The previous GitHub Actions run was blocked before execution by account
billing/spending limits. Local validation does not imply that hosted CI ran;
check the pushed commit's Actions status separately.
