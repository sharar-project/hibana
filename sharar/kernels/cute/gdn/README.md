# Serving GDN prefill on SM12x

Derived from FlashInfer 0.6.17's `gdn_kernels/delta_rule_dsl` implementation,
with Apache-2.0 license retained here. The tuned local source came from
`profile/gdn-prefill-flashinfer-state-io-gb10/kernel/`: 32-token chunks,
coalesced FP32 state transfers through a padded shared-memory tile, and the
same warp-specialized pipeline. Required FlashInfer/CuTe versions are pinned
in `constraints/gb10-cu130.txt`; upstream helper modules are imported lazily.

Production calls go through `sharar.kernels.gdn_prefill`, not this low-level API.
It normalizes Q/K using the original helper, packs arbitrary state-pool slots,
runs the kernel, then scatters final and requested checkpoint states back.
There is no dependence on files in `profile/` or `kernel/` at runtime.

`SHARAR_GDN_PREFILL=auto` selects this implementation for BF16 D128 on SM120/121
with supported head ratios and contiguous FP32 state. `triton` forces the
original backend; `cute` forces this one and rejects unsupported geometry.
Auto retains Triton under `SHARAR_STABLE_NUMERICS=1`. Compilation/launch failures
are surfaced, not silently retried with a different backend. Decode,
speculative verification, scheduler budgets and prefill caps are unchanged.

## Prefix checkpoints

Unlike FLA's entering-64-token-chunk history, snapshots are the FP32 state
after the requested boundary. The CPU prefill plan contains a sparse lookup:
per-sequence offsets followed by compact snapshot indices (-1 means skip).
The kernel probes at the greatest-common-divisor interval of requested local
boundaries, but writes ONLY requested states. A tail boundary such as8128
therefore does not force128 full state snapshots for an8K prefill. Final states
and checkpoint snapshots use the same bit-preserving coalesced store.

The low-level API also retains dense periodic snapshots when
`indexed_checkpoints=False`; serving uses `True` with the sparse lookup buffer.
Only32-token-aligned intervals are supported; the existing serving planner
continues to request64-token-aligned boundaries. The checkpoint extension does
not promise numerical identity with the original Triton64-token algorithm.

TP uses this rank's head counts, not hardcoded global geometry. Tests cover
16K/48V and8K/24V heads, ragged inputs, nonconsecutive slots, reordered prefix
snapshots, continuation, untouched slots, FP32 recurrence and graph replay.
These rank-local tests are not a two-machine TP/RDMA integration test.

```bash
SHARAR_RUN_GPU=1 uv run python -m pytest -q \
  tests/gpu/test_gdn_cute.py tests/gpu/test_gdn_plan.py
```
