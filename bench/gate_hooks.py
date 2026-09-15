"""Candidate hooks for the family gate (`tools/family-gate.py`, `bench.gb10_family_eval run --candidate`).

A hook is `fn(stage, executor, args)` called at 'model' (after the weights
load; executor is None), 'spec' (after enable_spec, before graph capture) and
'ready' (after warmup). Hooks here are the live ones the gate runs; completed
screens keep theirs in `bench/archive/gb10_family_candidates.py` and are
re-exported so a gate order can name them without the archive path.
"""
from __future__ import annotations

from bench.archive.gb10_family_candidates import (  # noqa: F401  re-exported for `--candidates`
    draft_nvfp4_body,
    draft_nvfp4_body_fc8,
    exact_bundle_fp8_m16,
    fp8_m16,
    top32,
)


def stable_numerics(stage, ex, args):
    """Shape-stable prefill numerics (`sharar/kernels/stable.py`), the research
    baseline for any change that alters the verify row count: the b/a projection,
    the FP8 5120x6144 window and the gated norm stop depending on M."""
    if stage == 'model':
        from sharar.kernels import stable
        stable.ENABLED = True
        print('  stable_numerics: SHARAR_STABLE_NUMERICS path enabled for this process', flush=True)


def bf16_draft(stage, ex, args):
    """Marker hook: the gate order token `B` runs the BF16 draft; this exists so a
    candidate that must pair with the BF16 draft (draft_nvfp4_body) has a matching control."""
    return None
