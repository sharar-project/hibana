"""Candidate hooks for the family gate (`tools/family-gate.py`, `bench.gb10_family_eval run --candidate`).

A hook is `fn(stage, executor, args)` called at 'model' (after the weights
load; executor is None), 'spec' (after enable_spec, before graph capture) and
'ready' (after warmup). Hooks here are the live ones the gate runs; completed
screens are in git history (tag `archive/pre-cleanup-2026-09-26`).
"""
from __future__ import annotations


def stable_numerics(stage, ex, args):
    """Shape-stable prefill numerics (`sharar/kernels/stable.py`), the research
    baseline for any change that alters the verify row count: the b/a projection,
    the FP8 5120x6144 window and the gated norm stop depending on M."""
    if stage == 'model':
        from sharar.engine import config
        config.install(config.current().replace(stable_numerics=True))
        print('  stable_numerics: SHARAR_STABLE_NUMERICS path enabled for this process', flush=True)


def bf16_draft(stage, ex, args):
    """Marker hook: the gate order token `B` runs the BF16 draft; this exists so a
    candidate that must pair with the BF16 draft (draft_nvfp4_body) has a matching control."""
    return None


def w3_decode(stage, ex, args):
    """Three-bit decode weights for every MLP and FP8 projection (sharar/kernels/w3.py), NVFP4/FP8 kept for
    prefill; the parallel draft's FP8 matrices get copies too when it is built after this stage."""
    if stage == 'model':
        from sharar.engine import config
        config.install(config.current().replace(w3_decode=True))
        print('  w3_decode: 3-bit decode path enabled; packing after load', flush=True)
    if stage == 'spec':
        from sharar.kernels import w3
        w3.attach_model(ex.model, log=print)
