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
    """Three-bit decode weights for every MLP and FP8 projection (sharar/kernels/w3.py), NVFP4/FP8 kept for prefill."""
    if stage == 'model':
        from sharar.engine import config
        config.install(config.current().replace(w3_decode=True))
        print('  w3_decode: 3-bit decode path enabled; packing after load', flush=True)
    if stage == 'spec':
        from sharar.kernels import w3
        w3.attach_model(ex.model, log=print)


def w3_head(stage, ex, args):
    """With w3_decode: also serve the lm_head from a 3-bit copy (SHARAR_W3_HEAD=1). Order it before w3_decode."""
    if stage == 'model':
        from sharar.engine import config
        config.install(config.current().replace(w3_head=True))


def w3_draft(stage, ex, args):
    """With w3_decode: also serve the parallel draft's FP8 matrices from 3-bit copies (SHARAR_W3_DRAFT=1). Order it before w3_decode."""
    if stage == 'model':
        from sharar.engine import config
        config.install(config.current().replace(w3_draft=True))


def w3_decode_head(stage, ex, args):
    """3-bit decode copies for the layers and the lm_head."""
    w3_head(stage, ex, args)
    w3_decode(stage, ex, args)


def w3_decode_head_draft(stage, ex, args):
    """3-bit decode copies for the layers, the lm_head and the parallel draft."""
    w3_head(stage, ex, args)
    w3_draft(stage, ex, args)
    w3_decode(stage, ex, args)
