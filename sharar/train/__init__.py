"""Draft adaptation: train the DFlash2 drafter on request traces.

Only the drafter changes. The served target, its NVFP4 vocabulary head and the
verifier stay fixed, so a trained drafter changes speed and never outputs.
See docs/draft-adaptation.md. Training runs in the runtime container on the
GPU while nothing is being served; it loads its own copy of the target.
"""
