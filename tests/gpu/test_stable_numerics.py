"""The M-invariant b/a kernel matches an FP32 reference and does not depend on M; the FP8 pad is exact."""

import torch



@torch.inference_mode()
def test_ba_project_is_row_invariant_and_close_to_fp32():
    from hibana.kernels.ba_proj import ba_project
    torch.manual_seed(0)
    x = torch.randn(3000, 5120, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(96, 5120, device='cuda', dtype=torch.bfloat16) * 0.02  # 2 x 48 value heads
    full = ba_project(x, w)
    for m in (1, 5, 16, 17, 26, 27, 33, 115, 425, 897, 1697, 2625):
        assert torch.equal(ba_project(x[:m], w), full[:m]), m
    reference = (x.float() @ w.float().T)
    assert ((full.float() - reference).abs() / reference.abs().clamp_min(1e-3)).median() < 4e-3
    assert torch.equal(ba_project(x[:40].view(2, 20, 5120), w), full[:40].view(2, 20, 96))


@torch.inference_mode()
def test_fp8_window_pad_reproduces_the_wide_algorithm(monkeypatch):
    from hibana import config
    from hibana.kernels import ops
    from hibana.weights.modelopt import load_fp8_linear
    from hibana.weights.safetensors import SnapshotWeights
    w = SnapshotWeights(config.snapshot_dir())
    try:
        out_proj = load_fp8_linear(w, ['model.language_model.layers.0.linear_attn.out_proj'], 'cuda')
    finally:
        w.close()
    x = torch.randn(64, 6144, device='cuda', dtype=torch.bfloat16)
    from hibana.engine import config as engine_config
    engine_config.install(engine_config.current().replace(stable_numerics=False))
    wide = ops.fp8_linear(x, out_proj)
    narrow = ops.fp8_linear(x[:30], out_proj)
    assert not torch.equal(narrow, wide[:30])  # cuBLASLt's 27..32 algorithm
    engine_config.install(engine_config.current().replace(stable_numerics=True))
    assert torch.equal(ops.fp8_linear(x[:30], out_proj), wide[:30])
    assert torch.equal(ops.fp8_linear(x[:26], out_proj), wide[:26])
