#!/usr/bin/env python3
"""Print shapes, dtypes and scale values of representative checkpoint tensors."""
import json
import sys
from pathlib import Path

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sharar import config  # noqa: E402

snap = config.snapshot_dir()
idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
handles = {}


def get(n):
    f = idx[n]
    if f not in handles:
        handles[f] = safe_open(str(snap / f), "pt", device="cpu")
    return handles[f].get_tensor(n)


L = "model.language_model.layers"
for l in (0, 3, 63):
    print("layer", l)
    if l % 4 == 3:
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = get(f"{L}.{l}.self_attn.{p}.weight")
            print(f"  {p}: {tuple(w.shape)} {w.dtype} ws={get(f'{L}.{l}.self_attn.{p}.weight_scale').item():.5g} is={get(f'{L}.{l}.self_attn.{p}.input_scale').item():.5g}")
        for p in ("q_norm", "k_norm"):
            w = get(f"{L}.{l}.self_attn.{p}.weight")
            print(f"  {p}: {tuple(w.shape)} {w.dtype} mean={w.float().mean():.4f}")
    else:
        for p in ("in_proj_qkv", "in_proj_z", "out_proj"):
            w = get(f"{L}.{l}.linear_attn.{p}.weight")
            print(f"  {p}: {tuple(w.shape)} {w.dtype} ws={get(f'{L}.{l}.linear_attn.{p}.weight_scale').item():.5g} is={get(f'{L}.{l}.linear_attn.{p}.input_scale').item():.5g}")
        for p in ("in_proj_a.weight", "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight"):
            w = get(f"{L}.{l}.linear_attn.{p}")
            print(f"  {p}: {tuple(w.shape)} {w.dtype} mean={w.float().mean():.4f}")
    for p in ("gate_proj", "up_proj", "down_proj"):
        w = get(f"{L}.{l}.mlp.{p}.weight")
        s = get(f"{L}.{l}.mlp.{p}.weight_scale")
        print(f"  {p}: {tuple(w.shape)} {w.dtype} scale {tuple(s.shape)} {s.dtype} ws2={get(f'{L}.{l}.mlp.{p}.weight_scale_2').item():.5g} is={get(f'{L}.{l}.mlp.{p}.input_scale').item():.5g}")
    for p in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        w = get(f"{L}.{l}.{p}")
        print(f"  {p}: {tuple(w.shape)} {w.dtype}")
w = get("lm_head.weight")
print("lm_head", tuple(w.shape), w.dtype, "ws2", get("lm_head.weight_scale_2").item(), "is", get("lm_head.input_scale").item(), "scale", tuple(get("lm_head.weight_scale").shape), get("lm_head.weight_scale").dtype)
w = get("model.language_model.embed_tokens.weight")
print("embed", tuple(w.shape), w.dtype)
w = get("model.language_model.norm.weight")
print("final norm", tuple(w.shape), w.dtype)
for n in sorted(k for k in idx if k.startswith("mtp")):
    w = get(n)
    print(n, tuple(w.shape), w.dtype)
