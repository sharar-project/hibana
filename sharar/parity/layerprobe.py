"""Per-block parity for chosen layers, feeding each block the reference's dumped input.

Usage: python -m sharar.parity.layerprobe <reference_dir> <prompt> <layer,layer,...>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from sharar import config
from sharar.kernels import ops
from sharar.model.qwen38 import DecoderLayer
from sharar.model.rope import PartialRotary
from sharar.weights.safetensors import SnapshotWeights


def cos(a, b):
    a = a.float().flatten()
    b = b.float().to(a.device).flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main(reference: str, prompt: str, layers: str) -> int:
    device = torch.device("cuda")
    snap = config.snapshot_dir()
    cfg = config.load_model_config(snap)
    types = config.layer_types(cfg)
    w = SnapshotWeights(snap)
    ref = torch.load(Path(reference) / "dump" / f"{prompt}.pt", map_location="cpu")
    ids = ref["input_ids"].to(device)
    t = ids.shape[0]
    positions = torch.arange(t, device=device)
    rope = PartialRotary(config.HEAD_DIM, 64, config.ROPE_THETA, config.MAX_POSITION_EMBEDDINGS, device)
    ragged = ops.RaggedAttention(device, config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM)
    print(f"{prompt}: T={t}")
    for i in (int(x) for x in layers.split(",")):
        layer = DecoderLayer(w, i, types[i], device, rope, ragged, True)
        h_i, r_i = ref["layers"][i]
        if i == 0:
            block_in = ref["embed"]
        else:
            h_p, r_p = ref["layers"][i - 1]
            block_in = (h_p.float() + r_p.float()).to(torch.bfloat16)
        block_in = block_in.to(device)
        mixer_ref = r_i.float().to(device) - block_in.float()
        with torch.inference_mode():
            xn = layer.input_norm(block_in)
            if types[i] == "full_attention":
                mixer_mine = layer.mixer.prefill(xn, positions)
            else:
                mixer_mine = layer.mixer.prefill(xn)
            mlp_mine = layer.mlp(layer.post_norm(r_i.to(device)))
            # whole layer from the reference input
            h_full, r_full = layer.prefill(block_in, None, positions) if i == 0 else layer.prefill(h_p.to(device), r_p.to(device), positions)
        stream_ref = h_i.float() + r_i.float()
        stream_full = h_full.float() + r_full.float()
        kind = "attn" if types[i] == "full_attention" else "gdn "
        print(
            f"  layer{i:02d} {kind}: mixer cos {cos(mixer_mine, mixer_ref):.5f}  mlp cos {cos(mlp_mine, h_i):.5f}  "
            f"layer stream cos {cos(stream_full, stream_ref):.5f}  (mixer scale {mixer_ref.abs().mean().item():.4f}, mlp scale {h_i.float().abs().mean().item():.4f}, stream scale {stream_ref.abs().mean().item():.3f})"
        )
        del layer
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
