"""Probe the GDN checkpoint semantics on one layer: which h index holds the
state after 64*c tokens, and the conv window order, versus a fresh prefill."""
import sys

import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.kernels import ops
from sharar.memory.state_pool import StatePool
from sharar.model.qwen38 import PREFIX, DecoderLayer
from sharar.model.rope import PartialRotary
from sharar.parity.prompts import random_ids
from sharar.weights.safetensors import SnapshotWeights

device = torch.device("cuda")
w = SnapshotWeights(config.snapshot_dir())
rope = PartialRotary(config.HEAD_DIM, 64, config.ROPE_THETA, 8192, device)
ragged = ops.RaggedAttention(device, config.NUM_ATTENTION_HEADS, config.NUM_KV_HEADS, config.HEAD_DIM)
layer = DecoderLayer(w, 0, "linear_attention", device, rope, ragged, True)
gdn = layer.mixer
pool = StatePool(8, device)
ids = torch.tensor(random_ids(512, 3), device=device)
embed = torch.nn.functional.embedding(ids, w.get(f"{PREFIX}.embed_tokens.weight").to(device))
x = layer.input_norm(embed)

# fresh prefill of the first 256 tokens into slot 1
ctx_a = ForwardContext(mode="prefill", kv_pool=None, state_pool=pool, positions=torch.arange(256, device=device), slot=1, seq_len=256)
pool.clear(1)
with torch.inference_mode():
    gdn.prefill(x[:256], ctx_a)
ref_ssm = pool.ssm[0][1].clone()
ref_conv = pool.conv[0][1].clone()

# full 512 prefill into slot 2 with a checkpoint requested at 256 -> slot 3
ctx_b = ForwardContext(mode="prefill", kv_pool=None, state_pool=pool, positions=torch.arange(512, device=device), slot=2, seq_len=512,
                       checkpoint_positions=[256], checkpoint_slots=[3])
pool.clear(2)
pool.clear(3)
with torch.inference_mode():
    gdn.prefill(x, ctx_b)
cp_ssm, cp_conv = pool.ssm[0][3], pool.conv[0][3]
scale = ref_ssm.abs().max().item()
print(f"checkpoint ssm vs fresh: max abs {(cp_ssm - ref_ssm).abs().max().item():.5f} (state max {scale:.4f}), cos {torch.nn.functional.cosine_similarity(cp_ssm.flatten(), ref_ssm.flatten(), dim=0).item():.6f}")
print(f"checkpoint conv vs fresh: max abs {(cp_conv.float() - ref_conv.float()).abs().max().item():.4f}; flipped order max abs {(cp_conv.flip(-1).float() - ref_conv.float()).abs().max().item():.4f}")
# also compare against raw h chunks around index 4 to find the right offset
mixed, z, b, a = gdn._project(x)
from sharar.kernels.conv1d import causal_conv1d_fn
from sharar.kernels.fla.chunk import chunk_gated_delta_rule
from sharar.kernels.fla.fused_gdn_gating import fused_gdn_gating
pool.clear(4)
qsl = torch.tensor([0, 512], dtype=torch.int32, device=device)
ci = torch.tensor([4], dtype=torch.int32, device=device)
with torch.inference_mode():
    conv_out = causal_conv1d_fn(mixed.transpose(0, 1), gdn.conv_weight, None, activation="silu", conv_states=pool.conv[0], has_initial_state=torch.tensor([False], device=device), cache_indices=ci, query_start_loc=qsl, seq_lens_cpu=[512]).transpose(0, 1)[:512]
    q, k, v = torch.split(conv_out, [2048, 2048, 6144], dim=-1)
    g, beta = fused_gdn_gating(gdn.A_log, a, b, gdn.dt_bias)
    core, _, h = chunk_gated_delta_rule(q=q.reshape(1, 512, 16, 128), k=k.reshape(1, 512, 16, 128), v=v.reshape(1, 512, 48, 128), g=g, beta=beta, initial_state=pool.ssm[0], initial_state_indices=ci, cu_seqlens=qsl, head_first=False, use_qk_l2norm_in_kernel=True)
print("h shape", tuple(h.shape), h.dtype)
for c in (3, 4, 5):
    hc = h[0, c].float()
    print(f"  h[0,{c}] vs fresh-256 final: max abs {(hc - ref_ssm).abs().max().item():.5f} cos {torch.nn.functional.cosine_similarity(hc.flatten(), ref_ssm.flatten(), dim=0).item():.6f}")
for name, rows in (("rows 253..255", mixed[253:256]), ("rows 254..256", mixed[254:257])):
    cand = rows.transpose(0, 1)
    print(f"  conv from {name}: max abs {(cand.float() - ref_conv.float()).abs().max().item():.4f}; flipped {(cand.flip(-1).float() - ref_conv.float()).abs().max().item():.4f}")
