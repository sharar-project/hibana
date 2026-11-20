"""Phase 2 static checks (GB10, one CUDA context, coexists with the serving engine).

1. The multimodal rope kernel on rows whose (t, h, w) components are equal is bitwise the
   original `_norm_rope_kernel` (the text path is unchanged wherever it runs).
2. The multimodal kernel on true image rows matches a torch reference of the interleaved
   section layout (per-pair axis = lane % 3, same cos_sin table).
3. `mrope.positions_3d` equals the vLLM reference algorithm on randomized prompts/grids.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from hibana.kernels.qk_norm_rope import qk_norm_rope
from hibana.model.flashnext.mrope import positions_3d

torch.manual_seed(0)
dev = "cuda"
T, H, D, R = 257, 3, 256, 64
theta = 10_000_000.0
eps = 1e-6
qkv = torch.randn(T, H * 2 * D + D, device=dev, dtype=torch.bfloat16)
qw = torch.randn(D, device=dev, dtype=torch.bfloat16)
kw = torch.randn(D, device=dev, dtype=torch.bfloat16)
inv = 1.0 / (theta ** (torch.arange(0, R, 2, dtype=torch.float32, device=dev) / R))
t = torch.arange(4096, dtype=torch.float32, device=dev)
freqs = torch.outer(t, inv)
cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

pos1 = torch.arange(1000, 1000 + T, device=dev, dtype=torch.int64)
pos3 = pos1[None].expand(3, -1).contiguous()

q_a, k_a = qk_norm_rope(qkv, pos1, cos_sin, qw, kw, H, 1, D, R, eps)
q_b, k_b = qk_norm_rope(qkv, pos3, cos_sin, qw, kw, H, 1, D, R, eps)
print("text rows: mrope kernel == original kernel:", torch.equal(q_a, q_b), torch.equal(k_a, k_b))

# true mixed rows: t random, h/w grid-like
pos3m = pos3.clone()
pos3m[1] = torch.arange(2000, 2000 + T, device=dev)
pos3m[2] = torch.arange(3000, 3000 + T, device=dev)
q_m, k_m = qk_norm_rope(qkv, pos3m, cos_sin, qw, kw, H, 1, D, R, eps)

def torch_ref(which, w):
    src = qkv[:, : D] if which == "q" else qkv[:, H * 2 * D: H * 2 * D + D]
    x = src.float()
    rstd = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / D + eps)
    y = (x * rstd * (1.0 + w.float())).to(torch.bfloat16).float()
    lane = torch.arange(R // 2, device=dev)
    axis = lane % 3
    pos = torch.where(axis == 0, pos3m[0, :, None], torch.where(axis == 1, pos3m[1, :, None], pos3m[2, :, None]))
    ang = pos.float() * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    y1, y2 = y[:, : R // 2], y[:, R // 2: R]
    out = torch.cat((y1 * cos - y2 * sin, y2 * cos + y1 * sin, y[:, R:]), dim=-1)
    return out.to(torch.bfloat16)

rq, rk = torch_ref("q", qw), torch_ref("k", kw)
print("mrope kernel == torch interleaved reference (q):", torch.equal(q_m[:, 0], rq) or (q_m[:, 0].float() - rq.float()).abs().max().item())
print("mrope kernel == torch interleaved reference (k):", torch.equal(k_m[:, 0], rk) or (k_m[:, 0].float() - rk.float()).abs().max().item())

# mixed-text-and-image prompt vs the vLLM algorithm, randomized
class Img:
    def __init__(s, start, gh, gw):
        s.start, s.grid, s.ids = start, (gh, gw), [0] * (gh * gw)
    @property
    def end(s):
        return s.start + len(s.ids)

rng = np.random.default_rng(7)
ok = True
for trial in range(50):
    n_img = int(rng.integers(1, 4))
    spans, at = [], 0
    for _ in range(n_img):
        t0 = int(rng.integers(0, 30))
        gh, gw = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        spans.append((at + t0, gh, gw))
        at += t0 + gh * gw + int(rng.integers(0, 20))
    L = at + int(rng.integers(1, 30))
    media = tuple(Img(s, gh, gw) for s, gh, gw in spans)
    got = positions_3d(L, media)
    llm, st = [], 0
    for (offset, gh, gw), img in zip(spans, media):
        n = gh * gw
        text_len = offset - st
        st_idx = llm[-1].max() + 1 if llm else 0
        llm.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)
        llm.append(np.indices((1, gh, gw)).reshape(3, -1) + text_len + st_idx)
        st = offset + n
    st_idx = llm[-1].max() + 1
    llm.append(np.broadcast_to(np.arange(L - st), (3, L - st)) + st_idx)
    ref = np.concatenate(llm, axis=1)
    if not np.array_equal(got, ref):
        ok = False
        print("MISMATCH", trial)
print("positions_3d == vLLM algorithm over 50 random prompts:", ok)
