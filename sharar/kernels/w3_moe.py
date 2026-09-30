"""Three-bit routed experts for Flash-Next (#73): the W3 format of `kernels/w3.py` per expert, and a grouped
decode kernel for up to 16 rows (`csrc/w3_gemm.cu`, namespace `moe`).

Format: each expert is two W3 matrices, fc1 [1280, 2560] and fc2 [2560, 640]: 3-bit codes into an eight-level
E4M3 codebook, one E4M3 scale per 32 weights, an FP32 multiplier per row (3.25 bits per weight plus 4 bytes a
row; 2.01 MB an expert against NVFP4's 2.76 MB). fc1's rows are reordered so every 32-row block holds sixteen
up rows and then the sixteen matching gate rows: a warp of the kernel then has an intermediate column's up and
gate values in the same lane and applies silu(gate) * up in its epilogue. Meant to replace the NVFP4 experts,
not to sit next to them (a prototype: the layer does not use it yet).

`from_moe_layer` converts a MoeLayer loaded with SHARAR_FLASHNEXT_EXPERTS_SIM3=32 (weights/expert_sim3.py)
without rounding again: its levels {+-0.5, +-1.5, +-3, +-4} become the codebook, its E4M3 group scales the
group scales, the expert's NVFP4 global scale the row multiplier, so the kernel computes on exactly the weights
the lossy gate screened. Activations are E4M3 per (row, 32-group) as in the trunk's W3 path (`w3.quant_groups`),
the intermediate likewise; CUTLASS's fused path quantizes both to NVFP4 instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar.kernels import w3

HIDDEN, INTER, TOPK, EXPERTS, MAX_ROWS = 2560, 640, 10, 512, 16
SIM3_CODEBOOK = (-4.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 4.0)
_PLAN_WORDS = 1 + 2 * MAX_ROWS * TOPK + 2 * MAX_ROWS * TOPK * MAX_ROWS
# E2M1 nibble (sign bit 8, magnitude code 1, 3, 5, 6 = 0.5, 1.5, 3, 4) -> index into SIM3_CODEBOOK; -1: not a sim3 value
_SIM3_CODE = (-1, 4, -1, 5, -1, 6, 7, -1, -1, 3, -1, 2, -1, 1, 0, -1)


@dataclass
class W3Experts:
    planes13: torch.Tensor  # [E, 2I/32, H/128, 3, 32, 4] uint32, fc1 rows in fc1_order
    scales13: torch.Tensor  # [E, 2I/32, H/128, 4, 32] uint8 E4M3
    row13: torch.Tensor     # [E, 2I] float32
    planes2: torch.Tensor   # [E, H/32, I/128, 3, 32, 4]
    scales2: torch.Tensor
    row2: torch.Tensor      # [E, H]
    codebook: tuple[float, ...]
    lut: int

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.planes13, self.scales13, self.row13, self.planes2, self.scales2, self.row2))

    def matrix(self, e: int, fc1: bool) -> w3.W3Weight:
        """Expert e's fc1 (rows in fc1_order) or fc2 as a W3Weight (views, no copy)."""
        p, s, r = (self.planes13, self.scales13, self.row13) if fc1 else (self.planes2, self.scales2, self.row2)
        n, k = p.shape[1] * 32, p.shape[2] * w3.CHUNK
        return w3.W3Weight(p[e], s[e], r[e], n, k, self.codebook, self.lut)


def fc1_order(inter: int, device=None) -> torch.Tensor:
    """Row permutation of w13 = [up; gate] (2 * inter rows): block b of 32 rows is up 16b..16b+15, then gate 16b..16b+15."""
    b = torch.arange(inter // 16, device=device)[:, None]
    j = torch.arange(16, device=device)[None, :]
    return torch.cat([16 * b + j, inter + 16 * b + j], dim=1).reshape(-1)


def sim3_codes(packed: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A sim3 matrix in NVFP4 form (uint8 [R, K/2] low nibble first, float8_e4m3fn [R, K/16], group 32) ->
    (codes into SIM3_CODEBOOK int32 [R, K], group scales float8_e4m3fn [R, K/32])."""
    nib = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1).long()
    q = torch.tensor(_SIM3_CODE, device=packed.device, dtype=torch.int32)[nib]
    sc = scales.view(torch.uint8).reshape(scales.shape[0], -1, 2)
    if (q < 0).any() or not torch.equal(sc[..., 0], sc[..., 1]):
        raise ValueError("not the 3-bit format: expected SHARAR_FLASHNEXT_EXPERTS_SIM3=32 experts")
    return q, sc[..., 0].contiguous().view(torch.float8_e4m3fn)


def unswizzle_batched(swizzled: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """Inverse of moe.swizzle_block_scales_batched: [E, ...] CUTLASS-interleaved -> float8_e4m3fn [E, rows, groups]."""
    e = swizzled.shape[0]
    mp, kp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    s = swizzled.reshape(e, -1).view(torch.float8_e4m3fn).reshape(e, mp // 128, kp // 4, 32, 4, 4)
    return s.permute(0, 1, 4, 3, 2, 5).reshape(e, mp, kp)[:, :rows, :groups]


def _pack(packed, scales, scale2, order, chunk):
    e, r, _ = packed.shape
    parts = []
    for lo in range(0, e, chunk):
        hi = min(e, lo + chunk)
        p, s = packed[lo:hi], scales[lo:hi]
        if order is not None:
            p, s = p[:, order], s[:, order]
        q, sf = sim3_codes(p.reshape(-1, p.shape[-1]), s.reshape(-1, s.shape[-1]))
        rs = scale2[lo:hi].float().reshape(-1, 1).expand(hi - lo, r).reshape(-1).contiguous()
        m = w3.pack_codes(q, sf, rs, SIM3_CODEBOOK, rows_per_chunk=r)
        parts.append((m.planes.reshape(hi - lo, r // 32, *m.planes.shape[1:]),
                      m.scales.reshape(hi - lo, r // 32, *m.scales.shape[1:]), m.row_scale.reshape(hi - lo, r)))
        del q, sf, m
    return [torch.cat([part[i] for part in parts]) for i in range(3)]


@torch.no_grad()
def from_sim3(w13, s13, scale2_13, w2, s2, scale2_2, chunk: int = 16) -> W3Experts:
    """sim3 experts (uint8 [E, 2I, H/2], float8_e4m3fn [E, 2I, H/16], FP32 global scales [E]; w2 likewise) -> W3Experts."""
    order = fc1_order(w13.shape[1] // 2, w13.device)
    p13, sc13, r13 = _pack(w13, s13, scale2_13, order, chunk)
    p2, sc2, r2 = _pack(w2, s2, scale2_2, None, chunk)
    return W3Experts(p13, sc13, r13, p2, sc2, r2, SIM3_CODEBOOK, w3.lut_of(SIM3_CODEBOOK))


def from_moe_layer(layer, chunk: int = 16) -> W3Experts:
    """The 3-bit experts of a MoeLayer loaded with SHARAR_FLASHNEXT_EXPERTS_SIM3=32 (the layer keeps its NVFP4 copy)."""
    r13, k13 = layer.w13.shape[1], layer.w13.shape[2] * 2
    r2, k2 = layer.w2.shape[1], layer.w2.shape[2] * 2
    s13 = unswizzle_batched(layer.s13, r13, k13 // 16)
    s2 = unswizzle_batched(layer.s2, r2, k2 // 16)
    # g1 = input_scale * global scale and a13 = 1 / input_scale (moe.py): the weights' own global scales
    return from_sim3(layer.w13, s13, layer.g1 * layer.a13, layer.w2, s2, layer.g2 * layer.a2, chunk)


# ------------------------------------------------------------------ the kernel

class Workspace:
    """Per-device scratch shared by every layer (calls run in stream order); each call writes what it reads."""

    def __init__(self, device: torch.device):
        pairs = MAX_ROWS * TOPK
        self.plan = torch.zeros(_PLAN_WORDS, dtype=torch.int32, device=device)
        self.xq = torch.zeros(MAX_ROWS * HIDDEN, dtype=torch.uint8, device=device)
        self.xs = torch.zeros(MAX_ROWS * HIDDEN // 32, dtype=torch.float32, device=device)
        self.hq = torch.zeros(pairs * INTER, dtype=torch.uint8, device=device)
        self.hs = torch.zeros(pairs * INTER // 32, dtype=torch.float32, device=device)
        self.y = torch.zeros(pairs * HIDDEN, dtype=torch.float32, device=device)


_WORKSPACES: dict[torch.device, Workspace] = {}


def w3_moe(experts: W3Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Routed experts for x [M <= 16, 2560] bf16 with the router's ids [M, 10] int32 and weights [M, 10] fp32 ->
    bf16 [M, 2560], the weighted sum in selection order (no shared expert). Graph-capturable."""
    ws = _WORKSPACES.get(x.device)
    if ws is None:
        ws = _WORKSPACES[x.device] = Workspace(x.device)
    return w3._module().w3_moe(x.contiguous(), ids.to(torch.int32).contiguous(), weights.float().contiguous(),
                               experts.planes13, experts.scales13, experts.row13, experts.planes2, experts.scales2, experts.row2,
                               experts.lut, ws.plan, ws.xq, ws.xs, ws.hq, ws.hs, ws.y)


def quant_e4m3(t: torch.Tensor, stage: str) -> torch.Tensor:
    """The kernel's activation rounding, dequantized: E4M3 per (row, 32-group), float32 [M, K]. x goes through
    `w3.quant_groups` (the prep kernel runs the same code), the intermediate through its torch mirror."""
    if stage == "x":
        q, s = w3.quant_groups(t.contiguous())
        return (q.float().reshape(t.shape[0], -1, 32) * s[..., None]).reshape(t.shape)
    g = t.float().reshape(t.shape[0], -1, 32)
    s = (g.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return ((g / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).reshape(t.shape)


@torch.no_grad()
def reference(experts: W3Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, quant=quant_e4m3) -> torch.Tensor:
    """The kernel's math in torch on the dequantized 3-bit weights (FP32 matmuls): x -> quant(x, "x"), fc1 -> up, gate
    in BF16, h = bf16(silu(gate) * up) -> quant(h, "h"), fc2 in FP32, weighted sum in selection order -> BF16."""
    import torch.nn.functional as F
    xd = quant(x, "x")
    inter = experts.row13.shape[1] // 2
    inverse = torch.argsort(fc1_order(inter, x.device))
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    out = torch.zeros((x.shape[0], experts.row2.shape[1]), dtype=torch.float32, device=x.device)
    for r in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[r, k])
            if e not in cache:
                cache[e] = (w3.dequantize(experts.matrix(e, True))[inverse], w3.dequantize(experts.matrix(e, False)))
            w13, w2 = cache[e]
            h1 = (xd[r:r + 1] @ w13.T).to(torch.bfloat16).float()
            h = (F.silu(h1[:, inter:]) * h1[:, :inter]).to(torch.bfloat16)
            out[r] += float(weights[r, k]) * (quant(h, "h") @ w2.T)[0]
    return out.to(torch.bfloat16)
