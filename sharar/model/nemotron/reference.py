# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference for NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (text model and MTP head; the vision tower,
its projector and `mlp1` are never read). Every fast kernel and the engine integration are checked against this.

The model follows Transformers' built-in `nemotron_h` (the snapshot's `modeling_nemotron_h_omni.py` only wraps
`AutoModelForCausalLM.from_config(llm_config)`); the MTP head, which Transformers drops on load, follows vLLM's and
SGLang's `nemotron_h_mtp.py`. Semantics, each confirmed against that code and the checkpoint:

- **Blocks.** 88 blocks from `llm_config.layers_block_type` (40 mamba, 40 moe, 8 attention), each
  `x = x + mixer(rms_norm(x, block.norm))`; the residual stream stays in the activation dtype (`residual_in_fp32` is
  false). Then `norm_f` and an untied FP8 `lm_head`. All RMS norms are plain `w * x / rms(x)` (weights are near 0.3-1.2,
  not near 0) with eps 1e-5; this reference multiplies in FP32 and casts once.
- **Mamba-2.** `in_proj` (FP8, [18560, 4096]) splits as `z` 8192 | `xBC` 10240 | `dt` 128, and `xBC` as `x` 8192 |
  `B` 8 x 128 | `C` 8 x 128. A depthwise causal conv (kernel 4, bias, weight [10240, 1, 4] cross-correlated so tap 3
  meets the current token) and SiLU run over `xBC` only; the conv state is the last 3 raw (pre-conv) `xBC` rows.
  `dt = softplus(dt + dt_bias)`, then clamped to `time_step_limit` = (0, inf) from config.json, i.e. not clamped (vLLM
  passes (0, inf) too). Transformers 5.15's mixer instead clamps to (time_step_min = 1e-3, inf), which bites: dt_bias
  reaches -10.5 (softplus 2.7e-5). `A = -exp(A_log)`; per head h (group h // 16 for B and C):
  `S = exp(dt A) S + dt x B^T`, `y = S C + D x`, with an FP32 state [128 heads, 64, 128]. The output is gated and
  normed with `norm_before_gate=False`: `rms_norm_per_group(y * silu(z))` over 8 groups of 1024 channels, times
  `mixer.norm.weight` [8192], then `out_proj` (FP8, [4096, 8192]).
- **Attention.** 32 query heads, 2 KV heads (query head h reads KV head h // 16), head dim 128, no bias, **no
  positional encoding at all** (no RoPE; order reaches attention only through the Mamba blocks), causal softmax with
  scale 128**-0.5. q/k/v/o are FP8 per-tensor.
- **Latent MoE.** Router in FP32: `s = sigmoid(x @ gate.weight^T)`; experts are chosen by top-22 of
  `s + e_score_correction_bias` (n_group = topk_group = 1, so no group stage), weighted by the *uncorrected* `s`,
  renormalised to sum 1 (+1e-20) and multiplied by `routed_scaling_factor` 5.0. Routed path: `fc1_latent_proj`
  (BF16, 4096 -> 1024), each chosen expert `down(relu(up(h))^2)` (NVFP4, up [2688, 1024], down [1024, 2688], no gate
  projection), weighted sum in FP32, `fc2_latent_proj` (BF16, 1024 -> 4096). Shared expert: `down(relu(up(x))^2)`
  (NVFP4, [5376, 4096] / [4096, 5376]) on the full normed hidden `x`, added after `fc2_latent_proj`.
- **MTP** (`language_model.mtp.layers.{0,1}`, one step, shares `embeddings` and `lm_head` with the trunk). For
  position i it takes `h_i`, the trunk's hidden *after* `norm_f` (the tensor `lm_head` reads), and the embedding of the
  next token `t_{i+1}`: `x = eh_proj([enorm(embed(t_{i+1})), hnorm(h_i)])` (concat order e then h; eh_proj BF16
  [4096, 8192]); `x = x + attn(layers.0.norm(x))` (BF16 attention like the trunk's, with its own causal KV cache over
  MTP positions; `k_scale`/`v_scale` are FP8-KV-cache scales and unused here); `x = x + moe(layers.1.norm(x))` (an
  NVFP4 latent MoE like the trunk's); `h' = layers.1.final_layernorm(x)`; `logits = lm_head(h')` predict t_{i+2}.
  Chained drafts feed `h'` back as the next step's hidden with the drafted token (vLLM's recursion); see
  `NemotronReference.mtp_chain`.
- **Quantisation** (ModelOpt 0.37, `MIXED_PRECISION`). FP8: E4M3 `weight` times the per-tensor F32 `weight_scale`.
  NVFP4: `weight` U8 [N, K/2] holds two E2M1 codes per byte, element 2j in the low nibble and 2j+1 in the high one;
  `weight_scale` E4M3 [N, K/16] scales each run of 16 along K, `weight_scale_2` F32 is global:
  `w = e2m1 * weight_scale * weight_scale_2`. `input_scale` (static activation scales) is ignored: activations are not
  quantised here. Embeddings, router, latent projections, `eh_proj` and MTP attention are BF16; A_log, D, dt_bias,
  conv and norm weights are BF16 too (used in FP32), `e_score_correction_bias` F32.

Precision: weights stay on the GPU in their checkpoint encoding; routed experts are read from the shards on demand
(positioned reads, about 0.25 ms an expert from the page cache) into a bounded LRU of packed experts, so the model
runs in about 8 GiB plus that cache. A quantised linear is computed as `(x @ Wq^T) * s`: `Wq` is the E4M3 weight, or
E2M1 times its E4M3 block scale, both exact in BF16 (at most 6 significant bits), and the per-tensor `s`
(`weight_scale` / `weight_scale_2`) multiplies the FP32-accumulated output before its one rounding to `dtype`. So the
weights are the checkpoint's exactly and no step rounds a dequantised weight. Linears run in `dtype` (BF16 by default,
as Transformers runs the model; FP32 for the tests' parity checks); norms, conv, the SSM, softmax, routing and the
expert sum run in FP32, with results cast to `dtype` where Transformers casts them.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from sharar.weights.safetensors import SnapshotWeights

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_REVISION = "6aae0d6d0285e561bbb72efa10ba1425d41ebddc"
MODEL_CACHE_KEY = "models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
TRUNK = "language_model.backbone"
MTP = "language_model.mtp.layers"


def find_snapshot(cache_dir: str | os.PathLike | None = None) -> Path:
    """The pinned snapshot under `cache_dir`, the engine's model cache (SHARAR_HF_CACHE, else HF_HOME) or
    ~/.cache/huggingface/hub."""
    from sharar.engine.config import current
    roots = [cache_dir, current().hf_cache, Path.home() / ".cache/huggingface/hub"]
    for root in filter(None, roots):
        for base in (Path(root), Path(root) / "hub"):
            snap = base / MODEL_CACHE_KEY / "snapshots" / MODEL_REVISION
            if (snap / "config.json").is_file() and (snap / "model.safetensors.index.json").is_file():
                return snap
    raise FileNotFoundError(f"snapshot {MODEL_ID}@{MODEL_REVISION} not found")


def read_json(path: Path) -> dict:
    """json.load that decodes the `{"__float__": "Infinity"}` encoding config.json uses for time_step_limit."""
    def hook(d):
        return float(d["__float__"]) if set(d) == {"__float__"} else d
    return json.loads(Path(path).read_text(), object_hook=hook)


def render_chat(snap: Path, messages: list[dict], enable_thinking: bool = False) -> str:
    """The snapshot's chat template with the generation prompt (thinking off: the turn opens with
    `<think></think>`)."""
    import jinja2
    template = jinja2.Environment(loader=jinja2.BaseLoader()).from_string((snap / "chat_template.jinja").read_text())
    return template.render(messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking)


@dataclass(frozen=True)
class NemotronConfig:
    hidden: int
    vocab: int
    eps: float
    block_types: tuple[str, ...]  # "mamba" | "moe" | "attention" per trunk block
    mtp_block_types: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    mamba_heads: int
    mamba_head_dim: int
    n_groups: int
    d_state: int
    conv_kernel: int
    chunk: int
    time_step_limit: tuple[float, float]
    experts: int
    top_k: int
    moe_inter: int
    latent: int
    shared_inter: int
    routed_scale: float
    norm_topk: bool
    n_group: int
    topk_group: int
    eos: tuple[int, ...] = (2, 11)

    @classmethod
    def from_snapshot(cls, snap: Path) -> "NemotronConfig":
        c = read_json(snap / "config.json")["llm_config"]
        legacy = {"mamba": "mamba", "moe": "moe", "attention": "attention", "linear_attention": "mamba",
                  "full_attention": "attention"}
        if c["mlp_hidden_act"] != "relu2" or c["mamba_hidden_act"] != "silu" or c.get("n_shared_experts") != 1:
            raise ValueError("unexpected activation or shared-expert count")
        if c.get("num_nextn_predict_layers", 0) != 1 or c.get("tie_word_embeddings"):
            raise ValueError("expected one untied MTP step")
        return cls(
            hidden=c["hidden_size"], vocab=c["vocab_size"], eps=c["layer_norm_epsilon"],
            block_types=tuple(legacy[t] for t in c["layers_block_type"]),
            mtp_block_types=tuple(legacy[t] for t in c["mtp_layers_block_type"]),
            heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
            mamba_heads=c["mamba_num_heads"], mamba_head_dim=c["mamba_head_dim"], n_groups=c["n_groups"],
            d_state=c["ssm_state_size"], conv_kernel=c["conv_kernel"], chunk=c["chunk_size"],
            time_step_limit=tuple(float(v) for v in c["time_step_limit"]),
            experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"], moe_inter=c["moe_intermediate_size"],
            latent=c["moe_latent_size"], shared_inter=c["moe_shared_expert_intermediate_size"],
            routed_scale=float(c["routed_scaling_factor"]), norm_topk=bool(c["norm_topk_prob"]),
            n_group=c["n_group"], topk_group=c["topk_group"],
        )

    @property
    def mamba_inner(self) -> int:
        return self.mamba_heads * self.mamba_head_dim  # 8192

    @property
    def conv_dim(self) -> int:
        return self.mamba_inner + 2 * self.n_groups * self.d_state  # 10240


# ---------------------------------------------------------------------------------------------------------------------
# Dequantisation

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_PAIR_LUT: dict[tuple, torch.Tensor] = {}
EXPERT_GROUP = 32  # routed experts dequantised together (transient ~0.7 GiB in BF16)


def _pair_lut(device, low_first: bool) -> torch.Tensor:
    """[256, 2] FP32: the two E2M1 values a byte holds, in element order."""
    key = (str(device), low_first)
    if key not in _PAIR_LUT:
        codes = torch.arange(256)
        lo, hi = codes & 15, codes >> 4
        vals = torch.tensor(E2M1)
        pair = torch.stack((vals[lo], vals[hi]) if low_first else (vals[hi], vals[lo]), -1)
        _PAIR_LUT[key] = pair.to(device)
    return _PAIR_LUT[key]


def dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor,
                  low_nibble_first: bool = True) -> torch.Tensor:
    """ModelOpt NVFP4 -> FP32 [N, K]: weight U8 [N, K/2], weight_scale E4M3 [N, K/16], weight_scale_2 F32 scalar."""
    n, half = weight.shape
    if weight_scale.shape != (n, half // 8):
        raise ValueError(f"block scales {tuple(weight_scale.shape)} do not fit packed weight {tuple(weight.shape)}")
    values = F.embedding(weight.to(torch.int32), _pair_lut(weight.device, low_nibble_first)).view(n, 2 * half)
    scales = weight_scale.float() * weight_scale_2.float().reshape(())
    return (values.view(n, half // 8, 16) * scales[..., None]).view(n, 2 * half)


def nvfp4_unscaled(weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype,
                   low_nibble_first: bool = True) -> torch.Tensor:
    """E2M1 values times their E4M3 block scales, without weight_scale_2, computed in `dtype`: exact in BF16 (at
    most 6 significant bits, magnitudes 2^-10 .. 2688). Any leading dims of weight [..., N, K/2] are kept."""
    *lead, n, half = weight.shape
    lut = _pair_lut(weight.device, low_nibble_first).to(dtype)
    values = F.embedding(weight.reshape(-1, half).to(torch.int32), lut).view(*lead, n, half // 8, 16)
    return (values * weight_scale.to(dtype)[..., None]).view(*lead, n, 2 * half)


def dequant_nvfp4_many(lins: list["Linear"], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Several same-shape NVFP4 linears at once: (`nvfp4_unscaled` weights [n, N, K] in `dtype`, their
    weight_scale_2 [n] FP32)."""
    w = torch.stack([lin.w for lin in lins])
    scales = torch.stack([lin.scale for lin in lins])
    return (nvfp4_unscaled(w, scales, dtype, lins[0].low_nibble_first),
            torch.stack([lin.scale_2.float().reshape(()) for lin in lins]))


def dequant_fp8(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """ModelOpt per-tensor FP8 -> FP32."""
    return weight.float() * weight_scale.float().reshape(())


@dataclass
class Linear:
    """One bias-free linear in its checkpoint encoding: "bf16" (w), "fp8" (w, scale) or "nvfp4" (w, scale, scale_2)."""
    kind: str
    w: torch.Tensor
    scale: torch.Tensor | None = None
    scale_2: torch.Tensor | None = None
    low_nibble_first: bool = True

    def weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The dequantised weight, [N, K]."""
        if self.kind == "bf16":
            return self.w.to(dtype)
        if self.kind == "fp8":
            return dequant_fp8(self.w, self.scale).to(dtype)
        return dequant_nvfp4(self.w, self.scale, self.scale_2, self.low_nibble_first).to(dtype)

    def unscaled(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(weight without its per-tensor scale, that scale): exact in BF16, since an E4M3 value, and an E2M1 value
        times an E4M3 block scale, have at most 6 significant bits."""
        if self.kind == "bf16":
            return self.w.to(dtype), None
        if self.kind == "fp8":
            return self.w.to(dtype), self.scale
        return nvfp4_unscaled(self.w, self.scale, dtype, self.low_nibble_first), self.scale_2

    def __call__(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """x @ W^T: one matmul in `dtype` (default x's) with FP32 accumulation and output, the per-tensor scale
        applied to that FP32 output, one rounding to `dtype`."""
        dtype = dtype or x.dtype
        w, scale = self.unscaled(dtype)
        return scaled_mm(x, w, scale, dtype)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.scale, self.scale_2) if t is not None)


def scaled_mm(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype,
              out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """(x @ w^T) * scale for x [..., K], w [N, K] in `dtype`, FP32 accumulated and scaled, rounded once to
    `out_dtype` (default `dtype`)."""
    y = torch.mm(x.reshape(-1, x.shape[-1]).to(dtype), w.t(), out_dtype=torch.float32)
    if scale is not None:
        y = y * scale.float().reshape(())
    return y.to(out_dtype or dtype).view(*x.shape[:-1], w.shape[0])


def load_linear(w: SnapshotWeights, prefix: str, device, low_nibble_first: bool = True) -> Linear:
    """The linear at `prefix` in whatever encoding the checkpoint stored it."""
    weight = w.get(f"{prefix}.weight").to(device)
    if weight.dtype == torch.uint8:
        return Linear("nvfp4", weight, w.get(f"{prefix}.weight_scale").to(device),
                      w.get(f"{prefix}.weight_scale_2").float().to(device), low_nibble_first)
    if weight.dtype == torch.float8_e4m3fn:
        return Linear("fp8", weight, w.get(f"{prefix}.weight_scale").float().to(device))
    return Linear("bf16", weight)


# ---------------------------------------------------------------------------------------------------------------------
# Functional pieces (FP32 inside, inputs and outputs in the caller's dtype unless noted)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def gated_rms_norm(y: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """Mamba-2's RMSNormGated with norm_before_gate=False: RMS-normalise `y * silu(z)` per group of `group_size`
    channels, then scale. FP32 out."""
    g = y.float() * F.silu(z.float())
    shape = g.shape
    g = g.view(*shape[:-1], shape[-1] // group_size, group_size)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    return g.view(shape) * weight.float()


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x.float()).square().to(x.dtype)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                  state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv + SiLU over time. x [T, C]; weight [C, K] (the checkpoint's [C, 1, K] squeezed);
    state [K-1, C], the previous K-1 raw inputs (zeros at the start of a sequence). Returns (FP32 [T, C], new state:
    the last K-1 raw inputs, in x's dtype)."""
    k = weight.shape[-1]
    if state is None:
        state = x.new_zeros(k - 1, x.shape[1])
    full = torch.cat([state.to(x.dtype), x], 0)
    out = F.conv1d(full.float().t()[None], weight.float()[:, None, :], None if bias is None else bias.float(),
                   groups=x.shape[1])[0].t()
    return F.silu(out), full[-(k - 1):].clone()


def _segsum(a: torch.Tensor) -> torch.Tensor:
    """a [..., L] -> [..., L, L]: out[i, j] = a[j+1] + ... + a[i] for i >= j, -inf above the diagonal."""
    n = a.shape[-1]
    x = a[..., None].expand(*a.shape, n)
    x = x.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool, device=a.device), -1), 0)
    x = x.cumsum(-2)
    return x.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool, device=a.device)), -math.inf)


def ssd_sequential(x, dt, A, B, C, D, state=None):
    """The Mamba-2 recurrence one token at a time, all FP32. x [T, H, P]; dt [T, H] (after softplus and limits);
    A [H] (negative); B, C [T, G, N]; D [H]; state [H, P, N] or None. Returns (y [T, H, P], final state)."""
    t_len, h, p = x.shape
    rep = h // B.shape[1]
    Bh, Ch = B.repeat_interleave(rep, 1), C.repeat_interleave(rep, 1)
    s = x.new_zeros(h, p, B.shape[-1]) if state is None else state.clone().float()
    ys = []
    for t in range(t_len):
        s = s * torch.exp(dt[t] * A)[:, None, None] + (dt[t, :, None] * x[t])[..., None] * Bh[t][:, None, :]
        ys.append(torch.einsum("hpn,hn->hp", s, Ch[t]) + D[:, None] * x[t])
    return torch.stack(ys), s


def ssd_chunked(x, dt, A, B, C, D, chunk: int, state=None):
    """The same recurrence by the chunked SSD algorithm (mamba_ssm's ssd_minimal, with an initial state), FP32.
    Positions are padded to a multiple of `chunk` with dt = 0, which leaves the final state untouched."""
    t_len, h, p = x.shape
    n = B.shape[-1]
    rep = h // B.shape[1]
    pad = (-t_len) % chunk
    if pad:
        x, dt, B, C = (F.pad(v, (0, 0) * (v.dim() - 1) + (0, pad)) for v in (x, dt, B, C))
    c = x.shape[0] // chunk
    xdt = (x * dt[..., None]).view(c, chunk, h, p)
    dA = (dt * A).view(c, chunk, h).permute(0, 2, 1)  # [c, h, l]
    Bh = B.repeat_interleave(rep, 1).view(c, chunk, h, n)
    Ch = C.repeat_interleave(rep, 1).view(c, chunk, h, n)
    acs = dA.cumsum(-1)
    # within each chunk
    decay = torch.exp(_segsum(dA))  # [c, h, l, s]
    cb = torch.einsum("clhn,cshn->chls", Ch, Bh)
    y = torch.einsum("chls,cshp->clhp", cb * decay, xdt)
    # each chunk's own contribution to the state at its end
    to_end = torch.exp(acs[..., -1:] - acs)  # [c, h, l]
    chunk_states = torch.einsum("clhn,chl,clhp->chpn", Bh, to_end, xdt)
    # states entering each chunk, chunk by chunk
    s = x.new_zeros(h, p, n) if state is None else state.clone().float()
    entering = []
    for i in range(c):
        entering.append(s)
        s = s * torch.exp(acs[i, :, -1])[:, None, None] + chunk_states[i]
    entering = torch.stack(entering)
    y = y + torch.einsum("clhn,chpn,chl->clhp", Ch, entering, torch.exp(acs))
    y = y.reshape(c * chunk, h, p)[:t_len] + D[:, None] * x[:t_len]
    return y, s


# ---------------------------------------------------------------------------------------------------------------------
# Blocks

@dataclass
class MambaParams:
    in_proj: Linear
    conv_w: torch.Tensor  # [conv_dim, K]
    conv_b: torch.Tensor
    A_log: torch.Tensor
    D: torch.Tensor
    dt_bias: torch.Tensor
    norm_w: torch.Tensor  # gated norm [inner]
    out_proj: Linear


@dataclass
class AttentionParams:
    q: Linear
    k: Linear
    v: Linear
    o: Linear


@dataclass
class MoEParams:
    prefix: str  # "<...>.mixer", where the routed experts live
    gate_w: torch.Tensor  # [E, hidden] BF16
    bias: torch.Tensor  # [E] F32
    fc1: Linear
    fc2: Linear
    shared_up: Linear
    shared_down: Linear


@dataclass
class Block:
    kind: str
    norm: torch.Tensor
    mixer: MambaParams | AttentionParams | MoEParams


def load_block(w: SnapshotWeights, prefix: str, kind: str, device, low_nibble_first: bool = True) -> Block:
    """Block at `prefix` (e.g. "language_model.backbone.layers.7") of `kind` "mamba" | "attention" | "moe"."""
    m = f"{prefix}.mixer"
    lin = lambda name: load_linear(w, f"{m}.{name}", device, low_nibble_first)  # noqa: E731
    get = lambda name: w.get(f"{m}.{name}").to(device)  # noqa: E731
    if kind == "mamba":
        mixer = MambaParams(lin("in_proj"), get("conv1d.weight").squeeze(1), get("conv1d.bias"), get("A_log"),
                            get("D"), get("dt_bias"), get("norm.weight"), lin("out_proj"))
    elif kind == "attention":
        mixer = AttentionParams(lin("q_proj"), lin("k_proj"), lin("v_proj"), lin("o_proj"))
    elif kind == "moe":
        mixer = MoEParams(m, get("gate.weight"), get("gate.e_score_correction_bias").float(), lin("fc1_latent_proj"),
                          lin("fc2_latent_proj"), lin("shared_experts.up_proj"), lin("shared_experts.down_proj"))
    else:
        raise ValueError(kind)
    return Block(kind, w.get(f"{prefix}.norm.weight").to(device), mixer)


class _Shard:
    """One safetensors file opened for positioned reads: name -> (dtype, shape, first byte, end byte)."""

    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_RDONLY)
        n = int.from_bytes(os.pread(self.fd, 8, 0), "little")
        header = json.loads(os.pread(self.fd, n, 8))
        header.pop("__metadata__", None)
        dtypes = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}
        self.entries = {k: (dtypes[v["dtype"]], tuple(v["shape"]), 8 + n + v["data_offsets"][0],
                            8 + n + v["data_offsets"][1]) for k, v in header.items()}


class ExpertStore:
    """Routed experts, read from the shards on demand (positioned reads into pinned memory, one device copy per
    expert) and kept packed on the GPU in an LRU of `budget` bytes."""

    PARTS = ("up_proj.weight", "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale")

    def __init__(self, w: SnapshotWeights, device, budget: int, low_nibble_first: bool = True):
        self.w, self.device, self.budget, self.low = w, device, budget, low_nibble_first
        self.cache: OrderedDict[tuple[str, int], tuple[Linear, Linear]] = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self._shards: dict[str, _Shard] = {}
        self._staging = torch.empty(0, dtype=torch.uint8)

    def _read(self, prefix: str, e: int) -> tuple[Linear, Linear]:
        names = [f"{prefix}.experts.{e}.{p}" for p in self.PARTS]
        file = self.w._map[names[0]]
        shard = self._shards.get(file) or self._shards.setdefault(file, _Shard(self.w.snap / file))
        entries = [shard.entries[n] for n in names]
        total = sum(end - start for _, _, start, end in entries)
        if self._staging.numel() < total:
            self._staging = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        view, at = memoryview(self._staging.numpy()), 0
        for _, _, start, end in entries:
            if os.preadv(shard.fd, [view[at:at + end - start]], start) != end - start:
                raise EOFError(f"short read in {file}")
            at += end - start
        raw = self._staging[:total].to(self.device)
        tensors, at = [], 0
        for dtype, shape, start, end in entries:
            tensors.append(raw[at:at + end - start].view(dtype).view(shape))
            at += end - start
        s2 = [self.w.get(f"{prefix}.experts.{e}.{p}.weight_scale_2").float().to(self.device)
              for p in ("up_proj", "down_proj")]
        return (Linear("nvfp4", tensors[0], tensors[1], s2[0], self.low),
                Linear("nvfp4", tensors[2], tensors[3], s2[1], self.low))

    def get(self, prefix: str, e: int) -> tuple[Linear, Linear]:
        key = (prefix, e)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            self.hits += 1
            return hit
        self.misses += 1
        pair = self._read(prefix, e)
        size = pair[0].nbytes + pair[1].nbytes
        while self.cache and self.bytes + size > self.budget:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old[0].nbytes + old[1].nbytes
        if size <= self.budget:
            self.cache[key] = pair
            self.bytes += size
        return pair

    def clear(self) -> None:
        self.cache.clear()
        self.bytes = 0


def mamba_mixer(p: MambaParams, x: torch.Tensor, cfg: NemotronConfig, conv_state=None, ssm_state=None,
                chunked: bool | None = None, time_step_limit: tuple[float, float] | None = None):
    """x [T, hidden] (normed) -> (out [T, hidden], conv_state [K-1, conv_dim], ssm_state FP32 [H, P, N])."""
    dtype, t_len = x.dtype, x.shape[0]
    zxbcdt = p.in_proj(x)
    z, xbc, dt = zxbcdt.split([cfg.mamba_inner, cfg.conv_dim, cfg.mamba_heads], -1)
    xbc, conv_state = causal_conv1d(xbc, p.conv_w, p.conv_b, conv_state)
    xbc = xbc.to(dtype)
    xs, B, C = xbc.split([cfg.mamba_inner, cfg.n_groups * cfg.d_state, cfg.n_groups * cfg.d_state], -1)
    lo, hi = time_step_limit or cfg.time_step_limit
    dt = F.softplus(dt.float() + p.dt_bias.float())
    if (lo, hi) != (0.0, math.inf):
        dt = dt.clamp(lo, hi)
    A = -torch.exp(p.A_log.float())
    args = (xs.float().view(t_len, cfg.mamba_heads, cfg.mamba_head_dim), dt, A,
            B.float().view(t_len, cfg.n_groups, cfg.d_state), C.float().view(t_len, cfg.n_groups, cfg.d_state),
            p.D.float())
    if chunked is None:
        chunked = t_len > 1
    y, ssm_state = ssd_chunked(*args, cfg.chunk, ssm_state) if chunked else ssd_sequential(*args, ssm_state)
    y = gated_rms_norm(y.reshape(t_len, cfg.mamba_inner), z, p.norm_w, cfg.mamba_inner // cfg.n_groups, cfg.eps)
    return p.out_proj(y.to(dtype)), conv_state, ssm_state


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past: int) -> torch.Tensor:
    """Causal GQA softmax attention in FP32. q [T, Hq, D] for positions past..past+T-1; k, v [past+T, Hkv, D].
    Returns [T, Hq, D] FP32."""
    t_len, hq, d = q.shape
    rep = hq // k.shape[1]
    kf, vf = k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1)
    scores = torch.einsum("thd,shd->hts", q.float(), kf) * d ** -0.5
    rows = torch.arange(t_len, device=q.device)[:, None] + past
    cols = torch.arange(k.shape[0], device=q.device)[None]
    scores = scores.masked_fill(cols > rows, -math.inf)
    return torch.einsum("hts,shd->thd", scores.softmax(-1), vf)


def attention_mixer(p: AttentionParams, x: torch.Tensor, cfg: NemotronConfig, kv=None):
    """x [T, hidden] (normed); kv = (k, v) of earlier positions or None. Returns (out, (k, v) including x's)."""
    t_len, dtype = x.shape[0], x.dtype
    q = p.q(x).view(t_len, cfg.heads, cfg.head_dim)
    k = p.k(x).view(t_len, cfg.kv_heads, cfg.head_dim)
    v = p.v(x).view(t_len, cfg.kv_heads, cfg.head_dim)
    past = 0
    if kv is not None:
        past = kv[0].shape[0]
        k, v = torch.cat([kv[0], k]), torch.cat([kv[1], v])
    out = attend(q, k, v, past).to(dtype).reshape(t_len, cfg.heads * cfg.head_dim)
    return p.o(out), (k, v)


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, cfg: NemotronConfig):
    """Sigmoid top-k routing with the score-correction bias used for choice only. Returns (weights FP32 [T, k]
    including routed_scaling_factor, expert ids [T, k], the FP32 router logits)."""
    logits = x.float() @ gate_w.float().t()
    scores = logits.sigmoid()
    choice = scores + bias.float()
    if cfg.n_group > 1:
        per = cfg.experts // cfg.n_group
        group = choice.view(-1, cfg.n_group, per).topk(2, -1).values.sum(-1)
        keep = torch.zeros_like(group).scatter_(1, group.topk(cfg.topk_group, -1).indices, 1)
        choice = choice.masked_fill(~keep.bool().repeat_interleave(per, 1), -math.inf)
    ids = choice.topk(cfg.top_k, -1).indices
    weights = scores.gather(1, ids)
    if cfg.norm_topk:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * cfg.routed_scale, ids, logits


def moe_mixer(p: MoEParams, x: torch.Tensor, cfg: NemotronConfig, experts: ExpertStore,
              return_routing: bool = False):
    """x [T, hidden] (normed) -> [T, hidden]: latent routed experts plus the shared expert."""
    dtype = x.dtype
    weights, ids, _ = route(x, p.gate_w, p.bias, cfg)
    latent = p.fc1(x)
    acc = torch.zeros(x.shape[0], cfg.latent, dtype=torch.float32, device=x.device)
    chosen = ids.unique().tolist()
    for g in range(0, len(chosen), EXPERT_GROUP):
        group = chosen[g:g + EXPERT_GROUP]
        pairs = [experts.get(p.prefix, e) for e in group]
        ups, up_s = dequant_nvfp4_many([u for u, _ in pairs], dtype)
        downs, down_s = dequant_nvfp4_many([d for _, d in pairs], dtype)
        for j, e in enumerate(group):
            tok, slot = torch.where(ids == e)
            h = relu2(scaled_mm(latent[tok], ups[j], up_s[j], dtype))
            acc.index_add_(0, tok, scaled_mm(h, downs[j], down_s[j], dtype, torch.float32) * weights[tok, slot, None])
    out = p.fc2(acc.to(dtype)) + p.shared_down(relu2(p.shared_up(x)))
    return (out, (weights, ids)) if return_routing else out


# ---------------------------------------------------------------------------------------------------------------------
# Model

@dataclass
class State:
    """Per-sequence decode state: Mamba conv/SSM states and attention KV per trunk block, MTP attention KV."""
    conv: dict[int, torch.Tensor] = field(default_factory=dict)
    ssm: dict[int, torch.Tensor] = field(default_factory=dict)
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    mtp_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    length: int = 0


@dataclass
class MtpParams:
    enorm: torch.Tensor
    hnorm: torch.Tensor
    eh_proj: Linear
    attn: Block  # layers.0: norm + attention
    moe: Block  # layers.1: norm + MoE
    final_norm: torch.Tensor  # layers.1.final_layernorm


class NemotronReference:
    """The text model and MTP head, weight-dequantised. `blocks` limits which trunk blocks load (for per-block tests);
    `head=False` skips embeddings, norm_f, lm_head and MTP."""

    def __init__(self, snap: Path | None = None, device="cuda", dtype: torch.dtype = torch.bfloat16,
                 expert_cache_gib: float = 6.0, blocks=None, head: bool = True, mtp: bool = True,
                 low_nibble_first: bool = True, time_step_limit: tuple[float, float] | None = None, log=None):
        self.snap = Path(snap) if snap is not None else find_snapshot()
        self.cfg = NemotronConfig.from_snapshot(self.snap)
        self.device, self.dtype = torch.device(device), dtype
        self.time_step_limit = time_step_limit
        self.w = SnapshotWeights(self.snap)
        self.experts = ExpertStore(self.w, self.device, int(expert_cache_gib * 2**30), low_nibble_first)
        wanted = range(len(self.cfg.block_types)) if blocks is None else blocks
        self.blocks: dict[int, Block] = {}
        for i in wanted:
            self.blocks[i] = load_block(self.w, f"{TRUNK}.layers.{i}", self.cfg.block_types[i], self.device,
                                        low_nibble_first)
            if log and i % 8 == 7:
                log(f"loaded block {i}")
        self.embed = self.norm_f = self.lm_head = self.mtp = None
        if head:
            self.embed = self.w.get(f"{TRUNK}.embeddings.weight").to(self.device)
            self.norm_f = self.w.get(f"{TRUNK}.norm_f.weight").to(self.device)
            self.lm_head = load_linear(self.w, "language_model.lm_head", self.device)
            if mtp:
                if self.cfg.mtp_block_types != ("attention", "moe"):
                    raise ValueError(f"unexpected MTP pattern {self.cfg.mtp_block_types}")
                get = lambda name: self.w.get(f"{MTP}.{name}").to(self.device)  # noqa: E731
                self.mtp = MtpParams(
                    get("0.enorm.weight"), get("0.hnorm.weight"),
                    load_linear(self.w, f"{MTP}.0.eh_proj", self.device),
                    load_block(self.w, f"{MTP}.0", "attention", self.device, low_nibble_first),
                    load_block(self.w, f"{MTP}.1", "moe", self.device, low_nibble_first),
                    get("1.final_layernorm.weight"))

    # -- trunk --------------------------------------------------------------------------------------------------------

    def run_block(self, i: int, x: torch.Tensor, state: State | None = None, chunked: bool | None = None):
        """x + mixer(norm(x)) for trunk block i, reading and updating `state` (None: a fresh sequence, unkept)."""
        b = self.blocks[i]
        h = rms_norm(x, b.norm, self.cfg.eps)
        if b.kind == "mamba":
            out, conv, ssm = mamba_mixer(b.mixer, h, self.cfg, state and state.conv.get(i), state and state.ssm.get(i),
                                         chunked, self.time_step_limit)
            if state is not None:
                state.conv[i], state.ssm[i] = conv, ssm
        elif b.kind == "attention":
            out, kv = attention_mixer(b.mixer, h, self.cfg, state and state.kv.get(i))
            if state is not None:
                state.kv[i] = kv
        else:
            out = moe_mixer(b.mixer, h, self.cfg, self.experts)
        return x + out.to(x.dtype)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed[ids.to(self.device)].to(self.dtype)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """lm_head on post-norm hidden states; FP32 logits."""
        return self.lm_head(h.to(self.dtype)).float()

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, state: State | None = None, capture=(), logits: str | int | None = "all",
                chunked: bool | None = None) -> dict:
        """Run token ids [T] (continuing `state` if given, which is updated in place). Returns
        {"hidden": {i: [T, hidden] residual stream after block i for i in capture}, "pre_norm": after the last block,
        "final": after norm_f, "logits": FP32 for all positions ("all"), the last n (int) or none (None)}."""
        ids = ids.reshape(-1).to(self.device)
        x = self.embed_ids(ids)
        captured = {}
        for i in range(len(self.cfg.block_types)):
            x = self.run_block(i, x, state, chunked)
            if i in capture:
                captured[i] = x
        if state is not None:
            state.length += ids.numel()
        final = rms_norm(x, self.norm_f, self.cfg.eps)
        out = {"hidden": captured, "pre_norm": x, "final": final}
        if logits == "all":
            out["logits"] = self.logits(final)
        elif isinstance(logits, int):
            out["logits"] = self.logits(final[-logits:])
        return out

    @torch.inference_mode()
    def generate(self, prompt: torch.Tensor, max_new: int, stop=None, state: State | None = None) -> dict:
        """Greedy decode: prefill `prompt`, then one token per step through the carried state. Returns
        {"ids": generated ids, "final": [n, hidden] post-norm hidden of each position that produced a token
        (the last prompt position first), "state"}."""
        stop = set(self.cfg.eos if stop is None else stop)
        state = state or State()
        out = self.forward(prompt, state, logits=1)
        ids, finals = [], [out["final"][-1:]]
        nxt = int(out["logits"][-1].argmax())
        for _ in range(max_new):
            ids.append(nxt)
            if nxt in stop or len(ids) == max_new:
                break
            out = self.forward(torch.tensor([nxt]), state, logits=1)
            finals.append(out["final"])
            nxt = int(out["logits"][-1].argmax())
        return {"ids": ids, "final": torch.cat(finals), "state": state}

    # -- MTP ----------------------------------------------------------------------------------------------------------

    def mtp_input(self, hidden: torch.Tensor, next_ids: torch.Tensor, order: str = "eh") -> torch.Tensor:
        """eh_proj([enorm(embed(next)), hnorm(hidden)]) ("eh"; "he" swaps the halves, for the ablation)."""
        m = self.mtp
        e = rms_norm(self.embed_ids(next_ids), m.enorm, self.cfg.eps)
        h = rms_norm(hidden.to(self.dtype), m.hnorm, self.cfg.eps)
        return m.eh_proj(torch.cat([e, h] if order == "eh" else [h, e], -1))

    def mtp_tail(self, x: torch.Tensor, attn_out: torch.Tensor):
        """From the attention mixer's output on: residual, MoE block, final_layernorm. Returns (pre, post)."""
        m = self.mtp
        x = x + attn_out.to(x.dtype)
        x = x + moe_mixer(m.moe.mixer, rms_norm(x, m.moe.norm, self.cfg.eps), self.cfg, self.experts).to(x.dtype)
        return x, rms_norm(x, m.final_norm, self.cfg.eps)

    @torch.inference_mode()
    def mtp_forward(self, hidden: torch.Tensor, next_ids: torch.Tensor, state: State | None = None,
                    order: str = "eh") -> dict:
        """One MTP step over positions i (hidden [T, hidden] = trunk post-norm_f h_i, next_ids [T] = t_{i+1}),
        attending causally to earlier MTP positions in `state.mtp_kv` (updated). Returns {"pre": before
        final_layernorm, "final": after it, "logits": FP32 predictions of t_{i+2}}."""
        m = self.mtp
        x = self.mtp_input(hidden, next_ids.to(self.device), order)
        attn_out, kv = attention_mixer(m.attn.mixer, rms_norm(x, m.attn.norm, self.cfg.eps), self.cfg,
                                       state and state.mtp_kv)
        if state is not None:
            state.mtp_kv = kv
        pre, post = self.mtp_tail(x, attn_out)
        return {"pre": pre, "final": post, "logits": self.logits(post)}

    @torch.inference_mode()
    def mtp_chain(self, hidden: torch.Tensor, ids: torch.Tensor, depth: int, feed: str = "final",
                  order: str = "eh", teacher: bool = False) -> list[torch.Tensor]:
        """Chained MTP drafts from every position of one sequence at once. hidden [T, hidden] is the trunk's post-norm
        h_i for positions 0..T-1 of `ids` [T+1] (ids[i+1] is the token after position i). Step 1 is `mtp_forward`
        over the whole sequence (its KV are the MTP cache a draft round starts from). Step d > 1 drafts, for each
        base position i, the token at i+d+1 from (step d-1's `feed` output at i: "final" after final_layernorm as vLLM
        feeds it, or "pre"; the token step d-1 drafted, or with `teacher` the true one); its query sees the step-1 KV
        of positions <= i plus its own chain's earlier steps, as an MTP KV cache does during one draft round.
        Returns the FP32 logits of each step, [T, vocab] each."""
        m, cfg = self.mtp, self.cfg
        ids = ids.to(self.device)
        x = self.mtp_input(hidden, ids[1:], order)
        hn = rms_norm(x, m.attn.norm, cfg.eps)
        t_len = x.shape[0]
        q = m.attn.mixer.q(hn).view(t_len, cfg.heads, cfg.head_dim)
        k0 = m.attn.mixer.k(hn).view(t_len, cfg.kv_heads, cfg.head_dim)
        v0 = m.attn.mixer.v(hn).view(t_len, cfg.kv_heads, cfg.head_dim)
        a = attend(q, k0, v0, 0).to(self.dtype).reshape(t_len, -1)
        pre, post = self.mtp_tail(x, m.attn.mixer.o(a))
        out = [self.logits(post)]
        chain_k, chain_v = [], []
        rows = torch.arange(t_len, device=self.device)
        for d in range(2, depth + 1):
            prev = post if feed == "final" else pre
            if teacher:
                tok = ids[(rows + d).clamp(max=ids.numel() - 1)]
            else:
                tok = out[-1].argmax(-1)
            # chained tokens of later base positions run past the sequence; their rows are garbage and ignored
            x = self.mtp_input(prev, tok, order)
            hn = rms_norm(x, m.attn.norm, cfg.eps)
            q = m.attn.mixer.q(hn).view(t_len, cfg.heads, cfg.head_dim)
            chain_k.append(m.attn.mixer.k(hn).view(t_len, cfg.kv_heads, cfg.head_dim))
            chain_v.append(m.attn.mixer.v(hn).view(t_len, cfg.kv_heads, cfg.head_dim))
            rep = cfg.heads // cfg.kv_heads
            kf0, vf0 = k0.float().repeat_interleave(rep, 1), v0.float().repeat_interleave(rep, 1)
            kc = torch.stack(chain_k, 1).float().repeat_interleave(rep, 2)  # [T, d-1, H, D]
            vc = torch.stack(chain_v, 1).float().repeat_interleave(rep, 2)
            s0 = torch.einsum("thd,shd->hts", q.float(), kf0) * cfg.head_dim ** -0.5
            s0 = s0.masked_fill(rows[None, None, :] > rows[None, :, None], -math.inf)
            sc = torch.einsum("thd,tjhd->htj", q.float(), kc) * cfg.head_dim ** -0.5
            probs = torch.cat([s0, sc], -1).softmax(-1)
            a = torch.einsum("hts,shd->thd", probs[..., :t_len], vf0) + \
                torch.einsum("htj,tjhd->thd", probs[..., t_len:], vc)
            pre, post = self.mtp_tail(x, m.attn.mixer.o(a.to(self.dtype).reshape(t_len, -1)))
            out.append(self.logits(post))
        return out
