"""Differentiable DFlash2 forward over the serving draft's weights.

The serving forward (`sharar/model/parallel_draft.py`) uses fused Triton
kernels, paged draft KV rings and FlashInfer; none of that has a backward. This
module recomputes the same function in plain torch for a batch of anchored
blocks: every block is the anchor token followed by mask tokens, attends
bidirectionally within itself and to the projected target features of the
positions before its anchor within the sliding window, exactly as the
publisher's reference model and Sharar's serving plan (``is_causal: false``).

Trainable weights are kept as FP32 masters and cast to BF16 for the forward;
``sync()`` writes them back into the serving weight dictionary in place, so a
proposer built on the same ``ParallelDraftModel`` serves the trained weights.
The feature projection, per-layer K/V projections and K norms are frozen by
default so retained draft rings and captured graphs stay valid across a
reload; ``train_context=True`` lifts that.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sharar.model.parallel_draft import ParallelDraftModel, convolve_reference, rms_reference, rotate_reference

MASK_TOKEN = 248070
FP4_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)  # E2M1 codes
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 5120, 32, 8, 128
CONTEXT_KEYS = ("fc.weight", "hidden_norm.weight")


def context_key(name: str) -> bool:
    """Weights that define cached context K/V: frozen unless train_context."""
    return name in CONTEXT_KEYS or name.endswith(("self_attn.kv", "self_attn.k_norm.weight"))


PARAMETER_SETS = {
    "all": lambda name: True,  # every non-context weight (about 1.74B)
    "attn": lambda name: not name.endswith(("mlp.gate_up", "mlp.down_proj.weight")),  # no MLP matrices (about 0.4B)
    "selector": lambda name: name.startswith("candidate_selector."),
}


def trainable_key(name: str, train_params: str, train_context: bool) -> bool:
    if not train_context and context_key(name):
        return False
    try:
        return PARAMETER_SETS[train_params](name)
    except KeyError:
        raise ValueError(f"unknown parameter set {train_params!r}; choose from {sorted(PARAMETER_SETS)}") from None


class Nvfp4HeadFunction(torch.autograd.Function):
    """Serving's exact NVFP4 head in the forward; BF16 dequantized weight in the backward."""

    @staticmethod
    def forward(ctx, hidden, head, dequantized):
        from sharar.kernels import ops
        ctx.save_for_backward(dequantized)
        with torch.no_grad():
            return ops.nvfp4_linear(hidden.detach().contiguous(), head)

    @staticmethod
    def backward(ctx, grad):
        (dequantized,) = ctx.saved_tensors
        return grad.to(dequantized.dtype) @ dequantized, None, None


def dequantize_nvfp4(weight: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """BF16 [N, K] from packed FP4 nibbles, per-16 block FP8 scales and the global scale."""
    lo = (weight & 0xF).long()
    hi = (weight >> 4).long()
    values = FP4_VALUES.to(weight.device)
    full = torch.stack([values[lo], values[hi]], dim=-1).reshape(weight.shape[0], -1)  # element 2i is the low nibble
    scales = scale.float().repeat_interleave(16, dim=1)
    return (full * scales * scale_2.float()).to(torch.bfloat16)


class Head:
    """The target's shared NVFP4 vocabulary head, differentiable with respect to its input."""

    def __init__(self, target, snapshot_weights):
        self.head = target.lm_head
        self.dequantized = dequantize_nvfp4(snapshot_weights.get("lm_head.weight").to(target.device),
                                            snapshot_weights.get("lm_head.weight_scale").to(target.device),
                                            snapshot_weights.get("lm_head.weight_scale_2").reshape(()).to(target.device))

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape[:-1]
        return Nvfp4HeadFunction.apply(hidden.reshape(-1, hidden.shape[-1]), self.head, self.dequantized).view(*shape, -1)


class TrainableDraft:
    def __init__(self, model: ParallelDraftModel, head: Head, *, block: int, window: int | None = None,
                 train_context: bool = False, train_params: str = "all"):
        if model.precision != "bf16" or model.fp8_weights:
            raise ValueError("training needs the BF16 draft profile")
        self.model, self.head, self.block = model, head, block
        self.window = window or model.window
        self.layers = model.cfg["num_hidden_layers"]
        self.top_k = model.cfg["dflash_config"]["selector_top_k"]
        self.train_context = train_context
        self.train_params = train_params
        self.masters: dict[str, torch.nn.Parameter] = {}
        for name, weight in model.weights.items():
            if trainable_key(name, train_params, train_context):
                self.masters[name] = torch.nn.Parameter(weight.detach().float().clone())

    # ---- weights ----
    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.masters.values())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.masters.values())

    def weight(self, name: str) -> torch.Tensor:
        master = self.masters.get(name)
        return master.to(torch.bfloat16) if master is not None else self.model.weights[name]

    @torch.no_grad()
    def sync(self) -> None:
        """Write the masters into the serving weights in place."""
        for name, master in self.masters.items():
            self.model.weights[name].copy_(master.to(torch.bfloat16))

    @torch.no_grad()
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint layout of the pinned publisher checkpoint (unfused, BF16, CPU)."""
        self.sync()
        out = {}
        for name, weight in self.model.weights.items():
            weight = weight.detach().to("cpu", torch.bfloat16).contiguous()
            if name.endswith("self_attn.kv"):
                k, v = weight.chunk(2, dim=0)
                out[name[:-2] + "k_proj.weight"], out[name[:-2] + "v_proj.weight"] = k.contiguous(), v.contiguous()
            elif name.endswith("mlp.gate_up"):
                gate, up = weight.chunk(2, dim=0)
                out[name[:-7] + "gate_proj.weight"], out[name[:-7] + "up_proj.weight"] = gate.contiguous(), up.contiguous()
            else:
                out[name] = weight
        return out

    def save(self, directory: str | Path, snapshot: Path, manifest: dict) -> None:
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(directory / "model.safetensors"), metadata={"format": "pt"})
        (directory / "config.json").write_text((snapshot / "config.json").read_text())
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # ---- forward ----
    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ self.weight(name).T

    def project_context(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        joined = torch.cat([features[i] for i in self.model.features], dim=-1)
        return rms_reference(self._linear(joined, "fc.weight"), self.weight("hidden_norm.weight"))

    def context_kv(self, layer: int, projected: torch.Tensor, positions: torch.Tensor):
        p = f"layers.{layer}.self_attn."
        k, v = self._linear(projected, p + "kv").chunk(2, dim=-1)
        k = rms_reference(k.view(-1, KV_HEADS, HEAD_DIM), self.weight(p + "k_norm.weight"))
        return rotate_reference(k, positions, self.model.cos, self.model.sin), v.view(-1, KV_HEADS, HEAD_DIM)

    def attention_mask(self, anchors: torch.Tensor, context_len: int) -> torch.Tensor:
        """[A*B, T + A*B] bool: context before the anchor inside the window, plus the own block."""
        rows = anchors.shape[0]
        offsets = torch.arange(self.block, device=anchors.device)
        q_anchor = anchors[:, None].expand(rows, self.block).reshape(-1)
        q_offset = offsets[None, :].expand(rows, self.block).reshape(-1)
        q_block = torch.arange(rows, device=anchors.device)[:, None].expand(rows, self.block).reshape(-1)
        kv = torch.arange(context_len + rows * self.block, device=anchors.device)
        context = (kv[None, :] < context_len) & (kv[None, :] < q_anchor[:, None]) \
            & (kv[None, :] >= (q_anchor + q_offset - (self.window - 1))[:, None])
        own = (kv[None, :] >= context_len) & (((kv[None, :] - context_len) // self.block) == q_block[:, None])
        return context | own

    def forward_blocks(self, projected: torch.Tensor, ids: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """projected [T, H] context features, ids [T], anchors [A] -> final-normed hidden [A, B, H]."""
        rows, block = anchors.shape[0], self.block
        context_len = projected.shape[0]
        tokens = torch.full((rows, block), MASK_TOKEN, dtype=torch.int64, device=ids.device)
        tokens[:, 0] = ids[anchors]
        hidden = F.embedding(tokens, self.model.target.embed)
        positions = (anchors[:, None] + torch.arange(block, device=ids.device)[None, :]).reshape(-1)
        context_positions = torch.arange(context_len, device=ids.device)
        mask = self.attention_mask(anchors, context_len)[None, None]
        cos, sin = self.model.cos, self.model.sin
        for layer in range(self.layers):
            p = f"layers.{layer}."
            residual = hidden
            x = rms_reference(hidden, self.weight(p + "input_layernorm.weight"))
            x, coefficients = self._prepare_conv(x, p + "attention_conv")
            flat = x.reshape(-1, HIDDEN)
            q = rms_reference(self._linear(flat, p + "self_attn.q_proj.weight").view(-1, HEADS, HEAD_DIM), self.weight(p + "self_attn.q_norm.weight"))
            k, v = self._linear(flat, p + "self_attn.kv").chunk(2, dim=-1)
            k = rms_reference(k.view(-1, KV_HEADS, HEAD_DIM), self.weight(p + "self_attn.k_norm.weight"))
            q = rotate_reference(q, positions, cos, sin)
            k = rotate_reference(k, positions, cos, sin)
            k_ctx, v_ctx = self.context_kv(layer, projected, context_positions)
            keys = torch.cat([k_ctx, k], dim=0).transpose(0, 1)[None]  # [1, 8, N, D]
            values = torch.cat([v_ctx, v.view(-1, KV_HEADS, HEAD_DIM)], dim=0).transpose(0, 1)[None]
            attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys, values, attn_mask=mask,
                                                      scale=HEAD_DIM ** -0.5, enable_gqa=True)
            attended = attended[0].transpose(0, 1).reshape(rows, block, HEADS * HEAD_DIM)
            projected_out = self._linear(attended, p + "self_attn.o_proj.weight")
            hidden = residual + self._finish_conv(projected_out, coefficients, p + "attention_conv")
            residual = hidden
            x = rms_reference(hidden, self.weight(p + "post_attention_layernorm.weight"))
            x, coefficients = self._prepare_conv(x, p + "mlp_conv")
            gate, up = self._linear(x, p + "mlp.gate_up").chunk(2, dim=-1)
            x = self._linear(F.silu(gate) * up, p + "mlp.down_proj.weight")
            hidden = residual + self._finish_conv(x, coefficients, p + "mlp_conv")
        return rms_reference(hidden, self.weight("norm.weight"))

    def _prepare_conv(self, hidden: torch.Tensor, name: str):
        if name + ".base_kernel" not in self.model.weights:
            return hidden, None
        coefficients = self._linear(hidden, name + ".kernel_projection.weight").view(*hidden.shape[:-1], 2, 2, HIDDEN // 16)
        base = self.weight(name + ".base_kernel")
        return convolve_reference(hidden, coefficients[:, :, 0], base[0]), coefficients[:, :, 1]

    def _finish_conv(self, hidden: torch.Tensor, coefficients, name: str) -> torch.Tensor:
        if coefficients is None:
            return hidden
        return convolve_reference(hidden, coefficients, self.weight(name + ".base_kernel")[1])

    # ---- selector ----
    def selector_weights(self):
        p = "candidate_selector."
        return (self.weight(p + "hidden_projection.weight"), self.weight(p + "predecessor_codebook"),
                self.weight(p + "successor_codebook"))

    @torch.no_grad()
    def lattice(self, projected: torch.Tensor, ids: torch.Tensor, anchors: torch.Tensor):
        """Serving's candidate lattice: candidates [A, B-1, W] and scores [A, B-1, W, W]."""
        hidden = self.forward_blocks(projected, ids, anchors)[:, 1:]
        logits = self.head(hidden)
        values, candidates = logits.topk(self.top_k, dim=-1)
        hidden_projection, predecessor_codebook, successor_codebook = self.selector_weights()
        gate = hidden @ hidden_projection.T
        anchor_ids = ids[anchors]
        predecessors = torch.cat([anchor_ids[:, None, None].expand(-1, 1, self.top_k), candidates[:, :-1]], dim=1)
        a = predecessor_codebook[predecessors]
        b = successor_codebook[candidates]
        scores = (a * gate[:, :, None]) @ b.transpose(-1, -2) + values[:, :, None]
        return candidates, scores

    @torch.no_grad()
    def propose(self, projected: torch.Tensor, ids: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Serving's greedy proposals [A, B-1] for each anchor (teacher-forced context)."""
        from sharar.train.loss import greedy_walk
        return greedy_walk(*self.lattice(projected, ids, anchors))

    @torch.no_grad()
    def forced_agreement(self, projected: torch.Tensor, ids: torch.Tensor, anchor: torch.Tensor, path: torch.Tensor) -> float:
        """Share of steps where the greedy choice given the previous token of ``path`` equals ``path``'s next token."""
        candidates, scores = self.lattice(projected, ids, anchor)
        candidates, scores = candidates[0], scores[0]
        previous, agree = 0, 0
        for j in range(candidates.shape[0]):
            chosen = int(scores[j, previous].argmax())
            agree += int(candidates[j, chosen] == path[j])
            found = (candidates[j] == path[j]).nonzero()
            if found.numel() == 0:
                break  # serving's token is not among this step's candidates: the paths cannot be compared further
            previous = int(found[0])
        return agree / candidates.shape[0]
