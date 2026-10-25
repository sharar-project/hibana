# SPDX-License-Identifier: Apache-2.0
"""NVIDIA-Nemotron-3.5-Super's text model assembled for the engine: 88 pre-norm residual blocks (40 Mamba-2, 40
latent MoE, 8 attention), the final norm and the FP8 lm_head. The vision tower and its projector are never read.

`forward()` follows the 27B's contract (`ForwardContext` prefill / decode / verify; `norm` is the final norm the head
reads, `pre_norm` the residual stream before it, which the MTP head consumes). Block indices are the checkpoint's, so
`Target.gdn_layer_ids` (the Mamba blocks) and `attention_layer_ids` address the pools directly. Weights stay in the
checkpoint's formats: FP8 Mamba and attention projections and head, NVFP4 experts, BF16 router, latent projections
and embeddings. With SHARAR_W3_DECODE=1 and SHARAR_W3_CHECKPOINT the FP8 projections also carry calibrated 3-bit
copies (`attach_calibrated`, `weights/nemotron_w3_calibrate.py`) that calls of at most 64 rows read.
A W3 checkpoint that carries the routed experts of every MoE block (`layers.N.experts.w1` / `.w2`,
`weights/nemotron_w3_experts.py`) serves those 3-bit experts as the only expert copy, the MTP head's too when it
carries `mtp.experts.*` (SHARAR_NEMOTRON_EXPERTS=nvfp4 keeps the NVFP4 ones); its trunk entries, if any, attach as above.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from sharar.distributed import all_gather, all_reduce, get_tp_info
from sharar.engine.config import current
from sharar.engine.context import ForwardContext, scratch_prefill_context
from sharar.engine.contracts import ForwardResult
from sharar.kernels import ops
from sharar.model.nemotron import config as c
from sharar.model.nemotron.attention import NemotronAttention
from sharar.model.nemotron.config import PREFIX
from sharar.model.nemotron.mamba import NemotronMamba
from sharar.model.nemotron.moe import NemotronMoe
from sharar.model.nemotron.norm import RMSNorm
from sharar.target import NEMOTRON, Target
from sharar.weights import files
from sharar.weights.modelopt import load_fp8_linear
from sharar.weights.safetensors import SnapshotWeights

BACKBONE = f"{PREFIX}.backbone"


def drop_page_cache(snap: Path) -> None:
    """Tell the kernel the checkpoint's shards are not needed again: everything is on the device by now. The load leaves
    ~30 GiB of them cached, and /proc/meminfo's MemAvailable (what the memory plan sizes the pools from) counts only
    part of a hot page cache as available, so a served engine otherwise plans against ~9 GiB less than it has (measured
    on the GB10: evicting the shards raised MemAvailable from 103.6 to 112.0 GiB with the model unloaded)."""
    import os
    for shard in snap.glob("model-*.safetensors"):
        fd = os.open(shard, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


class Block:
    """norm -> mixer -> residual add (folded into the next block's norm)."""

    def __init__(self, w, index: int, kind: str, device: torch.device, ragged, target: Target, experts3=None):
        p = f"{BACKBONE}.layers.{index}"
        self.index, self.kind = index, kind
        self.norm = RMSNorm(w.get(f"{p}.norm.weight").to(device), target.rms_norm_eps)
        if kind == "linear_attention":
            self.mixer = NemotronMamba(w, f"{p}.mixer", index, device, target)
        elif kind == "full_attention":
            self.mixer = NemotronAttention(w, f"{p}.mixer", index, device, ragged, target)
        else:
            self.mixer = NemotronMoe(w, f"{p}.mixer", device, target, w3=experts3)

    @property
    def bytes(self) -> int:
        return self.norm.bytes + self.mixer.bytes

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None, ctx: ForwardContext):
        inplace = not ctx.capture
        consume = inplace and ctx.mode == "prefill"
        scale = getattr(self.mixer, "input_scale", None)  # the FP8 input of the mixer's first projection, when it has one
        hidden, hidden_q, residual = self.norm.fused_add_quant(hidden, residual, scale, inplace=inplace, consume_input=consume)
        if self.kind == "moe":
            return all_reduce(self.mixer(hidden, ctx)), residual
        return all_reduce(self.mixer.forward(hidden, ctx, hidden_q)), residual


def w3_expert_blocks(calibrated, kinds) -> set[int]:
    """The MoE block indices the W3 checkpoint (`calibrated`, or None) serves 3-bit experts for: every MoE block's when
    it carries them (a partial set is an error) and SHARAR_NEMOTRON_EXPERTS is w3 (the default); else none."""
    if calibrated is None or current().nemotron_experts != "w3":
        return set()
    have = calibrated.expert_layers()
    moe = {i for i, kind in enumerate(kinds) if kind == "moe"}
    if have and have != moe:
        raise ValueError(f"the W3 checkpoint carries routed experts for {len(have & moe)} of {len(moe)} MoE blocks")
    return have


def w3_expert_saving(target: Target, blocks: int) -> int:
    """Resident bytes 3-bit latent experts save against NVFP4 over `blocks` blocks (codes, group and row scales against
    E2M1 codes and E4M3 per-16 block scales)."""
    i, lat = target.moe_intermediate_size, c.MOE_LATENT_SIZE
    weights = 2 * i * lat
    nvfp4 = weights // 2 + weights // 16
    w3 = weights * 3 // 8 + weights // 32 + (i + lat) * 4
    return blocks * target.num_experts * (nvfp4 - w3)


class Nemotron:
    def __init__(self, snap: Path, device: torch.device | str = "cuda", log=print, target: Target = NEMOTRON, **_ignored):
        from sharar.cache.ledger import check_load_headroom
        tp = get_tp_info()
        if tp.size != 1:
            raise ValueError("Nemotron serves on one Spark")
        # Calibrated only: nearest rounding's output error on held-out activations is two to four times GPTQ's
        # (docs/w3-calibration.md, Nemotron), so there is no nearest-rounded fallback.
        calibrated = None
        if current().w3_decode:
            if not current().w3_checkpoint:
                raise ValueError("Nemotron's 3-bit copies are calibrated only: SHARAR_W3_DECODE=1 needs SHARAR_W3_CHECKPOINT")
            from sharar.weights.w3_checkpoint import CalibratedWeights
            calibrated = CalibratedWeights(current().w3_checkpoint, revision=target.model_revision)  # checked before the load
        self.device = torch.device(device)
        cfg = target.load_model_config(snap)
        expert_blocks = w3_expert_blocks(calibrated, target.layer_types_of(cfg))
        check_load_headroom(target.weight_load_peak_bytes - (w3_expert_saving(target, len(expert_blocks)) if expert_blocks else 0))
        self.target = target
        self.eos = cfg["llm_config"]["eos_token_id"]
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.embed = w.get(f"{BACKBONE}.embeddings.weight").to(self.device)
        self.ragged = ops.RaggedAttention(self.device, target.num_attention_heads, target.num_kv_heads, target.head_dim)
        self.layers: list[Block] = []
        try:
            for i, kind in enumerate(target.layer_types_of(cfg)):
                experts3 = (calibrated.load_latent_experts(f"layers.{i}", target.num_experts, target.moe_intermediate_size,
                                                           c.MOE_LATENT_SIZE, self.device) if i in expert_blocks else None)
                self.layers.append(Block(w, i, kind, self.device, self.ragged, target, experts3))
                w.close()  # GB10 counts source maps and allocations against one pool: reopen shards per block
                if log and (i + 1) % 11 == 0:
                    log(f"  loaded {i + 1}/{target.num_layers} blocks ({time.perf_counter() - started:.0f}s, "
                        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
        except BaseException:
            files.discard()
            raise
        self.final_norm = RMSNorm(w.get(f"{BACKBONE}.norm_f.weight").to(self.device), target.rms_norm_eps)
        self.lm_head = load_fp8_linear(w, [f"{PREFIX}.lm_head"], self.device)
        w.close()
        # the MTP head's 3-bit experts, when the checkpoint carries them (`NemotronMtpHead` takes them from here)
        self.mtp_experts3 = (calibrated.load_latent_experts("mtp", target.num_experts, target.moe_intermediate_size,
                                                            c.MOE_LATENT_SIZE, self.device)
                             if expert_blocks and "mtp.experts.w1" in calibrated.weights else None)
        torch.cuda.synchronize()
        drop_page_cache(snap)
        self.w3_identity = self.attach_calibrated(calibrated, log) if calibrated else None
        # Bytes one decode token streams: every block but the experts, the head, one embedding row and its routed experts
        # (a projection with a 3-bit copy counts the copy's bytes, which decode reads instead).
        moes = [layer.mixer for layer in self.layers if layer.kind == "moe"]
        routed = sum(m.expert_bytes for m in moes) * target.experts_per_token // target.num_experts
        self.stream_bytes = (sum(layer.bytes for layer in self.layers) - sum(m.expert_bytes for m in moes) + routed
                             + self.lm_head.bytes + self.embed.shape[1] * self.embed.element_size())
        if log and expert_blocks:
            log(f"  experts: 3-bit from {calibrated.directory} ({sum(m.expert_bytes for m in moes) / 2**30:.1f} GiB"
                f"{', and the MTP head' if self.mtp_experts3 is not None else ''})")
        if log:
            log(f"  weights loaded in {time.perf_counter() - started:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated "
                f"({self.stream_bytes / 1e9:.2f} GB per decode token)")

    def linears(self):
        """Every FP8 projection by name (`weights/linear.py`): Mamba in/out, attention qkv/o, and the lm_head. The
        experts, router and latent projections are the MoE's own."""
        for layer in self.layers:
            for name in ("in_proj", "out_proj", "qkv", "o"):
                lin = getattr(layer.mixer, name, None)
                if lin is not None:
                    yield f"layers.{layer.index}.mixer.{name}", lin
        yield "lm_head", self.lm_head

    def attach_calibrated(self, calibrated, log=print) -> str:
        """Calibrated 3-bit decode copies over the FP8 projections, by `linears()` name; calls of at most 64 rows then
        read them (`ops.fp8_linear`), prefill keeps the FP8 weights. A projection the checkpoint does not carry stays
        FP8 (a checkpoint fitted with `--exclude lm_head` keeps the FP8 head). Returns the manifest's identity."""
        if isinstance(calibrated, (str, Path)):
            from sharar.weights.w3_checkpoint import CalibratedWeights
            calibrated = CalibratedWeights(calibrated, revision=self.target.model_revision)
        linears = dict(self.linears())
        trunk = {n for n in calibrated.weights if ".experts." not in n}  # routed experts load with their blocks
        missing = sorted(trunk - set(linears))
        if missing:
            raise ValueError(f"calibrated names not in this model: {missing[:4]}")
        if not trunk:
            return calibrated.identity
        loads = [(name, linears[name], linears[name].shard) for name in linears if name in calibrated.weights]
        from sharar.kernels import w3
        for (_, lin, _), copy in zip(loads, calibrated.load_many(loads)):
            # the narrow kernel's 64 rows, as measured (the wide kernel's table is the 27B's, as for Flash-Next's trunk)
            lin.w3, lin.w3_rows = copy, w3.NARROW_ROWS
        torch.cuda.synchronize()
        if log:
            fp8 = [name for name in linears if name not in calibrated.weights]
            log(f"  w3: {len(loads)} calibrated matrices from {calibrated.directory} "
                f"({sum(lin.w3.bytes for _, lin, _ in loads) / 2**30:.2f} GiB)" + (f"; FP8 only: {', '.join(fp8)}" if fp8 else ""))
        return calibrated.identity

    def forward_extras(self):
        """No side inputs beyond the ForwardContext."""
        return None

    def register_pages(self, kv_pool) -> None:
        """No per-page rows beyond K/V."""

    def register_state(self, pool) -> None:
        """No per-slot state beyond the Mamba layers' conv and SSM states."""

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext, capture: bool = False) -> ForwardResult:
        hidden = self.embed_ids(input_ids)
        ctx.capture = capture
        out: dict = {"input_ids": input_ids, "positions": ctx.positions}
        if capture:
            out["embed"] = hidden
            out["layers"] = []
        residual = None
        for layer in self.layers:
            hidden, residual = layer.forward(hidden, residual, ctx)
            if layer.index in ctx.feature_request.layers:
                out.setdefault("features", {})[layer.index] = hidden + residual
            if capture:
                out["layers"].append((hidden, residual))
        normed, pre_norm = self.final_norm.fused_add(hidden, residual)
        out["norm"] = normed
        out["pre_norm"] = pre_norm
        return out

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self.embed)

    def prefill(self, input_ids: torch.Tensor, capture: bool = False) -> dict:
        """Stand-alone prefill without pools (parity tools)."""
        return self.forward(input_ids, scratch_prefill_context(input_ids.shape[0], self.device), capture)

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return all_gather(ops.fp8_linear(hidden, self.lm_head))

    head_logits = logits
