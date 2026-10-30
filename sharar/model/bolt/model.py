"""Bolt text inference with native packed weights and the Nemotron Mamba runtime.

Every block's expert count/width comes from its pinned Puzzle configuration.
IQ2 routed and shared experts stay packed; Q8 attention/head/latent projections
and NVFP4 Mamba projections run b12x's BF16-activation GEMMs (`linear.py`). No MTP is in this pin.
"""

from __future__ import annotations

import hashlib
import time
from importlib.metadata import version
from pathlib import Path

import torch

from sharar.distributed import get_tp_info
from sharar.engine.config import current
from sharar.kernels import ggml_bf16, ops
from sharar.model.bolt import tuning
from sharar.model.bolt.config import BOLT
from sharar.model.bolt.linear import PackedLinear, PlanRows, graph_rows
from sharar.model.nemotron.attention import NemotronAttention
from sharar.model.nemotron.mamba import NemotronMamba
from sharar.model.nemotron.model import BACKBONE, Block, Nemotron, drop_page_cache
from sharar.model.nemotron.moe import NemotronMoe, route_budget, route_topk
from sharar.model.nemotron.norm import RMSNorm
from sharar.weights.safetensors import SnapshotWeights


class BoltMoe:
    def __init__(self, w, prefix, spec, device):
        self.experts, self.inter, self.topk = spec.experts, spec.intermediate, spec.topk
        self.router = w.get(f"{prefix}.gate.weight").to(device).contiguous()
        self.bias = w.get(f"{prefix}.gate.e_score_correction_bias").to(device).float().contiguous()
        self.fc1 = PackedLinear.load(w, [f"{prefix}.fc1_latent_proj"], device)
        self.fc2 = PackedLinear.load(w, [f"{prefix}.fc2_latent_proj"], device)
        self.shared_up = PackedLinear.load(w, [f"{prefix}.shared_experts.up_proj"], device)
        self.shared_down = PackedLinear.load(w, [f"{prefix}.shared_experts.down_proj"], device)
        names = [f"{prefix}.experts.{e}.{proj}_proj.weight" for e in range(self.experts) for proj in ("up", "down")]
        tensors = w.get_whole(names, device)
        self.w1 = torch.stack([tensors[f"{prefix}.experts.{e}.up_proj.weight"] for e in range(self.experts)])
        self.w2 = torch.stack([tensors[f"{prefix}.experts.{e}.down_proj.weight"] for e in range(self.experts)])
        if self.w1.shape != (self.experts, self.inter, BOLT.latent // 256, 66) or self.w2.shape != (
                self.experts, BOLT.latent, self.inter // 256, 66):
            raise ValueError(f"{prefix}: encoded expert shapes differ from the pinned Puzzle block")
        if self.router.shape != (self.experts, BOLT.hidden) or self.bias.shape != (self.experts,):
            raise ValueError(f"{prefix}: router shape differs from the pinned Puzzle block")
        # expert sharing (SHARAR_MOE_KEEP / SHARAR_MOE_BUDGET), as Super's: each real token keeps its top `keep`, and a
        # decode or verify call reads at most min(budget, per_request x requests) distinct experts (`route_budget`)
        cfg = current()
        self.keep = cfg.moe_keep if cfg.moe_keep < self.topk else 0
        self.budget, self.per_request = (cfg.moe_budget, cfg.moe_budget_per_request) if self.keep else (0, 0)
        self.iq2 = self.b12x = None
        backend = cfg.bolt_experts
        if backend == "fp8":
            from sharar.kernels import iq2_moe
            self.iq2 = iq2_moe.pack_experts(self.w1, self.w2)
            del self.w1, self.w2
        elif backend == "b12x":
            from sharar.kernels import b12x_iq2_moe
            self.b12x = b12x_iq2_moe.Experts(self.w1, self.w2, self.topk, PlanRows(exact=graph_rows(current())))
            del self.w1, self.w2

    @property
    def expert_bytes(self):
        packed = self.iq2 if self.iq2 is not None else self.b12x
        return packed.bytes if packed is not None else self.w1.numel() + self.w2.numel()

    @property
    def bytes(self):
        return (self.expert_bytes + self.router.numel() * self.router.element_size() + self.bias.numel() * 4
                + sum(lin.bytes for lin in (self.fc1, self.fc2, self.shared_up, self.shared_down)))

    def forward(self, x, ctx=None, x_q=None):
        """`Block.forward`'s mixer call: the projections read BF16, so the norm's FP8 copy `x_q` goes unread."""
        return self(x, ctx)

    def fallback(self, rows: int, steps: int) -> int:
        """`NemotronMoe.fallback`'s guarantee when the floors' union exceeds the cap: cap / rows, at least
        min(SHARE_MIN, keep) and below the floor; 0 without a cap or when that is the floor itself."""
        if not self.budget:
            return 0
        cap = min(self.budget, self.per_request * (rows // steps)) if self.per_request else self.budget
        g = max(min(NemotronMoe.SHARE_MIN, self.keep), min(self.keep, cap // rows))
        return g if g < self.keep else 0

    def __call__(self, x, ctx=None):
        x = x.contiguous()
        logits = torch.mm(x, self.router.T, out_dtype=torch.float32)
        latent = self.fc1(x)
        fused = torch.cat((logits, latent.float()), dim=-1)
        slots, steps = None, 1
        if ctx is not None and ctx.mode in ("decode", "verify"):
            slots, steps = (ctx.slots, 1) if ctx.mode == "decode" else (ctx.seq_slots, ctx.verify_steps)
        # a dropped slot or a padded row names a real expert at weight 0 (`repeat`), never id E (no expert): with id E
        # in real rows, served generations with sharing broke down at once
        if self.keep and slots is not None:
            ids, weights, latent = route_budget(fused, self.bias, self.topk, BOLT.latent, self.keep, self.budget, slots,
                                                steps, per_request=self.per_request, repeat=True,
                                                fallback=self.fallback(fused.shape[0], steps))
        else:
            ids, weights, latent = route_topk(fused, self.bias, self.topk, BOLT.latent, slots, steps, repeat=True)
        if self.b12x is not None:
            routed = self.b12x(latent, ids, weights * BOLT.route_scale)
        elif self.iq2 is not None:
            from sharar.kernels import iq2_moe
            # Temporary workspace belongs to this call. The graph allocator reuses
            # it between layers; retaining one per layer would multiply scratch by 40.
            routed = iq2_moe.moe(self.iq2, latent, ids, weights * BOLT.route_scale)
        else:
            up = ggml_bf16.matmul(latent, self.w1, "IQ2_XXS", ids=ids, repeat_input=True, relu2=True)
            pairs = ggml_bf16.matmul(up, self.w2, "IQ2_XXS", ids=ids)
            routed = ggml_bf16.combine(pairs, weights, BOLT.route_scale)
        shared = self.shared_down(self.shared_up(x, relu2=True))
        return self.fc2(routed) + shared


class BoltBlock(Block):
    def __init__(self, w, index, device, ragged, target):
        self.index, self.kind = index, target.layer_types[index]
        p = f"{BACKBONE}.layers.{index}"
        self.norm = RMSNorm(w.get(f"{p}.norm.weight").to(device), target.rms_norm_eps)
        if self.kind == "linear_attention":
            self.mixer = NemotronMamba(w, f"{p}.mixer", index, device, target, load_linear=PackedLinear.load)
        elif self.kind == "full_attention":
            self.mixer = NemotronAttention(w, f"{p}.mixer", index, device, ragged, target, load_linear=PackedLinear.load)
        else:
            self.mixer = BoltMoe(w, f"{p}.mixer", BOLT.blocks[index], device)


class Bolt(Nemotron):
    def __init__(self, snap: Path, device="cuda", log=print, target=None, **_ignored):
        from sharar.cache.ledger import check_load_headroom
        from sharar.target import NEMOTRON_BOLT
        target = target or NEMOTRON_BOLT
        if get_tp_info().size != 1:
            raise ValueError("Bolt serves on one Spark")
        cfg = current()
        self.bolt_experts = cfg.bolt_experts
        # the SSD store keys recurrent state by the arithmetic that produced it: b12x's linears, their launch
        # configurations (split-K orders the sums) and the row counts planned exactly (the rest share the capacity
        # plan), the expert backend and expert sharing
        launch = hashlib.sha256(repr(sorted(tuning.CONFIGS.items())).encode()).hexdigest()[:16]
        self.arithmetic_identity = {"linears": f"b12x {version('b12x')}", "launch_configs": launch,
                                    "plan_rows": list(graph_rows(cfg)), "iq2_experts": cfg.bolt_experts}
        if cfg.bolt_experts == "fp8":
            from sharar.kernels.iq2_moe import _source_revision
            self.arithmetic_identity["kernel"] = _source_revision()
        if 0 < cfg.moe_keep < min(b.topk for b in BOLT.blocks if b.kind == "moe"):  # BoltMoe routes in full from top-k on
            self.arithmetic_identity["moe_sharing"] = f"{cfg.moe_keep}:{cfg.moe_budget_per_request}:{cfg.moe_budget}"
        calibrated = None
        if cfg.w3_decode:
            # calibrated only (`weights/bolt_w3_calibrate.py`), as for Super: there is no nearest-rounded fallback
            if not cfg.w3_checkpoint:
                raise ValueError("Bolt's 3-bit copies are calibrated only: SHARAR_W3_DECODE=1 needs SHARAR_W3_CHECKPOINT")
            from sharar.weights.w3_checkpoint import CalibratedWeights
            calibrated = CalibratedWeights(cfg.w3_checkpoint, revision=target.model_revision)
        if cfg.kv_format not in ("fp8", "sim-bf16"):
            raise ValueError("Bolt currently supports FP8 KV (sim-bf16 for parity)")
        target.load_model_config(snap)
        check_load_headroom(target.weight_load_peak_bytes)
        self.target, self.device = target, torch.device(device)
        self.eos, self.w3_identity, self.mtp_experts3 = 11, None, None
        started = time.perf_counter()
        w = SnapshotWeights(snap)
        self.layers = []
        try:
            self.embed = w.get(f"{BACKBONE}.embeddings.weight").to(self.device).contiguous()
            if self.embed.shape != (BOLT.vocab, BOLT.hidden) or self.embed.dtype != torch.bfloat16:
                raise ValueError("unexpected Bolt BF16 embedding shape")
            self.ragged = ops.RaggedAttention(self.device, target.num_attention_heads, target.num_kv_heads, target.head_dim)
            for i in range(target.num_layers):
                self.layers.append(BoltBlock(w, i, self.device, self.ragged, target))
                w.close()
                if log and (i + 1) % 11 == 0:
                    log(f"  loaded {i + 1}/{target.num_layers} Bolt blocks ({time.perf_counter() - started:.0f}s, "
                        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
            self.final_norm = RMSNorm(w.get(f"{BACKBONE}.norm_f.weight").to(self.device), target.rms_norm_eps)
            self.lm_head = PackedLinear.load(w, ["language_model.lm_head"], self.device)
            if calibrated is not None:
                self.w3_identity = self.attach_calibrated(calibrated, log)
            self.draft_lm_head = None
            if cfg.draft_head != "fp8":
                # the drafter's own head (SHARAR_DRAFT_HEAD): a nearest-rounded copy of the Q8_0 rows, packed before
                # `prepare` releases them; verification keeps the exact head, so it moves acceptance, not output
                from sharar.proposers.draft_head import pack_rows
                from sharar.weights.ggml import dequantize
                head = self.lm_head
                self.draft_lm_head = pack_rows(lambda r0, r1: dequantize(head.weight[r0:r1], "Q8_0").float(),
                                               head.out_features, head.in_features, cfg.draft_head)
            from sharar.kernels import b12x_iq2_moe
            b12x_iq2_moe.prepare([layer.mixer.b12x for layer in self.layers if layer.kind == "moe" and layer.mixer.b12x], log)
            rows = graph_rows(cfg)
            for _, lin in self.linears():
                # the head reads at most the graphs' rows (longer calls chunk): a prefill capacity would hold a
                # [4096, 131072] workspace
                lin.prepare(PlanRows(2 * max(rows), rows) if lin is self.lm_head else PlanRows(exact=rows))
        finally:
            w.close()
        torch.cuda.synchronize()
        drop_page_cache(snap)
        moes = [layer.mixer for layer in self.layers if layer.kind == "moe"]
        self.stream_bytes = (sum(layer.bytes for layer in self.layers) + self.final_norm.bytes + self.lm_head.bytes
                             + self.embed[0].numel() * self.embed.element_size() - sum(m.expert_bytes for m in moes)
                             + sum(m.expert_bytes * m.topk // m.experts for m in moes))
        if log:
            log(f"  Bolt loaded in {time.perf_counter() - started:.0f}s; {torch.cuda.memory_allocated() / 2**30:.2f} GiB CUDA allocated")

    def linears(self):
        for layer in self.layers:
            for name in ("in_proj", "out_proj", "qkv", "o", "fc1", "fc2", "shared_up", "shared_down"):
                lin = getattr(layer.mixer, name, None)
                if lin is not None:
                    yield f"layers.{layer.index}.mixer.{name}", lin
        yield "lm_head", self.lm_head

    def attach_calibrated(self, calibrated, log=print) -> str:
        """Calibrated 3-bit decode copies over the projections the checkpoint carries, by `linears()` name; calls of
        at most 64 rows read them, longer ones keep b12x on the checkpoint encodings. Runs before `prepare` releases
        the Q8_0 bytes. Returns the manifest's identity."""
        from sharar.kernels import w3
        from sharar.weights.sharding import REPLICATED
        linears = dict(self.linears())
        missing = sorted(set(calibrated.weights) - set(linears))
        if missing:
            raise ValueError(f"calibrated names not in this model: {missing[:4]}")
        loads = [(name, linears[name], REPLICATED) for name in linears if name in calibrated.weights]
        for (_, lin, _), copy in zip(loads, calibrated.load_many(loads)):
            lin.w3, lin.w3_rows = copy, w3.NARROW_ROWS
        torch.cuda.synchronize()
        if log:
            kept = [name for name in linears if name not in calibrated.weights]
            log(f"  w3: {len(loads)} calibrated matrices from {calibrated.directory} "
                f"({sum(lin.w3.bytes for _, lin, _ in loads) / 2**30:.2f} GiB); {len(kept)} keep their checkpoint encoding")
        return calibrated.identity

    @torch.inference_mode()
    def logits(self, hidden):
        return self.lm_head(hidden.contiguous())

    @torch.inference_mode()
    def head_logits(self, hidden):
        """The drafter's vocabulary projection: its 3-bit/2-bit head copy when SHARAR_DRAFT_HEAD asks for one, else
        the head itself. The target's verification always reads `logits`."""
        if self.draft_lm_head is None:
            return self.logits(hidden)
        from sharar.kernels import w3
        parts = [w3.w3_linear(part.contiguous(), self.draft_lm_head) for part in hidden.reshape(-1, hidden.shape[-1]).split(w3.MAX_ROWS)]
        return parts[0] if len(parts) == 1 else torch.cat(parts)
