"""Bolt's checkpoint-native weight-only projections; activations stay BF16.

Every dense projection runs b12x's SM120 BF16-activation GEMM, as vLLM's b12x linears do: NVFP4 through the
W4A16 recipe (vLLM's B12xNvFp4W4A16LinearKernel), Q8_0 and IQ2_XXS through the planned block-quant recipes
(vLLM's ModelOptBlockQuantLinearMethod). Weights stay in their checkpoint encodings; nothing quantizes activations.
"""

from dataclasses import dataclass

import torch

from sharar.kernels import w3
from sharar.weights.ggml import dequantize
from sharar.weights.modelopt import swizzle_block_scales, unpack_fp4, unswizzle_block_scales

CAPACITY = 4096  # most rows a planned call runs at once (the default prefill budget); longer calls are chunked


@dataclass(frozen=True)
class PlanRows:
    """The row counts a block-quant plan runs: `exact` get their own tuned programs, the rest share `capacity`'s."""
    capacity: int = CAPACITY
    exact: tuple[int, ...] = ()


def graph_rows(cfg) -> tuple[int, ...]:
    """The row counts Bolt's CUDA graphs run each projection at: a batch bucket times one (decode) or a speculative
    round's rows (the drafts, or the drafts and the verified token); every round length up to nine when SHARAR_SPEC
    does not fix the depth."""
    from sharar.engine.profile import ServingProfile
    depths = [int(cfg.spec)] if cfg.spec and cfg.spec.isdigit() else []
    depths += [int(part.split(":")[1]) for part in (cfg.spec_depths or "").split(",") if part]
    lengths = {1} | ({s for d in depths if d for s in (d, d + 1)} if depths else set(range(1, 10)))
    return tuple(sorted({b * s for b in ServingProfile(max_batch=cfg.max_batch).graph_sizes for s in lengths}))


@dataclass
class PackedLinear:
    weight: torch.Tensor | None  # Q8_0/IQ2_XXS: released once `prepare` packs it (b12x keeps a lossless retiling)
    quant: str
    scales: torch.Tensor | None = None  # NVFP4: given in checkpoint order, held in the CUTLASS 128x4 layout
    gain: torch.Tensor | None = None
    input_scale: object = None  # the common norm path emits BF16, without an FP8 copy
    w3: object = None  # a calibrated 3-bit copy (`Bolt.attach_calibrated`): calls of at most `w3_rows` rows read it
    w3_rows: int = 0
    b12x: object = None  # b12x's weight: NVFP4's borrows `weight` and `scales`
    plan: object = None  # the b12x plan `prepare` declares and compiles
    rows: PlanRows | None = None

    def __post_init__(self):
        self.shape = (self.weight.shape[0], self.weight.shape[1] * {"Q8_0": 32, "IQ2_XXS": 256, "NVFP4": 2}[self.quant])
        # NVFP4: the checkpoint's E2M1 bytes, swizzled E4M3 block scales and weight_scale_2 as a multiplier.
        if self.quant == "NVFP4" and self.b12x is None:
            from b12x.gemm import blockscaled
            self.scales = swizzle_block_scales(self.scales.view(torch.float8_e4m3fn))
            self.gain = self.gain.float().reshape(()).contiguous()
            self.b12x = blockscaled.pack_weight(self.weight, self.scales, recipe="nvfp4", global_scale=self.gain)

    @classmethod
    def load(cls, w, prefixes, device, shard=None):
        weights = [w.get(f"{p}.weight").to(device) for p in prefixes]
        if weights[0].ndim == 3:
            if any(v.ndim != 3 or v.shape[1:] != weights[0].shape[1:] for v in weights):
                raise ValueError("fused Bolt projections must share their encoding and input width")
            quant = {34: "Q8_0", 66: "IQ2_XXS"}[weights[0].shape[-1]]
            return cls(torch.cat(weights, dim=0).contiguous(), quant)
        if len(prefixes) != 1:
            raise ValueError("Bolt's NVFP4 projections are separate")
        p = prefixes[0]
        return cls(weights[0].contiguous(), "NVFP4", w.get(f"{p}.weight_scale").to(device).contiguous(),
                   w.get(f"{p}.weight_scale_2").to(device).float().reshape(()))

    def prepare(self, rows: PlanRows = PlanRows()) -> None:
        """Declare and compile the b12x plan for `rows` (Q8_0/IQ2_XXS: packing first and releasing the checkpoint
        copy). Compiling happens here, on the loading thread: b12x prepares lazily on the thread that owns its
        preparation session, and the engine's inference thread does not."""
        if self.plan is not None:
            return
        from b12x.gemm import blockscaled
        from b12x.preparation.types import require_prepared
        recipe = self.quant.lower()
        if self.quant != "NVFP4":
            self.b12x = blockscaled.pack_weight(self.weight, recipe=recipe)
        extra = {"global_scale_kind": "multiplier"} if self.quant == "NVFP4" else {}
        query = blockscaled.BlockscaledQuery(recipe=recipe, num_tokens=rows.capacity, in_features=self.in_features,
                                             padded_in_features=self.in_features, out_features=self.out_features,
                                             activation_mode="a16", **extra)
        self.plan = blockscaled.plan_regimes(query, exact_m=tuple(m for m in sorted(set(rows.exact)) if m < rows.capacity))
        require_prepared(self.plan, "gemm.blockscaled_precision", self.b12x.values.device)
        self.rows = rows
        if self.quant != "NVFP4":
            self.weight = None

    @property
    def out_features(self):
        return self.shape[0]

    @property
    def in_features(self):
        return self.shape[1]

    @property
    def bytes(self):
        if self.weight is None:
            return self.b12x.values.numel() + self.b12x.metadata.numel()
        return self.weight.numel() + sum(t.numel() * t.element_size() for t in (self.scales, self.gain) if t is not None)

    def dequantize(self):
        if self.quant == "NVFP4":
            scales = unswizzle_block_scales(self.scales.reshape(-1), self.out_features, self.in_features // 16)
            return unpack_fp4(self.weight, scales) * self.gain
        if self.weight is None:
            raise ValueError("a prepared Q8_0/IQ2_XXS projection keeps only b12x's packing; read the snapshot")
        return dequantize(self.weight, self.quant)

    def __call__(self, x, *, relu2=False):
        from b12x.gemm import blockscaled
        if x.dtype != torch.bfloat16:
            raise ValueError("Bolt's b12x projections require BF16 inputs")
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if self.w3 is not None and x2.shape[0] <= self.w3_rows and w3.enabled():
            # the copy's kernel quantizes activations to E4M3 per row and 32-group
            y = w3.w3_linear(x2, self.w3)
        else:
            if self.plan is None:
                self.prepare()
            step = self.rows.capacity
            y = (blockscaled.mm(x2, self.b12x, plan=self.plan) if x2.shape[0] <= step else
                 torch.cat([blockscaled.mm(x2[i:i + step], self.b12x, plan=self.plan) for i in range(0, x2.shape[0], step)]))
        return (y.relu().square() if relu2 else y).view(*x.shape[:-1], self.out_features)
