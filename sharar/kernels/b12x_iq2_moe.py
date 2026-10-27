"""Bolt's routed IQ2_XXS experts on the b12x package's fused MoE, as vLLM serves them (`ModelOptBlockQuantMoEMethod`
through `B12xExperts`), without vLLM: BF16 activations, ReLU^2 between the up and down projections, the router's
weights applied in the weighted sum, the checkpoint blocks retiled losslessly. (Super's NVFP4 experts take FlashInfer's
b12x port instead: `kernels/b12x_moe.py`.)

Each MoE block declares one execution plan of `PlanRows.capacity` rows with an exact variant per CUDA-graph row count;
a call binds the smallest variant that holds it. `prepare` compiles every block's variants in one preparation session
on the loading thread (b12x prepares only on the thread that owns its session) and gives all blocks one scratch
buffer, the largest any variant needs: the blocks run one after another on one stream.

Against the BF16 reference, the relative L2 error on Bolt's block-1 geometry is 0.33% where the grouped FP8 operator
(`kernels/iq2_moe.py`) has 3.3%, at the same speed for an eight-row verification (346 against 347 us).
"""
from __future__ import annotations

import torch


class Experts:
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, topk: int, rows):
        """w1 [E, I, H/256, 66] up and w2 [E, H, I/256, 66] down projections in IQ2_XXS blocks; `rows` a `PlanRows`."""
        from b12x.moe import fused_moe as moe
        self.experts, self.inter, self.hidden, self.topk = w1.shape[0], w1.shape[1], w2.shape[1], topk
        self.device = w1.device
        self.bytes = w1.numel() + w2.numel()  # the retiling keeps every byte and adds none
        plan = moe.plan_weights(source=moe.PackedSource(format="iq2_xxs", w13_layout="w31"),
                                activation=moe.ActivationSpec(mode="a16", nonlinearity="relu2", io_dtype=torch.bfloat16),
                                geometry=moe.MoEGeometry(num_experts=self.experts, hidden_size=self.hidden,
                                                         intermediate_size=self.inter))
        self.prepared = moe.prepare_weights(plan=plan, weights=moe.BlockQuantWeights(w1, w2, codec="iq2_xxs"))
        exact = tuple(m for m in sorted(set(rows.exact)) if m < rows.capacity)
        self.decl = moe.plan_execution(experts=self.prepared, routing=moe.RoutingSpec(deterministic_output=True),
                                       capacity=moe.ExecutionCapacity(max_tokens=rows.capacity, top_k=topk,
                                                                      warmup_token_counts=exact))
        self.scratch = None  # `prepare` assigns the shared buffer

    def request(self, name: str):
        """The preparation request: one representative call per variant, on dummy routes."""
        from b12x.preparation import PreparedCall

        def factory(count):
            def make(state):
                scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=self.device) for s in state.scratch.scratch_specs())
                x = torch.zeros(count, self.hidden, device=self.device, dtype=torch.bfloat16)
                ids = torch.arange(count * self.topk, device=self.device, dtype=torch.int32).view(count, self.topk) % self.experts
                weights = torch.full((count, self.topk), 1 / self.topk, device=self.device)
                out = torch.empty_like(x)
                bound = state.bind(scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=out)
                return PreparedCall(run=bound.run, output=out, owners=(scratch, bound, x, ids, weights))
            return make
        return self.decl.request(name=name, prepare_calls={c: factory(c) for c in self.decl.token_counts})

    def scratch_bytes(self) -> int:
        from b12x.preparation.types import require_prepared
        need = 0
        for variant in require_prepared(self.decl, "moe.decode").variants.values():
            specs = variant.scratch.scratch_specs()
            if len(specs) != 1 or specs[0].dtype != torch.uint8:
                raise RuntimeError("expected one uint8 b12x MoE scratch buffer per variant")
            need = max(need, int(torch.Size(specs[0].shape).numel()))
        return need

    def __call__(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """sum_k weights[:, k] * down_k(relu(up_k(x))^2) for x [M, H] BF16, int32 ids and FP32 weights [M, topk]."""
        from b12x.moe import fused_moe as moe
        out = torch.empty_like(x)
        binding = moe.bind(self.decl, scratch=self.scratch, a=x.contiguous(), topk_ids=ids.contiguous(),
                           topk_weights=weights.contiguous(), output=out)
        moe.run(binding=binding)
        return out


def prepare(blocks: list[Experts], log=None) -> int:
    """Compile every block's variants in one session on this thread and share one scratch buffer. Returns its bytes."""
    import time

    from b12x.preparation import PreparationSession
    if not blocks:
        return 0
    started = time.perf_counter()
    with PreparationSession(device="cuda", autotune=False, compile_workers=4) as session:
        session.prepare(tuple(b.request(f"bolt.moe.{i}") for i, b in enumerate(blocks)))
        session.freeze()
    size = max(b.scratch_bytes() for b in blocks)
    shared = (torch.empty(size, dtype=torch.uint8, device=blocks[0].device),)
    for b in blocks:
        b.scratch = shared
    if log:
        log(f"  b12x MoE: {len(blocks)} blocks prepared in {time.perf_counter() - started:.0f}s; scratch {size / 2**20:.0f} MiB")
    return size
