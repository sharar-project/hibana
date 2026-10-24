"""Nemotron-3.5-Super's routed NVFP4 experts through FlashInfer's b12x SM12x MoE (`b12x_fused_moe`'s backend, ReLU²).

The checkpoint's tensors as `cutlass_fused_moe` reads them: packed weights [E, N, K/2] and the block scales in the
SM120 128x4 interleave, which b12x takes as the 6-D MMA view of the same storage (no copy). b12x's kernels quantize
their inputs with dividing global scales (`cute_dsl/fp4_common.py`: block scale amax / (6 gs), codes v / (block scale
x gs)), so the checkpoint's input scales enter as they are (in1 for FC1's input, in2 for FC2's), and FC1's
dequantization multiplier is w1_alpha x input_global_scale = g1 x in1 (folded once here, as `B12xMoEWrapper` would per
call); routed_scaling_factor rides in FC2's alpha (in2 x g2 x 5), as on the cutlass path.

Workspaces: eager calls (prefill chunks) use FlashInfer's own cache; a call under CUDA-graph capture gets a workspace
of its own shape, kept for the graph's life (b12x's cached workspace can be reallocated by a larger eager call).

Not deterministic: two calls on the same inputs agree to cosine 0.9997-0.99996 (block 1, M 1-600), and against the
reference with NVFP4 activations a block through it reaches 0.9993-0.99996 where cutlass reaches 0.9994-0.99999 and
Sharar's small-call kernels 0.999998; GSM8K (exact, 300 paired) with it past 16 rows: 97.7% against 97.7%."""

from __future__ import annotations

import torch

_GRAPH_WORKSPACES: dict[tuple, object] = {}


class B12xExperts:
    def __init__(self, w1: torch.Tensor, s1: torch.Tensor, g1: torch.Tensor, in1: torch.Tensor, w2: torch.Tensor,
                 s2: torch.Tensor, g2: torch.Tensor, in2: torch.Tensor, routed_scaling: float):
        """w1 [E, N, K/2], w2 [E, K, N/2] uint8; s1, s2 their swizzled E4M3 block scales; g1, g2 [E] FP32 weight
        global scales; in1, in2 the layer's activation input scales (one per layer)."""
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_weight_views

        self.experts, self.n, self.k = w1.shape[0], w1.shape[1], w1.shape[2] * 2
        e, n, k = self.experts, self.n, self.k
        self.w1, self.w2 = w1, w2
        self.sf1 = convert_sf_to_mma_layout(s1.view(torch.float8_e4m3fn).reshape(-1, k // 16), m=n, k=k, num_groups=e)
        self.sf2 = convert_sf_to_mma_layout(s2.view(torch.float8_e4m3fn).reshape(-1, n // 16), m=k, k=n, num_groups=e)
        one = lambda v: torch.as_tensor(v, dtype=torch.float32, device=w1.device).reshape(1).contiguous()  # noqa: E731
        self.gs1, self.gs2 = one(in1), one(in2)  # the quantization's dividing global scales
        self.alpha1 = (g1.float() * in1.float()).contiguous()  # w1_alpha (g1) x input_global_scale (in1), pre-folded
        self.alpha2 = (g2.float() * in2.float() * routed_scaling).contiguous()
        self.views = _get_weight_views(w1_fp4=w1, w1_blockscale=self.sf1, w2_fp4=w2, w2_blockscale=self.sf2,
                                       w1_alphas=self.alpha1, w2_alphas=self.alpha2, n=n, k=k, activation_precision="fp4")
        if self.views.w1_scale_storage.data_ptr() != s1.data_ptr() or self.views.w2_scale_storage.data_ptr() != s2.data_ptr():
            raise RuntimeError("b12x copied the expert block scales (expected the MMA view of the swizzled storage)")

    def _graph_workspace(self, m: int, topk: int, device: torch.device):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            allocate_sm120_moe_workspace, select_sm120_moe_backend)
        backend = select_sm120_moe_backend(num_tokens=m, num_topk=topk, activation_precision="fp4")
        key = (str(device), self.experts, self.n, self.k, topk, m, backend)
        ws = _GRAPH_WORKSPACES.get(key)
        if ws is None:
            ws = _GRAPH_WORKSPACES[key] = allocate_sm120_moe_workspace(
                state_E=self.experts, weight_E=self.experts, k=self.k, n=self.n, num_topk=topk, device=device,
                routed_rows=m * topk, quant_mode="nvfp4", backend=backend, activation="relu2")
        return ws

    def __call__(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """BF16 x [M, K], ids / weights [M, k] (weights renormalized) -> BF16 [M, K], the weighted routed sum."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import launch_sm120_moe
        m, topk = ids.shape
        out = torch.empty((m, self.k), dtype=torch.bfloat16, device=x.device)
        ws = self._graph_workspace(m, topk, x.device) if torch.cuda.is_current_stream_capturing() else None
        return launch_sm120_moe(
            a=x, topk_ids=ids, topk_weights=weights, w1_weight=self.w1, w1_weight_sf=self.sf1, w1_alpha=self.alpha1,
            fc2_input_scale=self.gs2, input_global_scale=self.gs1, w2_weight=self.w2, w2_weight_sf=self.sf2,
            w2_alpha=self.alpha2, num_experts=self.experts, top_k=topk, num_local_experts=self.experts,
            scatter_output=out, activation="relu2", quant_mode="nvfp4", _workspace=ws, _weight_views=self.views)
