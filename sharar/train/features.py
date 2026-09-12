"""Target features for a whole sequence through the serving executor.

Teacher forcing: the sequence (prompt plus the target's recorded output) is
admitted as a prompt and prefilled in chunks; the target's post-block residual
streams at the draft's feature layers are captured from each forward. The
executor is created without the prefix cache so nothing is retained.
"""

from __future__ import annotations

import torch

from sharar.engine.contracts import FeatureRequest
from sharar.engine.sampler import SamplingParams


class FeatureExtractor:
    def __init__(self, executor, layers: tuple[int, ...], chunk: int = 2048):
        if executor.spec is not None:
            raise ValueError("feature extraction needs an executor without a proposer")
        if executor.cache is not None:
            raise ValueError("feature extraction needs an executor without the prefix cache")
        self.ex = executor
        self.layers = tuple(layers)
        self.chunk = chunk
        executor.feature_request = FeatureRequest(layers=self.layers)

    def __call__(self, ids: list[int]) -> dict[int, torch.Tensor]:
        captured: list[dict[int, torch.Tensor]] = []
        model = self.ex.model
        original = model.forward

        def forward(input_ids, ctx, capture=False):
            out = original(input_ids, ctx, capture)
            captured.append({lid: value.clone() for lid, value in out["features"].items()})
            return out

        req = None
        model.forward = forward
        try:
            with torch.inference_mode():
                req = self.ex.admit(list(ids), SamplingParams(max_new_tokens=1, ignore_eos=True))
                while not req.prefill_done:
                    self.ex.prefill(req, self.chunk)
        finally:
            model.forward = original
            if req is not None:
                with torch.inference_mode():
                    req.cacheable = False
                    self.ex.release(req)
        # Concatenate outside inference mode: the results are ordinary tensors
        # that autograd may save when the context projections are trained.
        features = {lid: torch.cat([c[lid] for c in captured], dim=0) for lid in self.layers}
        if any(value.shape[0] != len(ids) for value in features.values()):
            raise RuntimeError("captured features do not cover the sequence")
        return features
