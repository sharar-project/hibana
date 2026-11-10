"""Offline activation-aware quantization into the W3 format (kernels/w3.py).

Uses the GPTQ column-error update (Frantar et al., arXiv:2210.17323): the
upper Cholesky factor of the inverse damped input covariance distributes
each column's error into columns that have not yet been quantized. Groups
remain in their original order so serving needs no activation permutation.
Scales are selected on an activation-weighted clipping grid, rounded to the
kernel's E4M3 scale format before the column updates.
"""
from __future__ import annotations

import torch

from sharar.kernels import w3


@torch.no_grad()
def fit_scales(weight, diagonal, levels, ratios=(.6, .7, .8, .9, 1.0)):
    n, k = weight.shape
    grouped = weight.float().reshape(n, k // w3.GROUP, w3.GROUP)
    base = grouped.abs().amax(-1) / max(abs(x) for x in levels)
    row_scale = (base.amax(-1) / w3.FP8_MAX).clamp_min(1e-12)
    lv = torch.tensor(levels, dtype=torch.float32, device=weight.device)
    mids = (lv[1:] + lv[:-1]) / 2
    importance = diagonal.float().reshape(1, k // w3.GROUP, w3.GROUP)
    best_loss = torch.full_like(base, float("inf"))
    best = torch.zeros_like(base)
    for ratio in ratios:
        scale = (base * ratio / row_scale[:, None]).clamp(0, w3.FP8_MAX).to(torch.float8_e4m3fn).float()
        effective = scale * row_scale[:, None]
        q = torch.bucketize(grouped / effective.clamp_min(1e-30)[:, :, None], mids)
        error = grouped - lv[q] * effective[:, :, None]
        loss = (error.square() * importance).sum(-1)
        improved = loss < best_loss
        best_loss = torch.minimum(loss, best_loss)
        best = torch.where(improved, scale, best)
    return best.to(torch.float8_e4m3fn), row_scale


class _ColumnBlock:
    """Reuse one captured 128-column quantization update throughout a matrix."""
    def __init__(self, rows, block, levels, device, capture):
        self.work = torch.zeros((rows, block), device=device)
        self.inverse = torch.eye(block, device=device)
        self.scales = torch.ones((rows, block // w3.GROUP), device=device)
        self.errors = torch.empty_like(self.work)
        self.codes = torch.empty((rows, block), dtype=torch.int32, device=device)
        self.levels = torch.tensor(levels, device=device)
        self.midpoints = (self.levels[1:] + self.levels[:-1]) / 2
        self.graph = None
        if capture and device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self._run()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._run()

    def _run(self):
        for col in range(self.work.shape[1]):
            scale = self.scales[:, col // w3.GROUP]
            value = self.work[:, col].clone()
            code = torch.bucketize((value / scale.clamp_min(1e-30)).contiguous(), self.midpoints, out_int32=True)
            quantized = self.levels[code.long()] * scale
            error = (value - quantized) / self.inverse[col, col]
            self.codes[:, col].copy_(code)
            self.errors[:, col].copy_(error)
            self.work[:, col:].sub_(error[:, None] * self.inverse[col, col:][None, :])

    def __call__(self, weight, inverse, scales):
        self.work.copy_(weight)
        self.inverse.copy_(inverse)
        self.scales.copy_(scales)
        if self.graph is None:
            self._run()
        else:
            self.graph.replay()
        return self.codes, self.errors


@torch.no_grad()
def quantize(weight: torch.Tensor, covariance: torch.Tensor, *, damping: float = .01,
             codebook="lloyd", capture: bool = True, rows_per_chunk: int = 1024):
    """Calibrate a matrix, returning kernel-ready weights and a small report."""
    n, k = weight.shape
    if n % 128 or k % 128 or covariance.shape != (k, k):
        raise ValueError("calibration needs aligned [N,K] weights and [K,K] covariance")
    if not 0 < damping < 1 or rows_per_chunk % 128 or rows_per_chunk < 128:
        raise ValueError("invalid damping or row chunk size")
    if not torch.isfinite(weight).all() or not torch.isfinite(covariance).all():
        raise ValueError("calibration inputs must be finite")
    levels = w3._codebook(codebook)
    h = covariance.float().clone()
    diagonal = h.diagonal().clone()
    if diagonal.max() <= 0 or (diagonal < 0).any():
        raise ValueError("calibration covariance must contain positive input energy")
    h.diagonal().add_(damping * diagonal.mean().clamp_min(1e-12))
    chol = torch.linalg.cholesky(h)
    del h
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    del chol
    packed = []
    block = None
    for start in range(0, n, rows_per_chunk):
        original = weight[start:start + rows_per_chunk].float()
        scales, row_scale = fit_scales(original, diagonal, levels)
        effective = scales.float() * row_scale[:, None]
        work = original.clone()
        codes = torch.empty_like(work, dtype=torch.int32)
        if block is None or block.work.shape[0] != work.shape[0]:
            block = _ColumnBlock(work.shape[0], 128, levels, weight.device, capture)
        for col in range(0, k, 128):
            end = col + 128
            q, errors = block(work[:, col:end], inverse[col:end, col:end], effective[:, col // 32:end // 32])
            codes[:, col:end].copy_(q)
            if end < k:
                work[:, end:].sub_(errors @ inverse[col:end, end:])
        packed.append(w3.pack_codes(codes, scales, row_scale, levels))
        del work, codes
    result = w3.W3Weight(*(torch.cat([getattr(p, key) for p in packed]) for key in ("planes", "scales", "row_scale")),
                         n, k, levels, w3.lut_of(levels))
    report = {"method": "gptq", "damping": damping, "group_size": 32, "scale_grid": [.6, .7, .8, .9, 1.0], "bytes": result.bytes}
    return result, report
