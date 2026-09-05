# SPDX-License-Identifier: Apache-2.0
"""Reconstruct accepted GDN state from compact verified recurrence inputs.

The recurrence and tile/reduction order match the vendored fused update.
All layers run in one launch. StateManager validates private slots and counts
before authorizing in-place reconstruction. Each program reads its entire
K-by-BV state tile before writing that same tile; distinct V tiles never alias.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct(h0, raw_k, raw_v, gates, betas, slots, counts, out,
                 STATE_LAYER: tl.constexpr, STATE_SLOT: tl.constexpr,
                 B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
                 K: tl.constexpr, V: tl.constexpr, S: tl.constexpr,
                 BK: tl.constexpr, BV: tl.constexpr, INPLACE: tl.constexpr):
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, hv = (lh // HV).to(tl.int64), lh % HV
    head = hv // (HV // H)
    slot = tl.load(slots + row).to(tl.int64)
    count = tl.load(counts + row)
    ok, ov = tl.arange(0, BK), iv * BV + tl.arange(0, BV)
    mask = (ok[:, None] < K) & (ov[None, :] < V)
    state = tl.load(h0 + layer * STATE_LAYER + slot * STATE_SLOT + hv * K * V
                    + ov[None, :] * K + ok[:, None], mask, 0).to(tl.float32)
    for t in range(count):
        k = tl.load(raw_k + ((layer * B + row) * H + head) * S * K + t * K + ok,
                    ok < K, 0).to(tl.float32)
        v = tl.load(raw_v + ((layer * B + row) * HV + hv) * S * V + t * V + ov,
                    ov < V, 0).to(tl.float32)
        g = tl.load(gates + ((layer * B + row) * HV + hv) * S + t)
        beta = tl.load(betas + ((layer * B + row) * HV + hv) * S + t)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        state *= tl.exp(g)
        v -= tl.sum(state * k[:, None], 0)
        v *= beta
        state += k[:, None] * v[None, :]
    if INPLACE:
        offset = layer * STATE_LAYER + slot * STATE_SLOT + hv * K * V
    else:
        offset = ((layer * B + row) * HV + hv) * K * V
    tl.store(out + offset + ov[None, :] * K + ok[:, None], state, mask)


def reconstruct(initial, raw_k, raw_v, gates, betas, slots, counts, out=None):
    """Validated unique slots only; optional separate output is the test oracle."""
    layers, capacity, hv, steps, v = raw_v.shape
    h, k = raw_k.shape[2], raw_k.shape[-1]
    _reconstruct[(triton.cdiv(v, 32), slots.numel(), layers * hv)](
        initial, raw_k, raw_v, gates, betas, slots, counts, initial if out is None else out,
        initial.stride(0), initial.stride(1), capacity, h, hv, k, v, steps,
        triton.next_power_of_2(k), min(triton.next_power_of_2(v), 32), out is None,
        num_warps=1, num_stages=3,
    )


class AcceptedState:
    """One bounded verify window, independent of cache capacity or proposer."""

    def __init__(self, pool, batch, steps, key_heads):
        layers, _, hv, v, k = pool.ssm_all.shape
        device = pool.ssm_all.device
        self.batch, self.steps = batch, steps
        self.raw_k = torch.empty((layers, batch, key_heads, steps, k), dtype=torch.bfloat16, device=device)
        self.raw_v = torch.empty((layers, batch, hv, steps, v), dtype=torch.bfloat16, device=device)
        self.gates = torch.empty((layers, batch, hv, steps), dtype=torch.float32, device=device)
        self.betas = torch.empty_like(self.gates)
        self.layer_inputs = {lid: dict(cache_ring=True, replayssm_rawk=self.raw_k[j],
                                     replayssm_rawv=self.raw_v[j], replayssm_g=self.gates[j],
                                     replayssm_beta=self.betas[j]) for j, lid in enumerate(pool.layer_ids)}

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.raw_k, self.raw_v, self.gates, self.betas))

    def run(self, pool, slots, counts):
        """Write only StateManager-authorized committed slots, on the caller's stream."""
        reconstruct(pool.ssm_all, self.raw_k, self.raw_v, self.gates, self.betas, slots, counts)
