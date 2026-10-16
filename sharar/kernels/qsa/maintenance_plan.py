# SPDX-License-Identifier: Apache-2.0
"""Completed-group metadata, independent of QSA layer weights and cache values."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressionPlan:
    members: torch.Tensor
    writes: torch.Tensor
    positions: torch.Tensor

    @property
    def groups(self):
        return self.members.shape[0]


def make_compression_plan(seqs, device):
    """Build once per forward from host-known (prefix, new_count, pages).

    Pages must be unique valid physical pages of a common pool, excluding its
    dummy page. Values are deliberately not copied to the host or synchronized
    here. Graph replay can update the fixed-shape tensors in place.
    """
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    members, positions = [], []
    back = torch.arange(3, -1, -1, device=device, dtype=torch.int64)
    for first, count, pages in seqs:
        if not isinstance(first, int) or not isinstance(count, int) or first < 0 or count < 0:
            raise ValueError('prefix/new_count must be nonnegative host integers')
        if pages.device != device or pages.dtype != torch.int32 or pages.ndim != 1:
            raise ValueError('pages must be a one-dimensional int32 tensor on the plan device')
        if len(pages) < first + count:
            raise ValueError('page table does not cover prefix plus new tokens')
        end0 = (first//4)*4+3
        if end0 >= first+count:
            continue
        ends = torch.arange(end0, first+count, 4, device=device, dtype=torch.int64)
        members.append(pages[ends[:, None]-back])
        positions.append(ends-3)
    if not members:
        return CompressionPlan(torch.empty((0, 4), device=device, dtype=torch.int32),
                               torch.empty(0, device=device, dtype=torch.int32),
                               torch.empty(0, device=device, dtype=torch.int64))
    all_members = torch.cat(members).contiguous()
    return CompressionPlan(all_members, all_members[:, -1].contiguous(), torch.cat(positions).contiguous())
