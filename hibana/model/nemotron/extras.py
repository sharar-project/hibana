# SPDX-License-Identifier: Apache-2.0
"""One Mamba prefill plan per forward, carried through the model's side-input contract."""
from hibana.kernels.mamba2 import SideInputs


class NemotronExtras:
    variants = ("dense",)

    def variant(self, seqs):
        return "dense"

    def for_prefill(self, chunks):
        return SideInputs()

    def for_decode(self, reqs, seqs):
        return None

    def for_verify(self, reqs, drafts, seqs):
        return None

    def buffers(self, rows):
        return None

    def fill(self, side, reqs, drafts, seqs, rows, columns=None):
        """Decode and verify need no side buffers."""
