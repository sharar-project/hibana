"""How a projection's checkpoint tensors split across tensor-parallel ranks: the linear TP roles of SGLang
(mini-sglang `layers/linear.py`), applied when a format's loader reads its tensors.

- `REPLICATED` (`LinearReplicated`): every rank holds the whole tensor.
- `ColumnParallel(parts)` (`LinearColParallelMerged`, `LinearQKVMerged`): output rows split; the input is
  replicated and the output stays local (heads, MLP intermediate, vocabulary). `parts` are this rank's rows of
  each block fused along dim 0 (q|k|v, gate|up, ...): the checkpoint holds size x part rows per block and a rank
  keeps its own slice of each, so every rank keeps whole heads of each block.
- `ROW_PARALLEL` (`LinearRowParallel`, `LinearOProj`): input columns split; the local output is a partial sum the
  model all-reduces (`distributed.all_reduce`) where it rejoins the residual stream.
- `ExpertParallel(experts, rows_per_expert)`: a stack of `experts` matrices of `rows_per_expert` output rows each; a rank keeps a
  contiguous range of whole experts (the 3-bit routed experts: their down projection's 320 columns per rank would not
  fill the format's 128-wide chunks). The routed sum over a rank's experts is a partial sum, all-reduced as above.

`rows` and `cols` cut a tensor's output and input dimensions; calling a role cuts a [N, K]-shaped tensor in both.
That covers FP8 weights [N, K]; NVFP4 packed weights [N, K/2] and block scales [N, K/16] (shards split K on a
16-value block boundary, so a rank's bytes and scales are exactly its columns'); BF16 weights, convolution taps and
per-head vectors along dim 0. Packed layouts name their units: the 3-bit copies' fragment tensors index 32 rows
by dim 0 and a 128-wide K chunk by dim 1 (`w3_checkpoint.CalibratedWeights.load`). Per-tensor scales are replicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar.distributed import div_even, get_tp_info


class Shard:
    def rows(self, t: torch.Tensor, unit: int = 1) -> torch.Tensor:
        """This rank's output rows of `t`, whose dim 0 indexes `unit` rows each."""
        return t

    def cols(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """This rank's input columns of `t`, along `dim`."""
        return t

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.cols(self.rows(t))


REPLICATED = Shard()


@dataclass(frozen=True)
class ColumnParallel(Shard):
    parts: tuple[int, ...]  # this rank's rows of each fused block

    def rows(self, t: torch.Tensor, unit: int = 1) -> torch.Tensor:
        tp = get_tp_info()
        if tp.size == 1:
            return t
        local = [div_even(p, unit) for p in self.parts]
        blocks = t.split([p * tp.size for p in local])
        return torch.cat([b[tp.rank * p:(tp.rank + 1) * p] for b, p in zip(blocks, local)])


@dataclass(frozen=True)
class RowParallel(Shard):
    def cols(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        tp = get_tp_info()
        if tp.size == 1:
            return t
        k = div_even(t.shape[dim], tp.size)
        return t.narrow(dim, tp.rank * k, k)


ROW_PARALLEL = RowParallel()


def expert_range(experts: int) -> tuple[int, int]:
    """This rank's experts under expert parallelism: [lo, hi) of `experts`."""
    tp = get_tp_info()
    n = div_even(experts, tp.size)
    return tp.rank * n, (tp.rank + 1) * n


@dataclass(frozen=True)
class ExpertParallel(Shard):
    experts: int          # experts in the stack
    rows_per_expert: int  # output rows of each

    def rows(self, t: torch.Tensor, unit: int = 1) -> torch.Tensor:
        lo, hi = expert_range(self.experts)
        per = div_even(self.rows_per_expert, unit)
        return t[lo * per:hi * per]
