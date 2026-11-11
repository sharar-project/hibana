import pytest
import torch

from hibana.cache.spec_budget import SpecConfig
from hibana.model.draft_linear import DraftFp8Linear


def test_precision_contract_rejects_unsupported_proposers_and_formats():
    with pytest.raises(ValueError, match='parallel draft'):
        SpecConfig('mtp', 3, 'reconstruct', 'fp8')
    with pytest.raises(ValueError):
        SpecConfig('dflash2', 7, 'reconstruct', 'fp4')
    assert SpecConfig('dflash2', 15, 'reconstruct', 'fp8').draft_precision == 'fp8'


def test_packed_weight_owns_smaller_storage_and_zero_weights_have_finite_scale():
    w = torch.zeros(128, 128, dtype=torch.bfloat16)
    linear = DraftFp8Linear.from_bf16(w)
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.bytes == w.numel() + 4
    assert linear.weight.data_ptr() != w.data_ptr()
    assert torch.isfinite(linear.scale).all() and (linear.scale > 0).all()
    with pytest.raises(ValueError):
        DraftFp8Linear.from_bf16(w.float())
