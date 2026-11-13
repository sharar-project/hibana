"""Verifier dispatch must leave ordinary decode, wider batches and vocab alone."""
import pytest

from hibana.kernels.verify_gemm import is_attention_qkv, is_mlp


@pytest.mark.parametrize('m', [1, 2, 3, 4, 8, 15, 17, 32, 48, 64, 2048])
def test_other_row_counts_keep_reference_dispatch(m):
    assert not is_attention_qkv(m, 14336, 5120)
    assert not is_mlp(m, 34816, 5120)
    assert not is_mlp(m, 5120, 17408)


def test_only_validated_target_projections_are_selected():
    assert is_attention_qkv(16, 14336, 5120)
    assert is_mlp(16, 34816, 5120)
    assert is_mlp(16, 5120, 17408)
    for n, k in [(10240, 5120), (6144, 5120), (5120, 6144), (248320, 5120), (4096, 5120)]:
        assert not is_attention_qkv(16, n, k)
        assert not is_mlp(16, n, k)
