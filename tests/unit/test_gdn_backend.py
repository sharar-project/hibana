import pytest
import torch

from hibana.engine.config import EngineConfig, using
from hibana.kernels.gdn_prefill import backend


@pytest.mark.parametrize("policy", ["auto", "triton"])
def test_cpu_uses_existing_backend(policy):
    q = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 6, 128, dtype=torch.bfloat16)
    state = torch.empty(1, 6, 128, 128)
    with using(EngineConfig(gdn_prefill=policy)):
        assert backend(q, q, v, state) == "triton"


@pytest.mark.parametrize("policy,match", [("cute", "SM120/121"), ("invalid", "invalid")])
def test_forced_unsupported_or_invalid_backend_fails(policy, match):
    q = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 6, 128, dtype=torch.bfloat16)
    with using(EngineConfig(gdn_prefill=policy)):
        with pytest.raises(ValueError, match=match):
            backend(q, q, v, torch.empty(1, 6, 128, 128))
