"""The recurrent-state formats (cache/ssm_format.py): sizes, views and slot semantics without a GPU."""
from math import prod

import pytest
import torch

from sharar.cache.ssm_format import FP32, SsmFormat, WindowLayout
from sharar.cache.state_pool import StatePool
from sharar.engine.config import EngineConfig
from sharar.target import DEFAULT_TARGET, FLASHNEXT, NEMOTRON


def test_fp32_is_the_targets_state():
    t = DEFAULT_TARGET
    assert not FP32.quantized and FP32.layout(t) is None and FP32.describe() == "fp32"
    assert FP32.state_bytes_per_slot(t) == t.state_bytes_per_slot


def test_int8_layout_regions_are_disjoint_aligned_and_sized():
    t = DEFAULT_TARGET
    fmt = SsmFormat("int8", 16, 4)
    lay = fmt.layout(t)
    offsets = lay.offsets
    spans = sorted((offsets[name], offsets[name] + prod(shape) * dtype.itemsize) for name, dtype, shape in lay.regions)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert all(off % 256 == 0 for off in offsets.values()) and lay.nbytes % 256 == 0 and spans[-1][1] <= lay.nbytes
    # INT8 base, its scales and compensators: ~1.19 bytes per state element (the paper's figure), plus the window.
    h, hv, k, v = t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim
    base = hv * v * k + 4 * hv * (v + k) + 2 * 4 * hv * (k + v)
    assert base / (hv * v * k) == pytest.approx(1.19, abs=0.01)
    assert lay.nbytes >= base + 2 * 16 * (h * k + hv * v)
    # The 27B's slot, conv window included: about 147 MiB in FP32, 58 MiB in INT8.
    assert fmt.state_bytes_per_slot(t) < 0.4 * FP32.state_bytes_per_slot(t)


def test_int4_packs_two_values_per_byte_with_eight_compensators():
    t = DEFAULT_TARGET
    fmt = SsmFormat("int4")
    assert fmt.rank == 8 and fmt.describe() == "int4:w16:r8"  # the paper's r=8 at 4 bits
    lay = fmt.layout(t)
    z = dict((name, (dtype, shape)) for name, dtype, shape in lay.regions)["z"]
    assert z == (torch.uint8, (t.linear_num_value_heads, t.linear_value_head_dim, t.linear_key_head_dim // 2))
    hv, k, v = t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim
    assert SsmFormat("int8").layout(t).nbytes - lay.nbytes == hv * v * k // 2 - 4 * hv * (k + v) * 2  # half the payload, 4 more compensators
    # The FP16 window records and compensators do not shrink: the 27B's slot is ~44 MiB (INT8 ~58, FP32 ~147).
    assert fmt.state_bytes_per_slot(t) < 0.8 * SsmFormat("int8").state_bytes_per_slot(t) < 0.32 * FP32.state_bytes_per_slot(t)


def test_views_alias_one_row_per_layer_and_slot():
    lay = WindowLayout(2, 4, 16, 16, 8, 2)
    raw = torch.zeros((3, 5, lay.nbytes), dtype=torch.uint8)
    views = lay.views(raw)
    assert views["z"].shape == (3, 5, 4, 16, 16) and views["z"].dtype == torch.int8
    assert views["log_k"].shape == (3, 5, 2, 8, 16) and views["log_len"].shape == (3, 5, 4)
    views["scale"][1, 2, 3, 4] = 1.5
    views["log_len"][2, 4, 0] = 7
    touched = (raw != 0).nonzero()[:, :2].unique(dim=0).tolist()
    assert touched == [[1, 2], [2, 4]]  # each write lands in its own (layer, slot) row
    with pytest.raises(ValueError):
        lay.views(raw.view(torch.int8))


def test_settings_and_validation():
    c = EngineConfig.from_env({"SHARAR_SSM_STATE": "int8", "SHARAR_SSM_WINDOW": "32", "SHARAR_SSM_COMPENSATORS": "8"})
    fmt = SsmFormat.from_config(c)
    assert fmt == SsmFormat("int8", 32, 8) and fmt.describe() == "int8:w32:r8"
    assert SsmFormat.from_config(EngineConfig.from_env({})) == FP32
    assert SsmFormat.from_config(EngineConfig.from_env({"SHARAR_SSM_STATE": "int4"})) == SsmFormat("int4", 16, 8)
    for bad in (dict(kind="int2"), dict(kind="int8", window=12), dict(kind="int8", rank=17)):
        with pytest.raises(ValueError):
            SsmFormat(**bad)


def test_quantized_pool_slots_clear_and_copy_whole_rows():
    fmt = SsmFormat("int8", 8, 2)
    pool = StatePool(4, torch.device("cpu"), target=DEFAULT_TARGET, ssm_format=fmt)
    assert pool.ssm_all.dtype == torch.uint8 and pool.window is not None
    assert pool.bytes_per_slot == fmt.state_bytes_per_slot(DEFAULT_TARGET)
    lid = pool.layer_ids[3]
    views = pool.window.layer[lid]
    views["z"][1].fill_(5)
    views["log_len"][1].fill_(3)
    views["comp_u"][1].fill_(0.25)
    pool.copy_slot(1, 2)
    assert torch.equal(pool.ssm_all[:, 2], pool.ssm_all[:, 1]) and int(views["log_len"][2][0]) == 3
    slot = pool.alloc()  # a fresh slot is the zero state: empty window, zero boundary
    assert not pool.ssm_all[:, slot].any()
    pool.clear(1)
    assert not pool.ssm_all[:, 1].any() and pool.window.single[lid]["z"].shape[0] == 1
    assert StatePool(2, torch.device("cpu")).window is None


def test_spec_decoder_requires_the_ring_for_a_quantized_pool():
    from types import SimpleNamespace
    from sharar.engine.spec import SpecDecoder
    pool = SimpleNamespace(window=object(), ssm_all=torch.zeros(1, 1, 1), conv_all=torch.zeros(1, 1, 1, 1), layer_ids=[0])
    ex = SimpleNamespace(state_pool=pool, max_batch=1, device=torch.device("cpu"), target=DEFAULT_TARGET)
    proposer = SimpleNamespace(k=3)
    with pytest.raises(ValueError, match="reconstruct"):
        SpecDecoder(ex, proposer, "snapshots")


def test_fp16_is_nemotrons_and_halves_its_state():
    t = NEMOTRON
    sr = SsmFormat("fp16_sr")
    assert not sr.quantized and sr.layout(t) is None
    assert (sr.describe(), FP32.describe()) == ("fp16_sr", "fp32")
    assert sr.dtype == torch.float16 and FP32.dtype == torch.float32 and SsmFormat("int8").dtype is None
    conv = t.conv_dim * t.conv_state_len * 2 * len(t.gdn_layer_ids)
    assert sr.state_bytes_per_slot(t) - conv == (FP32.state_bytes_per_slot(t) - conv) // 2
    assert sr.state_bytes_per_slot(t) / 2 ** 20 == pytest.approx(82.3, abs=0.1)  # 162.3 MiB in FP32
    pool = StatePool(3, torch.device("cpu"), target=t, ssm_format=sr)
    assert pool.ssm_all.dtype == torch.float16 and pool.window is None and pool.bytes_per_slot == sr.state_bytes_per_slot(t)
    assert SsmFormat.from_config(EngineConfig.from_env({"SHARAR_SSM_STATE": "fp16_sr"})) == sr
    for old in ("fp16", "fp16-rn"):  # no other spelling
        with pytest.raises(ValueError, match="fp16_sr"):
            SsmFormat(old)


def test_formats_stay_on_their_recurrence():
    """FP16 is Mamba-2's (the GDN targets keep their FP32 pools, bit-identical), the windows the GDN's
    (`Target.recurrence`, the launcher's table of the same name)."""
    from sharar.launch.settings import TARGET_SETTINGS
    from sharar.target import TARGETS
    for gdn in (DEFAULT_TARGET, FLASHNEXT):
        with pytest.raises(ValueError, match="gdn state is stored as fp32, int8 or int4"):
            SsmFormat("fp16_sr").state_bytes_per_slot(gdn)
        with pytest.raises(ValueError, match="gdn state"):
            StatePool(2, torch.device("cpu"), target=gdn, ssm_format=SsmFormat("fp16_sr"))
        assert StatePool(2, torch.device("cpu"), target=gdn).ssm_all.dtype == torch.float32
    for kind in ("int8", "int4"):
        with pytest.raises(ValueError, match="mamba2 state is stored as fp32 or fp16_sr"):
            SsmFormat(kind).state_bytes_per_slot(NEMOTRON)
    assert {t.recurrence for t in TARGETS.values()} == {"gdn", "mamba2"}
    for name, t in TARGETS.items():
        assert TARGET_SETTINGS[name].get("recurrence", "gdn") == t.recurrence
