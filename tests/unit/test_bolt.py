"""Bolt is a separate pin with Puzzle experts and the shared Mamba lifecycle."""

import json
from pathlib import Path

import pytest

from sharar.launch import preflight, settings
from sharar.model.bolt import config
from sharar.target import NEMOTRON, NEMOTRON_BOLT, get

CONFIG = Path(__file__).resolve().parents[1] / "data" / "nemotron_bolt_config.json"


def test_puzzle_geometry_and_shared_recurrence(tmp_path):
    cfg = config.load_config(CONFIG)
    target = get("bolt")
    assert target is NEMOTRON_BOLT and target.family == "nemotron"
    assert len(cfg.blocks) == target.num_layers == 88
    assert len(target.gdn_layer_ids) == len(target.moe_block_specs) == 40
    assert len(target.attention_layer_ids) == 8
    assert target.state_bytes_per_slot == NEMOTRON.state_bytes_per_slot == cfg.state_bytes_per_slot()
    assert target.mtp_layers == 0 and target.num_experts == 0
    assert {e for _, e, _, _ in target.moe_block_specs} == {240, 304, 368, 432, 496}
    assert {k for _, _, _, k in target.moe_block_specs} == {9}
    (tmp_path / "config.json").write_text(CONFIG.read_text())
    assert target.layer_types_of(target.load_model_config(tmp_path)) == list(config.LAYER_TYPES)


@pytest.mark.parametrize("field,value", [("n_routed_experts", 512), ("num_experts_per_tok", 22), ("moe_intermediate_size", 2688)])
def test_block_metadata_takes_precedence_and_is_validated(tmp_path, field, value):
    raw = json.loads(CONFIG.read_text())
    raw["llm_config"]["block_configs"][1][field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="block 1"):
        NEMOTRON_BOLT.load_model_config(tmp_path)


@pytest.mark.parametrize("profile", settings.PROFILE_NAMES)
@pytest.mark.parametrize("name", ["bolt", "nemotron-bolt"])
def test_profiles_keep_native_weights_all_experts_and_no_mtp(tmp_path, profile, name):
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    s = settings.resolve(name, profile, env=env)
    assert (s.target, s.port, s.service, s.spec) == ("bolt", 30006, "sharar-bolt", 0)
    assert s.get("SHARAR_W3_DECODE") == s.get("SHARAR_MOE_KEEP") == s.get("SHARAR_DEVICE_ACCEPT") == "0"
    assert s.get("SHARAR_SSM_STATE") == "fp32"
    assert not s.parallel_draft
    assert settings.model_config(s.target) is config
    assert preflight.pinned_snapshots(s) == [(s.hf_cache, config.MODEL_ID, config.MODEL_REVISION)]
    assert preflight.ensure_w3(s, fetch=False) is None


def test_bolt_rejects_unsupported_drafter_and_parallelism(tmp_path):
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    with pytest.raises(ValueError, match="no in-checkpoint MTP"):
        settings.resolve("bolt", env=env, overrides={"SHARAR_SPEC": "3"})
    with pytest.raises(ValueError, match="one Spark"):
        settings.resolve("bolt", env=env, tp=2)


def test_bolt_accepts_mamba_state_storage(tmp_path):
    from sharar.cache.ssm_format import SsmFormat
    for kind in ("fp32", "fp16", "fp16-rn"):
        SsmFormat(kind).check(NEMOTRON_BOLT)
    with pytest.raises(ValueError, match="GDN format"):
        SsmFormat("int8").check(NEMOTRON_BOLT)


def test_torch_reference_parses_puzzle_and_has_no_mtp(tmp_path):
    from sharar.model.nemotron.reference import NemotronConfig
    (tmp_path / "config.json").write_text(CONFIG.read_text())
    cfg = NemotronConfig.from_snapshot(tmp_path)
    assert cfg.time_step_limit == (0., float("inf"))
    assert cfg.mtp_block_types == () and cfg.top_k == 9
    assert cfg.mamba_inner == 8192 and cfg.conv_dim == 10240
