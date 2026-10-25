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
def test_profiles_keep_native_weights_and_use_pinned_lilicorr(tmp_path, profile, name):
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    s = settings.resolve(name, profile, env=env)
    assert (s.target, s.port, s.service, s.spec) == ("bolt", 30006, "sharar-bolt", 7)
    assert s.get("SHARAR_W3_DECODE") == s.get("SHARAR_MOE_KEEP") == s.get("SHARAR_DEVICE_ACCEPT") == "0"
    assert s.get("SHARAR_SSM_STATE") == "fp32"
    assert s.get("SHARAR_BOLT_EXPERTS") == "b12x"
    assert s.parallel_draft and s.proposer == 'dflash2'
    assert settings.model_config(s.target) is config
    assert preflight.pinned_snapshots(s) == [(s.hf_cache, config.MODEL_ID, config.MODEL_REVISION),
                                             (s.hf_cache, *config.DRAFT_PINS['dflash2'])]
    assert preflight.ensure_w3(s, fetch=False) is None


@pytest.mark.parametrize("backend", ["bf16", "fp8", "b12x"])
def test_expert_backend_override_and_engine_report(tmp_path, backend):
    from sharar.engine.config import EngineConfig
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    s = settings.resolve("bolt", "fast", env=env, overrides={"SHARAR_BOLT_EXPERTS": backend})
    assert EngineConfig.from_env(s.values).describe()["bolt_experts"] == backend
    for value in ("auto", "iq2", "FP8"):
        with pytest.raises(ValueError, match="SHARAR_BOLT_EXPERTS"):
            settings.resolve("bolt", env=env, overrides={"SHARAR_BOLT_EXPERTS": value})
        with pytest.raises(ValueError, match="SHARAR_BOLT_EXPERTS"):
            EngineConfig(bolt_experts=value)


def test_ssd_identity_separates_bf16_and_fp8_experts():
    from types import SimpleNamespace as NS
    from sharar.cache.ssd_store import root_hash
    from sharar.engine.executor import Executor
    ex = Executor.__new__(Executor)
    ex.target = NEMOTRON_BOLT
    ex.state_pool = NS(bytes_per_slot=NEMOTRON_BOLT.state_bytes_per_slot, ssm_format=NS(kind="fp32"))
    ex.kv_pool = NS(nvfp4=False)
    ex.model = NS(arithmetic_identity=None)
    bf16 = ex._cache_identity()
    assert "arithmetic" not in bf16
    ex.model.arithmetic_identity = {"iq2_experts": "fp8", "kernel": "revision-a"}
    fp8 = ex._cache_identity()
    assert fp8["arithmetic"]["iq2_experts"] == "fp8"
    assert root_hash(bf16) != root_hash(fp8)
    ex.model.arithmetic_identity = {"iq2_experts": "fp8", "kernel": "revision-b"}
    assert root_hash(fp8) != root_hash(ex._cache_identity())


def test_bolt_rejects_unsupported_drafter_and_parallelism(tmp_path):
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    with pytest.raises(ValueError, match="no in-checkpoint MTP"):
        settings.resolve("bolt", env=env, overrides={"SHARAR_SPEC": "3", "SHARAR_PROPOSER": "mtp"})
    with pytest.raises(ValueError, match="one Spark"):
        settings.resolve("bolt", env=env, tp=2)


def test_bolt_lilicorr_depth_limits_and_plain_decode(tmp_path):
    env = {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none")}
    for depth in (1, 3, 7):
        assert settings.resolve('bolt', env=env, overrides={'SHARAR_SPEC': str(depth)}).spec == depth
    with pytest.raises(ValueError, match='depth 1..7'):
        settings.resolve('bolt', env=env, overrides={'SHARAR_SPEC': '8'})
    with pytest.raises(ValueError, match='published W4A16/BF16'):
        settings.resolve('bolt', env=env, overrides={'SHARAR_DRAFT_PRECISION': 'fp8'})
    plain = settings.resolve('bolt', env=env, overrides={'SHARAR_SPEC': '0'})
    assert not plain.parallel_draft
    assert preflight.pinned_snapshots(plain) == [(plain.hf_cache, config.MODEL_ID, config.MODEL_REVISION)]


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


def test_plan_rows_follow_the_graphs():
    """b12x plans tune the rows the CUDA graphs run: batch buckets times one or a speculative round."""
    from types import SimpleNamespace

    from sharar.model.bolt.linear import graph_rows
    assert graph_rows(SimpleNamespace(max_batch=4, spec="7", spec_depths=None)) == (1, 2, 4, 7, 8, 14, 16, 28, 32)
    assert graph_rows(SimpleNamespace(max_batch=1, spec="auto", spec_depths="")) == tuple(range(1, 10))
    assert 6 in graph_rows(SimpleNamespace(max_batch=2, spec="7", spec_depths="2:2"))


def test_w3_calibration_names_cover_the_trunk_not_the_shared_experts():
    from sharar.weights import bolt_w3_calibrate as calibrate
    names = calibrate.projection_names(NEMOTRON_BOLT)
    assert len(names) == 40 * 2 + 40 * 2 + 8 * 2 + 1 and names[-1] == "lm_head"
    assert not any("shared" in n for n in names)
    attention = next(i for i, k in enumerate(NEMOTRON_BOLT.layer_types) if k == "full_attention")
    assert calibrate.source_prefixes(f"layers.{attention}.mixer.qkv") == [
        f"language_model.backbone.layers.{attention}.mixer.{p}_proj" for p in "qkv"]
    with pytest.raises(ValueError):
        calibrate.source_prefixes("layers.1.mixer.shared_up")
