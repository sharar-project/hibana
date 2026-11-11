"""EngineConfig: one value from the environment, installed once, read at construction."""
import pytest
from pathlib import Path

from hibana.engine import config
from hibana.engine.config import EngineConfig


def test_defaults_match_the_old_module_defaults():
    c = EngineConfig.from_env({})
    assert (c.w3_decode, c.w3_checkpoint, c.w3_warning, c.stable_numerics, c.device_accept) == (False, None, None, False, False)
    assert (c.flashnext_trunk, c.draft_head, c.draft_head_rescore, c.draft_vocab) == ("fp8", "fp8", 8, 0)
    assert (c.accept_default, c.qsa_sparse_graphs, c.ssd_gib, c.traces_max_mib) == ("off", True, 200.0, 32768)
    assert not c.ssd and c.max_context is None and c.spec is None and c.cuda_arch == "12.1a"


def test_accept_soft_is_a_finite_number_at_least_zero():
    assert EngineConfig.from_env({}).accept_soft == 0.0 and EngineConfig.from_env({"SHARAR_ACCEPT_SOFT": "0.5"}).accept_soft == 0.5
    for bad in ("inf", "nan", "-1", "x"):
        with pytest.raises(ValueError, match="SHARAR_ACCEPT_SOFT must be a finite number >= 0"):
            EngineConfig.from_env({"SHARAR_ACCEPT_SOFT": bad})


def test_environment_names_are_unchanged():
    c = EngineConfig.from_env({"SHARAR_W3_DECODE": "1", "SHARAR_DRAFT_HEAD": "w3", "SHARAR_SPEC": "4", "SHARAR_QSA_SPARSE_GRAPHS": "0",
                               "SHARAR_SSD": "1", "SHARAR_MAX_CONTEXT": "0", "SHARAR_TRACES_MAX_MIB": "",
                               "SHARAR_FLASHNEXT_CACHE": "/fn", "HF_HOME": "/hf"})
    assert c.w3_decode and c.draft_head == "w3" and c.spec == "4" and not c.qsa_sparse_graphs
    assert c.ssd and c.max_context is None and c.traces_max_mib == 32768
    assert c.model_cache("SHARAR_FLASHNEXT_CACHE") == "/fn" and c.hf_cache == "/hf"
    assert EngineConfig.from_env({"SHARAR_HF_CACHE": "/a", "HF_HOME": "/b"}).hf_cache == "/a"


def test_install_and_using_scope_the_process_value(monkeypatch):
    monkeypatch.setenv("SHARAR_W3_DECODE", "1")
    assert config.current().w3_decode
    config.install(config.current().replace(w3_decode=False))
    monkeypatch.setenv("SHARAR_W3_DECODE", "1")
    assert not config.current().w3_decode  # installed values are not re-read from the environment
    with config.using(EngineConfig(w3_decode=True)):
        assert config.current().w3_decode
    assert not config.current().w3_decode


def test_config_is_frozen_and_describable():
    c = EngineConfig()
    with pytest.raises(Exception):
        c.w3_decode = True
    assert c.describe()["draft_head"] == "fp8" and len(c.describe()) == len(EngineConfig.__dataclass_fields__)


def test_gdn_backend_configuration():
    assert EngineConfig.from_env({}).gdn_prefill == "auto"
    assert EngineConfig.from_env({"SHARAR_GDN_PREFILL": "triton"}).gdn_prefill == "triton"


def test_qsa_prefill_is_opt_in():
    assert EngineConfig.from_env({}).qsa_prefill_backend == "original"


@pytest.mark.parametrize("backend", ["original", "auto", "optimized"])
def test_qsa_prefill_backend_configuration(backend):
    cfg = EngineConfig.from_env({"SHARAR_QSA_PREFILL_BACKEND": backend})
    assert cfg.qsa_prefill_backend == backend
    assert cfg.describe()["qsa_prefill_backend"] == backend
    assert "qsa_optimized_prefill" not in cfg.describe()


@pytest.mark.parametrize("backend", ["original", "auto", "optimized"])
@pytest.mark.parametrize("resident", ["auto", "0"])
def test_qsa_backend_preserves_ple_and_moe_settings(backend, resident):
    cfg = EngineConfig.from_env({"SHARAR_QSA_PREFILL_BACKEND": backend,
                                "SHARAR_PLE_RESIDENT": resident,
                                "SHARAR_MOE_KEEP": "4", "SHARAR_MOE_BUDGET": "12:64"})
    assert cfg.qsa_prefill_backend == backend
    assert cfg.ple_resident == resident
    assert cfg.describe()["qsa_prefill_backend"] == backend
    assert cfg.describe()["ple_resident"] == resident
    assert (cfg.moe_keep, cfg.moe_budget, cfg.moe_budget_per_request) == (4, 64, 12)
    assert (cfg.describe()["moe_keep"], cfg.describe()["moe_budget"], cfg.describe()["moe_budget_per_request"]) == (4, 64, 12)
    assert cfg.replace(qsa_prefill_backend="original").ple_resident == resident
    assert (cfg.replace(qsa_prefill_backend="original").moe_keep,
            cfg.replace(qsa_prefill_backend="original").moe_budget) == (4, 64)
    assert cfg.replace(ple_resident="auto").qsa_prefill_backend == backend


@pytest.mark.parametrize("backend", ["eager", "fused", "AUTO", ""])
def test_qsa_prefill_backend_rejects_invalid_values(backend):
    with pytest.raises(ValueError, match="SHARAR_QSA_PREFILL_BACKEND"):
        EngineConfig(qsa_prefill_backend=backend)
    with pytest.raises(ValueError, match="SHARAR_QSA_PREFILL_BACKEND"):
        EngineConfig().replace(qsa_prefill_backend=backend)
    if backend:
        with pytest.raises(ValueError, match="SHARAR_QSA_PREFILL_BACKEND"):
            EngineConfig.from_env({"SHARAR_QSA_PREFILL_BACKEND": backend})


def test_empty_qsa_prefill_environment_uses_default():
    assert EngineConfig.from_env({"SHARAR_QSA_PREFILL_BACKEND": ""}).qsa_prefill_backend == "original"


def test_no_module_reads_sharar_variables_at_import():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[2] / "hibana"
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in ("engine/config.py",) or rel.startswith(("kernels/fla/", "launch/")):
            continue  # launch/: the launcher's own settings (paths, peers, profiles), resolved before the engine starts
        if re.search(r"os\.environ[^\n]*SHARAR_|getenv\(\s*['\"]SHARAR_", path.read_text()):
            offenders.append(rel)
    assert offenders == []


def test_ssd_settings_parse_and_a_bad_block_size_fails_at_startup():
    cfg = EngineConfig.from_env({"SHARAR_SSD": "1", "SHARAR_SSD_BLOCK_TOKENS": "1024", "SHARAR_SSD_TRACE": "1"})
    assert cfg.ssd and cfg.ssd_block_tokens == 1024 and cfg.ssd_trace
    assert EngineConfig.from_env({"SHARAR_SSD_BLOCK_TOKENS": ""}).ssd_block_tokens == 256
    with pytest.raises(ValueError):
        EngineConfig.from_env({"SHARAR_SSD_BLOCK_TOKENS": "300"})
    cfg = EngineConfig.from_env({"SHARAR_SSD_WRITERS": "1", "SHARAR_SSD_STAGING_MIB": "2048", "SHARAR_SSD_CHUNK_MIB": "64"})
    assert cfg.ssd_writers == 1 and cfg.ssd_buffers == {"staging_bytes": 2048 << 20, "chunk_bytes": 64 << 20}
    with pytest.raises(ValueError):
        EngineConfig.from_env({"SHARAR_SSD_WRITERS": "0"})


def test_the_ssd_tier_stores_in_the_users_data_directory_unless_moved():
    default = str(Path.home() / ".local" / "share" / "sharar" / "ssd")  # the launcher passes $SHARAR_RESULTS/ssd/<target>
    assert EngineConfig.from_env({}).ssd_dir == EngineConfig().ssd_dir == default
    cfg = EngineConfig.from_env({"SHARAR_SSD_DIR": "/nvme/ssd", "SHARAR_SSD_GIB": "50"})
    assert cfg.ssd_dir == "/nvme/ssd" and cfg.ssd_gib == 50.0
