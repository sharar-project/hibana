"""EngineConfig: one value from the environment, installed once, read at construction."""
import pytest
from pathlib import Path

from sharar.engine import config
from sharar.engine.config import EngineConfig


def test_defaults_match_the_old_module_defaults():
    c = EngineConfig.from_env({})
    assert (c.w3_decode, c.w3_checkpoint, c.w3_warning, c.stable_numerics, c.device_accept) == (False, None, None, False, False)
    assert (c.flashnext_trunk, c.flashnext_mixers, c.draft_head, c.draft_head_rescore, c.draft_vocab) == ("fp8", "fp8", "fp8", 8, 0)
    assert (c.accept_default, c.qsa_sparse_graphs, c.ssd_gib, c.traces_max_mib) == ("2:1.0", True, 200.0, 32768)
    assert not c.ssd and c.max_context is None and c.spec is None and c.cuda_arch == "12.1a"


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


def test_no_module_reads_sharar_variables_at_import():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[2] / "sharar"
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
