"""The draft a target serves: its pinned snapshot (beside the target or in hub/), or a trained checkpoint directory."""
import json

import pytest

from sharar.cache.spec_budget import SpecConfig
from sharar.proposers.factory import draft_snapshot


def _checkpoint(tmp_path, architecture="DFlash2DraftModel"):
    d = tmp_path / "best"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    (d / "model.safetensors").write_bytes(b"")
    return d


def test_spec_config_accepts_a_draft_dir_for_parallel_drafts_only(tmp_path):
    assert SpecConfig("dflash2", 15, "reconstruct", "fp8", str(tmp_path)).draft_dir == str(tmp_path)
    with pytest.raises(ValueError):
        SpecConfig("mtp", 3, "snapshots", "bf16", str(tmp_path))


def test_draft_snapshot_validates_the_directory(tmp_path):
    good = _checkpoint(tmp_path)
    assert draft_snapshot("dflash2", str(good)) == good
    with pytest.raises(ValueError):
        draft_snapshot("dflash2", str(_checkpoint(tmp_path / "other", "DSparkDraftModel")))  # another architecture
    with pytest.raises(FileNotFoundError):
        draft_snapshot("dflash2", str(tmp_path / "missing"))
    (good / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        draft_snapshot("dflash2", str(good))


def test_serve_cli_passes_the_draft_dir(monkeypatch, tmp_path):
    from sharar import cli

    calls = []
    monkeypatch.setattr("sharar.api.server.serve", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("SHARAR_DRAFT", raising=False)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert cli.main(["engine", "--spec", "15", "--draft", str(tmp_path)]) == 0
    assert calls[-1]["draft_dir"] == str(tmp_path)
    monkeypatch.setenv("SHARAR_DRAFT", str(tmp_path))
    assert cli.main(["engine", "--spec", "15"]) == 0
    assert calls[-1]["draft_dir"] == str(tmp_path)
    monkeypatch.delenv("SHARAR_DRAFT")
    assert cli.main(["engine", "--spec", "15"]) == 0 and calls[-1]["draft_dir"] is None
    with pytest.raises(SystemExit):
        cli.main(["engine", "--spec", "0", "--draft", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli.main(["engine", "--proposer", "mtp", "--spec", "3", "--draft", str(tmp_path)])


def test_each_target_serves_its_own_pinned_draft(monkeypatch, tmp_path):
    from sharar.model import qwen38_config
    from sharar.model.flashnext import config as flashnext
    from sharar.target import FLASHNEXT, QWEN38, Target
    assert QWEN38.draft_pin("dflash2") == qwen38_config.DRAFT_PINS["dflash2"]
    assert FLASHNEXT.draft_pin("dflash2") == flashnext.DRAFT_PINS["dflash2"] != QWEN38.draft_pin("dflash2")
    with pytest.raises(ValueError, match="no pinned mtp draft"):
        FLASHNEXT.draft_pin("mtp")
    monkeypatch.setattr(Target, "cache_root", lambda self, cache_dir=None: tmp_path)
    name, revision = FLASHNEXT.draft_pin("dflash2")
    repo = f"models--{name.replace('/', '--')}"
    assert draft_snapshot("dflash2", None, FLASHNEXT) == tmp_path / repo / "snapshots" / revision  # beside the target
    hub = tmp_path / "hub" / repo / "snapshots" / revision  # where the launcher downloads it
    hub.mkdir(parents=True)
    (hub / "config.json").write_text("{}")
    assert draft_snapshot("dflash2", None, FLASHNEXT) == hub
    # with no target snapshot cached at all: the draft is found from the cache root alone
    assert not (tmp_path / "models--target").exists()
    (hub / "config.json").unlink()
    root = tmp_path / repo / "snapshots" / revision
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    assert draft_snapshot("dflash2", None, FLASHNEXT) == root
