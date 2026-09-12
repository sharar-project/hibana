"""Serving a trained draft checkpoint directory instead of the pinned snapshot."""
import json

import pytest

from sharar.memory.spec_budget import SpecConfig
from sharar.proposers.factory import draft_snapshot


def _checkpoint(tmp_path, architecture="DFlash2DraftModel"):
    d = tmp_path / "best"
    d.mkdir()
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
        draft_snapshot("dspark", str(good))  # architecture belongs to the other proposer
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
    assert cli.main(["serve", "--spec", "15", "--draft", str(tmp_path)]) == 0
    assert calls[-1]["draft_dir"] == str(tmp_path)
    monkeypatch.setenv("SHARAR_DRAFT", "/elsewhere")
    assert cli.main(["serve", "--spec", "15"]) == 0
    assert calls[-1]["draft_dir"] == "/elsewhere"
    monkeypatch.delenv("SHARAR_DRAFT")
    assert cli.main(["serve", "--spec", "15"]) == 0 and calls[-1]["draft_dir"] is None
    with pytest.raises(SystemExit):
        cli.main(["serve", "--spec", "0", "--draft", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli.main(["serve", "--proposer", "mtp", "--spec", "3", "--draft", str(tmp_path)])
