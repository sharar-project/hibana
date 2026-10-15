"""Fetching a target's published calibrated 3-bit copies on first serve (weights/w3_checkpoint.fetch)."""
import json
import sys
import types
from dataclasses import replace

import pytest

from sharar.target import FLASHNEXT, QWEN38
from sharar.weights import w3_checkpoint


def publish(directory, target, corrupt=False):
    """A checkpoint as the Hub serves it: manifest.json and one file per weight."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.safetensors").write_bytes(b"weights")
    digest = w3_checkpoint.sha256(directory / "a.safetensors")
    if corrupt:
        (directory / "a.safetensors").write_bytes(b"tampered")
    manifest = {"format": w3_checkpoint.FORMAT_VERSION, "target": target.name, "target_revision": target.model_revision,
                "weights": {"layers.0.mixer.qkv": {"file": "a.safetensors", "sha256": digest}}}
    (directory / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def hub(monkeypatch):
    """A stand-in huggingface_hub whose snapshot_download publishes into local_dir and records the call."""
    calls = []
    state = {"corrupt": False}

    def snapshot_download(repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision))
        publish(local_dir, FLASHNEXT, state["corrupt"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls, state


def test_fetch_downloads_the_pinned_revision_and_links_current(tmp_path, hub):
    calls, _ = hub
    dest = w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=lambda *_: None)
    assert calls == [(FLASHNEXT.w3_repo, FLASHNEXT.w3_revision)]
    assert dest == tmp_path / f"flashnext-{FLASHNEXT.w3_revision[:12]}"
    current = tmp_path / "flashnext-current"
    assert current.is_symlink() and current.resolve() == dest.resolve()
    assert w3_checkpoint.CalibratedWeights(current, revision=FLASHNEXT.model_revision).weights
    w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=lambda *_: None)
    assert len(calls) == 1  # a finished download is not fetched again


def test_fetch_keeps_a_local_calibration_and_says_how_to_switch(tmp_path, hub):
    calls, _ = hub
    local = tmp_path / "20260929-flashnext-gptq"
    publish(local, FLASHNEXT)
    (tmp_path / "flashnext-current").symlink_to(local.name)
    notes = []
    assert w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=notes.append) == local.resolve()
    assert (tmp_path / "flashnext-current").resolve() == local.resolve() and calls == []  # nothing downloaded
    assert len(notes) == 1 and "local calibration" in notes[0] and "--switch" in notes[0]
    dest = w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=lambda *_: None, switch=True)
    assert (tmp_path / "flashnext-current").resolve() == dest.resolve() and local.is_dir()  # the local copy is kept


def test_fetch_moves_current_off_an_earlier_fetched_revision(tmp_path, hub):
    """A new pinned Flash-Next checkpoint: machines with the old one linked must follow the pin."""
    calls, _ = hub
    old = tmp_path / w3_checkpoint.fetched_name(FLASHNEXT, "970cbfc0cbc174d2f59d902eb4af0af07ece1564")
    publish(old, FLASHNEXT)
    (tmp_path / "flashnext-current").symlink_to(old.name)
    notes = []
    dest = w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=notes.append)
    assert calls == [(FLASHNEXT.w3_repo, FLASHNEXT.w3_revision)]
    assert (tmp_path / "flashnext-current").resolve() == dest.resolve() and old.is_dir()
    assert any(old.name in n and dest.name in n for n in notes)
    assert not (tmp_path / ".flashnext-current.tmp").exists()


def test_fetch_replaces_a_dangling_link(tmp_path, hub):
    (tmp_path / "flashnext-current").symlink_to("gone")
    dest = w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=lambda *_: None)
    assert (tmp_path / "flashnext-current").resolve() == dest.resolve()


def test_fetched_names_are_told_from_local_calibrations():
    assert w3_checkpoint.is_fetched(FLASHNEXT, w3_checkpoint.fetched_name(FLASHNEXT, FLASHNEXT.w3_revision))
    for name in ("20260929-flashnext-gptq", "flashnext-current", "qwen38-694e2790772b", "flashnext-694e2790772"):
        assert not w3_checkpoint.is_fetched(FLASHNEXT, name)


def test_fetch_rejects_a_corrupt_download_and_links_nothing(tmp_path, hub):
    _, state = hub
    state["corrupt"] = True
    with pytest.raises(ValueError, match="checksum"):
        w3_checkpoint.fetch(FLASHNEXT, tmp_path, log=lambda *_: None)
    assert not (tmp_path / "flashnext-current").exists()
    assert not (tmp_path / f"flashnext-{FLASHNEXT.w3_revision[:12]}").exists()


def test_fetch_rejects_a_checkpoint_for_another_revision(tmp_path, hub):
    with pytest.raises(ValueError, match="target revision"):
        w3_checkpoint.fetch(replace(FLASHNEXT, model_revision="0" * 40), tmp_path, log=lambda *_: None)


def test_a_target_without_a_published_checkpoint_says_so(tmp_path):
    with pytest.raises(LookupError, match="calibrate"):
        w3_checkpoint.fetch(replace(QWEN38, w3_repo=None, w3_revision=None), tmp_path)


@pytest.mark.parametrize("target", [QWEN38, FLASHNEXT])
def test_both_targets_pin_their_checkpoint_by_revision(target):
    assert target.w3_repo and len(target.w3_revision) == 40
