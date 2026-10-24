"""tools/h2h-pr.py's configuration arms (`--arm NAME=PROFILE[:KEY=VALUE,...]`): one checkout, several settings."""
import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "h2h-pr.py"


@pytest.fixture(scope="module")
def h2h():
    loader = SourceFileLoader("h2h_pr", str(TOOL))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("h2h_pr", loader))
    loader.exec_module(module)
    return module


def test_arms_keep_their_order_and_values_with_colons_and_commas(h2h):
    arms = h2h.parse_arms(["exact=exact:SHARAR_DRAFT_HEAD=fp8",
                           "deep=fast:SHARAR_SPEC=5,SHARAR_SPEC_DEPTHS=2:4,4:4,SHARAR_ACCEPT_DEFAULT=3:2.0", "fast=fast"],
                          argparse.ArgumentParser())
    assert list(arms) == ["exact", "deep", "fast"]
    assert arms["deep"] == ("fast", {"SHARAR_SPEC": "5", "SHARAR_SPEC_DEPTHS": "2:4,4:4", "SHARAR_ACCEPT_DEFAULT": "3:2.0"})
    assert arms["fast"] == ("fast", {}) and arms["exact"] == ("exact", {"SHARAR_DRAFT_HEAD": "fp8"})


@pytest.mark.parametrize("bad", ["a-b=fast", "x=turbo", "x=fast:SPEC=3"])
def test_bad_arms_are_refused(h2h, bad):
    with pytest.raises(SystemExit):
        h2h.parse_arms([bad], argparse.ArgumentParser())


def test_duplicate_names_are_refused(h2h):
    with pytest.raises(SystemExit):
        h2h.parse_arms(["a=fast", "a=exact"], argparse.ArgumentParser())
