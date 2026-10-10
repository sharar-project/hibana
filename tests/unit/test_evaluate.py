"""tools/evaluate.py: argument checks before anything is served."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
loader = SourceFileLoader("evaluate_tool", str(REPO / "tools" / "evaluate.py"))
evaluate = importlib.util.module_from_spec(importlib.util.spec_from_loader("evaluate_tool", loader))
loader.exec_module(evaluate)


@pytest.mark.parametrize("argv, message", [
    (["--target", "flashnext"], "nothing to run"),
    (["--target", "flashnext", "--suite", "quick", "--port", "30100"], "30200-30299"),
    (["--target", "flashnext", "--suite", "quick", "--name", "a/b"], "--name"),
])
def test_rejects_before_serving(argv, message, capsys):
    with pytest.raises(SystemExit):
        evaluate.main(argv)
    assert message in capsys.readouterr().err


def test_serves_under_its_own_service_name():
    assert evaluate.gate.SERVICE == "sharar-evaluate"
    conf = evaluate.gate.served_config({}, "fast", {}, 30200)
    assert "SHARAR_SERVICE=sharar-evaluate" in conf and "SHARAR_PORT=30200" in conf and "SHARAR_VENV=" in conf
