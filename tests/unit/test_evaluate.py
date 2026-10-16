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


def test_serves_this_checkout_as_a_user_would():
    served = evaluate.Served("evaluate", Path("/tmp/sharar-evaluate-test"), "flashnext", "fast", {}, 30200, "python", Path("."), {})
    assert served.command[3:] == ["sharar", "serve", "flashnext", "--fast", "--port", "30200"]
    assert "--no-fetch" not in served.command  # the build as served: the 3-bit checkpoint follows the pin
