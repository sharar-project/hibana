import subprocess
import sys


def test_sharar_never_imports_sglang():
    code = (
        "import sys, importlib; "
        "mods = ['sharar', 'sharar.config', 'sharar.cli', 'sharar.parity.prompts', "
        "'sharar.weights.modelopt']; "
        "[importlib.import_module(m) for m in mods]; "
        "print(any(m.split('.')[0] == 'sglang' for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_no_sglang_imports_in_source():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = []
    for path in list((root / "sharar").rglob("*.py")) + list((root / "tests").rglob("*.py")) + list((root / "bench").rglob("*.py")):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import sglang", "from sglang")):
                offenders.append(f"{path}: {stripped}")
    assert not offenders, offenders
