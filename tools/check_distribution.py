"""Build an sdist and its wheel, then check installed runtime assets offline.

Run in the prepared CPU environment. Temporary builds and installation do not
need a GPU, network or editable/source-tree imports in the installed probes.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def run(args, *, cwd, env=None):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='sharar-distribution-') as tmp:
        work = Path(tmp)
        source = work / 'source'
        source.mkdir()
        for name in ('pyproject.toml', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
            shutil.copy2(root / name, source / name)
        shutil.copytree(root / 'sharar', source / 'sharar',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'))
        artifacts = work / 'artifacts'
        artifacts.mkdir()
        run([sys.executable, '-c', 'from setuptools.build_meta import build_sdist; build_sdist("../artifacts")'], cwd=source)
        sdist = next(artifacts.glob('*.tar.gz'))
        with tarfile.open(sdist) as archive:
            archive.extractall(work / 'unpacked', filter='data')
        unpacked = next((work / 'unpacked').iterdir())
        run([sys.executable, '-c', 'from setuptools.build_meta import build_wheel; build_wheel("../../artifacts")'], cwd=unpacked)
        wheel = next(artifacts.glob('*.whl'))
        required = ['sharar/kernels/csrc/fp8_gemv.cu', 'sharar/benchmark_data/request_state_7aebbe6.txt']
        with zipfile.ZipFile(wheel) as archive:
            for name in required:
                assert archive.read(name) == (root / name).read_bytes(), f'missing or changed wheel asset: {name}'
        installed = work / 'installed'
        run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(installed), str(wheel)], cwd=work)
        env = {**os.environ, 'PYTHONPATH': str(installed), 'CUDA_VISIBLE_DEVICES': ''}
        run([sys.executable, '-c',
             'from sharar.kernels.fp8_gemv_cuda import _SRC; assert _SRC.is_file(); '
             'from sharar.benchmark_workloads import materialize; '
             'assert len(materialize(lambda text: list(text.encode()), "code", 10000)) == 10000'], cwd=work, env=env)
        run([sys.executable, '-m', 'sharar', 'benchmark', '--help'], cwd=work, env=env)
    print('Source distribution, wheel assets, installed CUDA loader and benchmark CLI passed.')


if __name__ == '__main__':
    main()
