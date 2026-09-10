"""Documentation moves cannot silently remove manifests from validation."""
import json

import pytest

from tools.check_evidence import check


def fixture(root):
    folder = root / 'docs' / 'archive'
    folder.mkdir(parents=True)
    (folder / 'study.md').write_text('# Study\n')
    manifest = folder / 'study.json'  # deliberately no *results.json suffix
    manifest.write_text(json.dumps({'schema': 2, 'summary_document': 'study.md',
        'full_evidence': {'bytes': 100, 'path': 'reviews/study/results.json', 'sha256': 'a' * 64}}))
    return manifest


def test_nested_manifest_and_relative_link(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'README.md').write_text('[study](docs/archive/study.md#details)\n')
    assert check(tmp_path) == (1, 2)


def test_nested_manifest_missing_summary_fails(tmp_path):
    manifest = fixture(tmp_path)
    manifest.with_suffix('.md').unlink()
    with pytest.raises(AssertionError):
        check(tmp_path)


def test_broken_local_link_fails(tmp_path):
    fixture(tmp_path)
    (tmp_path / 'README.md').write_text('[missing](docs/missing.md)\n')
    with pytest.raises(AssertionError, match='missing local link'):
        check(tmp_path)
