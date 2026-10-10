"""Moving or deleting documentation cannot leave dangling local links."""
import pytest

from tools.check_links import check


def test_relative_link_with_fragment(tmp_path):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'guide.md').write_text('# Guide\n')
    (tmp_path / 'README.md').write_text('[guide](docs/guide.md#details)\n')
    assert check(tmp_path) == 2


def test_broken_local_link_fails(tmp_path):
    (tmp_path / 'README.md').write_text('[missing](docs/missing.md)\n')
    with pytest.raises(AssertionError, match='missing local link'):
        check(tmp_path)
