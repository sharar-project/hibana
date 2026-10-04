"""Check compact evidence manifests and local Markdown file links."""
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

def check(root: Path) -> tuple[int, int]:
    manifests = list((root / 'docs').rglob('*.json'))
    for path in manifests:
        assert path.stat().st_size <= 32 * 1024, f'{path}: raw results belong outside Git'
        data = json.loads(path.read_text())
        assert data['schema'] == 2, f'{path}: expected a compact evidence manifest'
        artifact = data['full_evidence']
        assert re.fullmatch('[0-9a-f]{64}', artifact['sha256']), path
        location = Path(artifact['path'])
        assert artifact['bytes'] > 0 and not location.is_absolute() and '..' not in location.parts, path
        assert (path.parent / data['summary_document']).is_file(), path

    documents = [*root.glob('*.md'), *(root / 'docs').rglob('*.md'), *(root / 'profiling').rglob('*.md')]
    for path in documents:
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for match in re.finditer(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)', text):
            link = match[1].strip('<>')
            target = urlsplit(link)
            if target.scheme or target.netloc or not target.path or target.path.startswith('/'):
                continue
            dest = path.parent / unquote(target.path)
            assert dest.exists(), f'{path}: missing local link {link}'
    return len(manifests), len(documents)


if __name__ == '__main__':
    counts = check(Path(__file__).resolve().parents[1])
    print(f'Checked {counts[0]} compact evidence manifests and {counts[1]} Markdown documents.')
