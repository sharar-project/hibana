"""Check that local links in the repository's Markdown files point at files that exist."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def check(root: Path) -> int:
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
    return len(documents)


if __name__ == '__main__':
    print(f'Checked the local links of {check(Path(__file__).resolve().parents[1])} Markdown documents.')
