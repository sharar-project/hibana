"""Keep raw experiment dumps out of committed documentation."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for path in (root / 'docs').glob('*results.json'):
    assert path.stat().st_size <= 32 * 1024, f'{path}: raw results belong outside Git'
    data = json.loads(path.read_text())
    assert data['schema'] == 2, f'{path}: expected a compact evidence manifest'
    artifact = data['full_evidence']
    assert re.fullmatch('[0-9a-f]{64}', artifact['sha256']), path
    assert artifact['bytes'] > 0 and not Path(artifact['path']).is_absolute(), path
    assert (path.parent / data['summary_document']).is_file(), path
print('Committed evidence manifests are compact and valid.')
