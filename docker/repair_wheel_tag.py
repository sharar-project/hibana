"""Repair NVIDIA 0.8.1's SBSA WHEEL tag after checking the actual ELF.

The published aarch64 wheel contains an unsupported manylinux2014_sbsa tag.
Only this metadata record changes; library bytes and dependency pins stay exact.
Fail on any other version/platform/tag instead of hiding pip check failures.
"""
import base64
import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path
import platform
import struct

d = importlib.metadata.distribution('nvidia-cusparselt-cu13')
assert d.version == '0.8.1' and platform.machine() == 'aarch64'
wheel_entry = next(f for f in d.files if str(f).endswith('.dist-info/WHEEL'))
wheel = Path(d.locate_file(wheel_entry))
old = wheel.read_text()
assert 'Tag: py3-none-manylinux2014_sbsa\n' in old
libraries = [Path(d.locate_file(f)) for f in d.files if str(f).endswith('.so.0')]
assert libraries
for library in libraries:
    with library.open('rb') as stream:
        header = stream.read(20)
    assert header[:6] == b'\x7fELF\x02\x01' and struct.unpack('<H', header[18:20])[0] == 183
    print(f'Validated AArch64 ELF: {library.name}')
new = old.replace('Tag: py3-none-manylinux2014_sbsa\n', 'Tag: py3-none-manylinux2014_aarch64\n').encode()
wheel.write_bytes(new)
record = wheel.with_name('RECORD')
rows = list(csv.reader(io.StringIO(record.read_text())))
for row in rows:
    if row[0] == str(wheel_entry):
        row[1:] = ['sha256=' + base64.urlsafe_b64encode(hashlib.sha256(new).digest()).decode().rstrip('='), str(len(new))]
with record.open('w') as stream:
    csv.writer(stream).writerows(rows)
print('Repaired nvidia-cusparselt-cu13==0.8.1 WHEEL platform tag: sbsa -> aarch64')
