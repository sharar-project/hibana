"""Isolated loader peak measurement; no serving pools or graph allocation."""
import argparse
import json
from pathlib import Path

from sharar import config
from sharar.model.qwen38 import Qwen38
from memory_trace import measure_stage

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = {'model': config.MODEL_ID, 'revision': config.MODEL_REVISION}
try:
    with measure_stage(report, 'load'):
        model = Qwen38(config.snapshot_dir())
finally:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
