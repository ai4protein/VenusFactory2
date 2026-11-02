"""Prepare individual designed-protein stability tasks from bound curated records."""

import argparse
import json
from pathlib import Path

from agent.rsi.designed_protein_tasks import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.source_manifest.read_text()), args.output), indent=2))
