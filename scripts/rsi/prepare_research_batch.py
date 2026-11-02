"""Project an explicitly selected execution batch from an audited source catalog."""

import argparse
import json
from pathlib import Path

from agent.rsi.research_batches import export_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if set(selection) != {"parent_manifest", "parent_catalog", "candidate_ids"}:
        raise ValueError("Explicit bound parent manifest, catalog and candidate IDs required")
    result = export_batch(
        selection["parent_manifest"], selection["parent_catalog"], selection["candidate_ids"], args.output
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
