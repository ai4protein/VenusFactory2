#!/usr/bin/env python3
"""Inventory explicitly listed model libraries without external APIs."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.engine.model_scope_inventory import inventory_libraries
from agent.rsi.engine.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--target-scopes", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new snapshot path")
    report = inventory_libraries(json.loads(args.libraries.read_text()), target_scopes=args.target_scopes)
    atomic_json(args.output, report)
    keys = [
        "status",
        "registered_entries",
        "registered_scopes",
        "repeated_scope_registrations",
        "currently_reverified_scopes",
        "completion_claim",
    ]
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
