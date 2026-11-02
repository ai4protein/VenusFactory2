"""Prepare the pinned POISK source and run offline feasibility controls."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agent.rsi import poisk_source, poisk_training  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["source", "tasks", "controls"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source":
        assert args.references is not None
        result = poisk_source.prepare(args.input, args.references, args.output)
    elif args.action == "tasks":
        result = poisk_training.prepare(args.input, args.output)
    else:
        result = poisk_training.train_controls(args.input, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"tasks", "results", "excluded_scopes", "reference_bindings"}},
            indent=2,
        )
    )
