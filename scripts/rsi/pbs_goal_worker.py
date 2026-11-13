#!/usr/bin/env python3
"""Start only the fixed worker from this frozen source tree."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from backend.compute.access import require_local_execution
    require_local_execution()
    if args.record.is_symlink():
        raise ValueError("PBS worker record must not be a symlink")
    path = args.record.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Invalid PBS worker record")
    record = json.loads(path.read_text())
    if Path(record["source"]["root"]).resolve() != root:
        raise ValueError("PBS worker entry point differs from the frozen source")
    extras = record["execution"].get("extra_python_paths", [])
    if not isinstance(extras, list) or any(not isinstance(p, str) or not Path(p).is_absolute() or not Path(p).is_dir() for p in extras):
        raise ValueError("Invalid worker dependency paths")
    prefixes = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    interpreter_paths = [p for p in sys.path if p and any(Path(p).resolve().is_relative_to(base) for base in prefixes)]
    sys.path[:] = list(dict.fromkeys([str(root / "src"), str(root), *extras, *interpreter_paths]))
    from agent.harness.pbs_goal_worker import run_record
    result = run_record(path)
    print(json.dumps({"status": result["status"], "cycles": len(result["cycles"])}))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        # Provider/runtime failures may contain request details. The ledger keeps
        # their type; PBS logs get no exception text or credential-bearing trace.
        print(json.dumps({"status": "interrupted", "error_type": type(error).__name__}))
        code = 1
    raise SystemExit(code)
