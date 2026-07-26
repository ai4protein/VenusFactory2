#!/usr/bin/env python3
"""Dispatch to .install/install.py (venv) or .install/install_system.py."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", type=str, default="venv", help="venv or system")
    args, remaining = parser.parse_known_args(argv)

    script = _REPO_ROOT / (
        ".install/install_system.py" if args.env == "system" else ".install/install.py"
    )
    if not script.is_file():
        print(f"ERROR: missing installer script: {script}", file=sys.stderr)
        return 1

    proc = subprocess.run([sys.executable, str(script), *remaining], cwd=str(_REPO_ROOT))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
