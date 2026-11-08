#!/usr/bin/env python3
"""Run an installed structure runtime's native CLI from the repository root.

Administrator CLI only. Public API/MCP access still requires a published tool,
an enabled runtime and verified inference evidence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bootstrap.common import clean_env, read_json
from scripts.bootstrap.structure_runtimes import runtime_environment


def prepare(name, arguments, root=ROOT):
    profiles = read_json(root / "configs/model_runtimes.json", {})
    config = profiles.get(name, {})
    if not config.get("installation"):
        raise ValueError(f"{name} is not installed; run initialize.py install --tools {name}")
    receipt_path = root / config["installation"]
    receipt = read_json(receipt_path, {})
    if receipt.get("runtime") != name:
        raise ValueError("Runtime installation receipt is missing or does not match")
    target = receipt_path.parent
    python = root / config["python"]
    if not python.is_file():
        raise ValueError("Runtime interpreter is missing; run initialize.py doctor")
    env = clean_env()
    env.update(runtime_environment(name, target, python))
    if name == "alphafold2":
        executable = root / config["root"] / "run_alphafold.py"
        args = [str(python), str(executable), *arguments]
    else:
        executable = python.parent / ("mol" if name == "mol1" else "protenix")
        args = [str(executable), *arguments]
    return args, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("alphafold2", "protenix", "mol1"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command, env = prepare(args.runtime, arguments or ["--help"])
    os.execve(command[0], command, env)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
