"""Compatibility wrapper for the shared platform dependency installer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bootstrap.legacy_install import main

if __name__ == "__main__":
    raise SystemExit(main(environment="venv"))
