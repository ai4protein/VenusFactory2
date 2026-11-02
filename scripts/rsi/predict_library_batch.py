#!/usr/bin/env python3
"""Invoke registered models without external language-model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
from agent.rsi.engine.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new result path")
    requests = json.loads(args.requests.read_text())
    result = ScientistModelLibrary(args.library).predict_many(requests)
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requests": result["batch_receipt"]["requests"],
                "rows": result["batch_receipt"]["rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
