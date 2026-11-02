"""Prepare training-screened tasks for the existing native Scientist campaign CLI."""

import argparse
import json
from pathlib import Path

from agent.rsi.screened_research_batch import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(json.loads(args.request.read_text()), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
