#!/usr/bin/env python3
"""Freeze the yeast food-population source protocol without reading outcomes."""

import argparse
import json

from agent.rsi.yeast_population import freeze_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_source(args.raw_root, args.output), indent=2))


if __name__ == "__main__":
    main()
