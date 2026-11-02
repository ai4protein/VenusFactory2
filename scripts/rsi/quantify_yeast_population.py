#!/usr/bin/env python3
"""Reconstruct the frozen food-population protein-response dataset locally."""

import argparse
import json

from agent.rsi.yeast_population_quant import quantify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantify_source(
        args.raw_root,
        args.frozen_root,
        args.output,
        progress=lambda status: print(json.dumps(status, sort_keys=True), flush=True),
    )


if __name__ == "__main__":
    main()
