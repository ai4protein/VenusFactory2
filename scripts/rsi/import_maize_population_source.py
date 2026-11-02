"""Extract source-bound maize observations; this does not launch model tasks."""

import argparse
import json

from agent.rsi.maize_population_source import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = export_source(args.source, args.output)
    print(json.dumps({"summary": manifest["summary"], "policy": manifest["policy"]}, indent=2))


if __name__ == "__main__":
    main()
