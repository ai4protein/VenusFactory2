"""Import reviewed plant kinetics into observations, without launching models."""

import argparse
import json

from agent.rsi.sabio_kinetics_source import export_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = export_source(args.manifest, args.sha256, args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
