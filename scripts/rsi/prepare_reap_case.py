"""Admit public REAP data without executing author code or calling a model."""
import argparse
import json
from pathlib import Path

from agent.harness.contracts import file_hash
from agent.rsi import reap_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = reap_case.ingest(args.assets)
    result["summary"]["implementation_sha256"] = file_hash(Path(reap_case.__file__))
    args.output.mkdir(parents=True)
    (args.output / "reap_case.py").write_bytes(Path(reap_case.__file__).read_bytes())
    for name, value in result.items():
        (args.output / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
