"""Evaluate a terminal yeast pilot using its frozen campaign and numerical code."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The supplemental controller lives outside the already frozen experiment.
    # All campaign, publication and numerical imports resolve inside its snapshot.
    sys.path.insert(0, str(args.pilot.absolute() / "repository/src"))
    controller = Path(__file__).resolve().parents[2] / "src/agent/rsi/yeast_population_evaluation.py"
    spec = importlib.util.spec_from_file_location("agent.rsi.yeast_population_evaluation", controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate(args.pilot, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
