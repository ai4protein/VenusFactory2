"""Execute or independently verify the complete CLEAN split10 training scope."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import clean_training_plugin as plugin  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Host-only read-only replay of a completed plugin report")
    args = parser.parse_args()
    output = plugin.output_path(ROOT, args.output)
    if args.verify:
        result = plugin.verify_in_process(plugin.read(output / "report.json"), output, ROOT)
    else:
        report = plugin.run(ROOT, output)
        result = {"status": report["status"], "plugin": plugin.PLUGIN, "new_embedding_records": 22525}
    print(json.dumps(result, allow_nan=False), flush=True)
