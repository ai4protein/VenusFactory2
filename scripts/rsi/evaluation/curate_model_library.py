"""Plan, apply or reverse coverage-preserving research model curation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.rsi.engine import model_curation, model_retention
from agent.rsi.engine.scientist_model_library import ScientistModelLibrary
from agent.rsi.engine.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--policy", type=Path, default=ROOT / "configs/rsi/model_retention_v1.json")
    plan.add_argument("--output", type=Path, required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("--plan", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--entry-id", required=True)
    restore.add_argument("--reason", required=True)
    args = parser.parse_args()
    library = ScientistModelLibrary(args.library)
    if args.action == "plan":
        result = model_retention.plan_retention(library, json.loads(args.policy.read_text()))
        atomic_json(args.output, result, exclusive=True)
        print(
            json.dumps(
                {
                    "registered": result["registered_entries"],
                    "active": len(result["active_ids"]),
                    "archive": len(result["archive"]),
                    "coverage": result["coverage"],
                }
            )
        )
    elif args.action == "apply":
        state = model_curation.apply(library, json.loads(args.plan.read_text()))
        print(json.dumps({"revision": state["revision"], "archived": len(state["archived"])}))
    else:
        state = model_curation.restore(library, args.entry_id, reason=args.reason)
        print(json.dumps({"revision": state["revision"], "archived": len(state["archived"])}))


if __name__ == "__main__":
    main()
