"""Build a project/task catalog, export a Scientist plan or run offline controls."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.datasets.model_tasks import bound_file, read_task
from agent.rsi.datasets.research_project_catalog import build_catalog, campaign_plan
from agent.rsi.engine.storage import sha256_file


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main(args):
    manifest = json.loads(args.manifest.read_text())
    catalog = build_catalog(manifest)
    if args.profile and not args.goal_budget:
        raise ValueError("Campaign export requires an explicit per-goal budget")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "manifest.json", manifest)
    write_new(args.output / "catalog.json", catalog)
    if args.profile:
        path = args.output / "manifest.json"
        reference = {"path": str(path.absolute()), "sha256": sha256_file(path)}
        plan = campaign_plan(
            manifest,
            catalog,
            json.loads(args.profile.read_text()),
            json.loads(args.goal_budget.read_text()),
            reference,
            args.generations,
        )
        write_new(args.output / "campaign_plan.json", plan)
    if args.controls:
        from agent.rsi.engine.traditional_baselines import run_controls

        outcomes = []
        for entry in catalog["entries"]:
            if entry["status"] == "admitted":
                task, _ = read_task(bound_file(entry["task"]))
                report = run_controls(task)
                import agent.rsi.engine.traditional_baselines as controls

                report["implementation"] = {"path": controls.__file__, "sha256": sha256_file(Path(controls.__file__))}
                path = args.output / (entry["id"] + "_controls.json")
                write_new(path, report)
                outcomes.append(
                    {"id": entry["id"], "report": {"path": str(path.absolute()), "sha256": sha256_file(path)}}
                )
        write_new(
            args.output / "controls.json", {"outcomes": outcomes, "external_model_calls": 0, "new_autonomous_models": 0}
        )
    print(json.dumps(catalog["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--goal-budget", type=Path)
    parser.add_argument("--generations", type=int, default=1)
    main(parser.parse_args())
