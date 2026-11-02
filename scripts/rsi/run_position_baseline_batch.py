"""Run fixed positional controls on an explicitly selected admitted project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.rsi import position_baselines
from agent.rsi.model_tasks import bound_file, read_task
from agent.rsi.research_project_catalog import build_catalog
from agent.rsi.storage import atomic_json, sha256_file


def run(manifest_path, project_id, output):
    manifest_path = Path(manifest_path).absolute()
    manifest = json.loads(manifest_path.read_text())
    catalog = build_catalog(manifest)
    if project_id not in {p["id"] for p in manifest["projects"]}:
        raise ValueError("Project is not in the admitted manifest")
    selected = sorted(
        [e for e in catalog["entries"] if e["project_id"] == project_id and e["status"] == "admitted"],
        key=lambda e: e["id"],
    )
    if not selected:
        raise ValueError("No admitted tasks in selected project")
    if any(e["protein_scope"]["kind"] != "protein" for e in selected):
        raise ValueError("Positional batch requires individual-protein scopes")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": position_baselines.VERSION,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "project_id": project_id,
        "selection": "All admitted individual-protein tasks of the selected project",
        "tasks": [{"id": e["id"], "task": e["task"]} for e in selected],
        "implementation": [
            {"path": str(p), "sha256": sha256_file(p)}
            for p in [Path(__file__).resolve(), Path(position_baselines.__file__).resolve()]
        ],
        "score_partition": "validation",
        "future_scored": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    atomic_json(output / "protocol.json", protocol)
    outcomes = []
    for entry in selected:
        try:
            task, _ = read_task(bound_file(entry["task"]))
            report = position_baselines.run_controls(task)
        except Exception as exc:
            report = {"task_id": entry["id"], "status": "failed", "error_type": type(exc).__name__, "detail": str(exc)}
        path = output / (entry["id"] + ".json")
        atomic_json(path, report)
        outcomes.append(
            {
                "task_id": entry["id"],
                "status": report["status"],
                "report": {"path": str(path), "sha256": sha256_file(path)},
            }
        )
        atomic_json(output / "progress.json", {"finished": len(outcomes), "allocated": len(selected)})
        print(json.dumps({"task": entry["id"], "status": report["status"]}), flush=True)
    result = {
        "allocated": len(selected),
        "completed": sum(x["status"] == "completed" for x in outcomes),
        "unsupported": sum(x["status"] == "unsupported" for x in outcomes),
        "failed": sum(x["status"] == "failed" for x in outcomes),
        "outcomes": outcomes,
        "future_scored": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    atomic_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.project_id, args.output)
    raise SystemExit(int(bool(result["failed"])))
