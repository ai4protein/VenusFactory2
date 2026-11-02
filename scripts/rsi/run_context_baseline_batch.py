"""Run all fixed controls for one admitted protein-condition project."""

from __future__ import annotations

import argparse
import json
import signal
from collections import Counter
from pathlib import Path

from agent.rsi import context_baselines, reference_thermal_baselines
from agent.rsi.datasets.model_tasks import bound_file, read_task
from agent.rsi.datasets.research_project_catalog import build_catalog
from agent.rsi.engine.storage import atomic_json, sha256_file


def run(manifest_path, project_id, output):
    manifest_path = Path(manifest_path).absolute()
    manifest = json.loads(manifest_path.read_text())
    catalog = build_catalog(manifest)
    selected = sorted(
        [e for e in catalog["entries"] if e.get("project_id") == project_id and e["status"] == "admitted"],
        key=lambda e: e["id"],
    )
    if not selected:
        raise ValueError("No admitted tasks in the selected project")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "protocol.json",
        {
            "schema": context_baselines.VERSION,
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "project_id": project_id,
            "selection": "all admitted contextual tasks in the selected project",
            "tasks": [{"id": e["id"], "task": e["task"]} for e in selected],
            "implementation": [
                {"path": str(p), "sha256": sha256_file(p)}
                for p in [
                    Path(__file__).resolve(),
                    Path(context_baselines.__file__).resolve(),
                    Path(reference_thermal_baselines.__file__).resolve(),
                ]
            ],
            "score_partition": "validation",
            "future_scored": False,
            "maximum_seconds_per_task": 120,
            "external_model_calls": 0,
            "new_autonomous_models": 0,
        },
    )

    def timeout(signum, frame):
        raise TimeoutError("Fixed-control task exceeded the predeclared 120-second limit")

    previous = signal.signal(signal.SIGALRM, timeout)
    outcomes, method_statuses = [], Counter()
    try:
        for entry in selected:
            signal.setitimer(signal.ITIMER_REAL, 120)
            try:
                task, _ = read_task(bound_file(entry["task"]))
                if task["plugin_id"] not in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}:
                    raise ValueError("Context interface required for every selected task")
                report = context_baselines.run_controls(task)
                status = "completed" if all(r["status"] == "completed" for r in report["outcomes"]) else "partial"
                for row in report["outcomes"]:
                    method_statuses[(row["method"], row["status"])] += 1
            except Exception as exc:
                status = "failed"
                report = {
                    "task_id": entry["id"],
                    "status": status,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            path = output / (entry["id"] + ".json")
            atomic_json(path, report)
            outcomes.append(
                {"task_id": entry["id"], "status": status, "report": {"path": str(path), "sha256": sha256_file(path)}}
            )
            atomic_json(
                output / "progress.json",
                {
                    "finished": len(outcomes),
                    "allocated": len(selected),
                    "statuses": dict(Counter(r["status"] for r in outcomes)),
                },
            )
            print(json.dumps({"task": entry["id"], "status": status}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result = {
        "allocated": len(selected),
        "task_statuses": dict(Counter(r["status"] for r in outcomes)),
        "method_statuses": [
            {"method": method, "status": status, "count": count}
            for (method, status), count in sorted(method_statuses.items())
        ],
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
    raise SystemExit(int(bool(result["task_statuses"].get("failed"))))
