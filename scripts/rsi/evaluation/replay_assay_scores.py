"""Independently replay saved assay predictions and metrics, without model refitting."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
from pathlib import Path

from agent.rsi import assay_score_replay as replay


def read_bound(path, expected):
    payload = path.read_bytes()
    replay.require(hashlib.sha256(payload).hexdigest() == expected, "Bound evidence changed: " + str(path))
    return json.loads(payload)


def main(args):
    root = args.evaluation.absolute()
    report = read_bound(root / "report.json", args.report_sha256)
    protocol = read_bound(root / "protocol.json", report["protocol_sha256"])
    frozen = read_bound(Path(protocol["frozen_protocol"]), protocol["frozen_protocol_sha256"])
    for path, expected in {
        **frozen["source_bindings"],
        **frozen["implementation"],
        **protocol["terminal_and_reuse_bindings"],
    }.items():
        replay.require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Frozen source changed")
    tasks = {t["id"]: t for t in frozen["tasks"]}
    replay.require(set(tasks) == set(report["task_reports"]), "Final allocation differs")
    results, rows = [], []
    for task_id, expected in report["task_reports"].items():
        row = read_bound(root / (task_id + ".json"), expected)
        results.append(replay.replay_task(tasks[task_id], row))
        rows.append(row)
    acquired = [r for r in rows if r["scientist"]["status"] == "scored"]
    totals = {
        "allocated": len(rows),
        "scored_scientist_models": len(acquired),
        "not_registered": len(rows) - len(acquired),
        "acquisition_rate": len(acquired) / len(rows) if rows else None,
        "positive_group_balanced_final_skill": sum(
            (r["scientist"]["metrics"]["group_balanced_mse_skill_vs_training_mean"] or 0) > 0 for r in acquired
        ),
        "beats_best_control": sum(r["comparison"]["scientist_beats_best_control"] for r in acquired),
        "ties_best_control": sum(r["comparison"]["scientist_ties_best_control"] for r in acquired),
        "completed_control_fits": sum(c["status"] == "completed" for r in rows for c in r["controls"]),
        "failed_control_fits": sum(c["status"] != "completed" for r in rows for c in r["controls"]),
    }
    replay.compare(report["totals"], totals)
    result = {
        "status": "independent_prediction_score_replay_passed",
        "report_sha256": args.report_sha256,
        "tasks": results,
        "totals": totals,
        "baseline_refit_performed": False,
        "maximum_absolute_error": max(r["maximum_absolute_error"] for r in results),
        "new_autonomous_models": 0,
        "external_model_calls": 0,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
