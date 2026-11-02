"""Independent terminal audit of all REAP cross-protein allocations and controls."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import shutil
import sys
from pathlib import Path

from scripts.rsi.cases.audit_reap_generated import audit as audit_group
from scripts.rsi.cases.audit_reap_generated import compare_metrics, independent_metrics, read, require

from agent.harness.contracts import digest, file_hash
from agent.rsi import generated_features
from agent.rsi.cases.reap_experience import design, load_library
from agent.rsi.cases.reap_model_case import prepare_reap_model_case


def verify_barrier(root, groups):
    report = read(root / "report.json")
    barrier = read(root / "development_barrier.json")
    require(
        file_hash(root / "development_barrier.json") == report["development_barrier_sha256"], "Global barrier changed"
    )
    require(barrier["allocations"] == 36 and barrier["future_metrics_present"] is False, "Incomplete scoring barrier")
    require([x["id"] for x in barrier["groups"]] == [g["id"] for g in groups], "Barrier group order differs")
    for group, binding in zip(groups, barrier["groups"], strict=True):
        folder = root / f"group_{group['id']:02d}"
        path = folder / "development_terminal.json"
        require(file_hash(path) == binding["terminal_sha256"], "Development terminal snapshot changed")
        old, new = read(path), read(folder / "report.json")
        require(old["status"] == "development_completed" and old["source_intact"], "Development did not finish intact")
        require(
            new["status"] == "campaign_completed" and new["future_scoring_started_after_global_development"],
            "Future scoring incomplete",
        )
        require(len(old["arms"]) == len(new["arms"]) == len(binding["allocations"]) == 4, "An allocation was dropped")
        for earlier, current, identity in zip(old["arms"], new["arms"], binding["allocations"], strict=True):
            require(
                earlier["status"] not in {"running", "allocated"} and "future_metrics" not in earlier,
                "Premature scoring",
            )
            require(all(earlier[k] == identity[k] for k in identity), "Terminal allocation identity changed")
            require(
                earlier
                == {
                    k: v
                    for k, v in current.items()
                    if k not in {"future_metrics", "future_scoring", "future_metric_role"}
                },
                "Development outcome changed during scoring",
            )
    return barrier


def physical_calls(folder):
    counts = {"model_calls": 0, "tool_calls": 0}
    for phase in ["develop", "reuse"]:
        for filename, field, event, key in [
            ("model_requests.jsonl", "event", "request", "model_calls"),
            ("events.jsonl", "kind", "tool_call_start", "tool_calls"),
        ]:
            path = folder / phase / "execution" / filename
            if path.exists():
                counts[key] += sum(
                    json.loads(line).get(field) == event for line in path.read_text().splitlines() if line.strip()
                )
    return counts


def audit(root, implementation_root, output):
    root, implementation_root = root.resolve(), implementation_root.resolve()
    report, protocol = read(root / "report.json"), read(root / "protocol.json")
    require(
        report["status"] == "campaign_completed" and report["completed_allocations"] == 36,
        "Complete 36-allocation campaign required",
    )
    require(digest(protocol) == report["protocol_sha256"], "Top-level protocol changed")
    groups = protocol["groups"]
    require(
        len(groups) == 9 and groups == design(tuple(groups[i]["seed"] for i in [0, 3, 6])),
        "Declared balanced design changed",
    )
    for name, sha in protocol["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Experiment implementation changed",
        )
    manifest, donors = load_library(root / "donor_library")
    require(manifest == protocol["library"], "Frozen source library changed")
    barrier = verify_barrier(root, groups)
    output.mkdir(parents=True, exist_ok=False)
    results, rows, tasks = [], [], {}
    for group in groups:
        folder = root / f"group_{group['id']:02d}"
        require(file_hash(folder / "report.json") == report["group_reports"][str(group["id"])], "Group report changed")
        child, child_protocol = read(folder / "report.json"), read(folder / "protocol.json")
        require(
            child_protocol["defer_future_scoring"] and child_protocol["experience_library"] == manifest,
            "Group exposure protocol differs",
        )
        result = audit_group(folder, implementation_root)
        (output / f"group_{group['id']:02d}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(
            {
                "group": group["id"],
                "audit_sha256": file_hash(output / f"group_{group['id']:02d}.json"),
                "status": result["status"],
            }
        )
        print(json.dumps(results[-1]), flush=True)
        for arm in child["arms"]:
            case = child_protocol["cases"][arm["family"]]
            require(
                arm["condition"] == group["condition"] and case["split"]["seed"] == group["seed"],
                "Condition or partition changed",
            )
            require(
                child_protocol["profiles"][arm["mode"]] == {**protocol["profile"], "mode": arm["mode"]},
                "Common execution profile differs",
            )
            expected_ids = [e["experience_id"] for e in case["experience_projection"]["entries"]]
            matches = [
                e["experience_id"]
                for e in case["experience_projection"]["entries"]
                if e["source_sha256"] == arm.get("source_sha256")
            ]
            if "assessment" in arm:
                require(
                    arm["exposed_experience_ids"] == expected_ids and arm["matches_exposed_source"] == matches,
                    "Recorded source exposure or retention differs",
                )
            measured = arm.get("future_metrics")
            row = {
                "group": group["id"],
                "arm": arm["id"],
                "family": arm["family"],
                "seed": group["seed"],
                "condition": group["condition"],
                "mode": arm["mode"],
                "status": arm["status"],
                "model_id": arm.get("model_id"),
                "verified_candidate": "assessment" in arm,
                "accepted": arm.get("assessment", {}).get("passed", False),
                "fresh_reuse": arm["status"] == "completed",
                "spearman": None if measured is None else measured["spearman"],
                "mse_skill": None if measured is None else measured["mse_skill_vs_training_mean"],
                "ranking_utility": -1.0 if measured is None else measured["spearman"],
                "utility_is_failure_floor": measured is None,
                "matches_exposed_source": arm.get("matches_exposed_source", []),
                **physical_calls(folder / f"arm_{arm['id']:03d}"),
            }
            rows.append(row)
            tasks[(arm["family"], group["seed"])] = case
    observed = read(root / "analysis.json")
    require(rows == observed["rows"] and len(rows) == 36, "All-allocation analysis or physical costs differ")
    table = {(x["family"], x["seed"], x["mode"], x["condition"]): x for x in rows}
    expected = []
    for family, seed, mode in sorted({key[:3] for key in table}):
        for control in ["none", "summary"]:
            a, b = table[(family, seed, mode, "source")], table[(family, seed, mode, control)]
            expected.append(
                {
                    "family": family,
                    "seed": seed,
                    "mode": mode,
                    "contrast": "source_minus_" + control,
                    "ranking_utility_difference": a["ranking_utility"] - b["ranking_utility"],
                    "reuse_success_difference": int(a["fresh_reuse"]) - int(b["fresh_reuse"]),
                    "model_call_difference": a["model_calls"] - b["model_calls"],
                }
            )
    require(expected == observed["paired_contrasts"] and len(expected) == 24, "Paired contrast differs")
    controls = report["direct_source_controls"]
    require(
        len(controls) == 6 and {(x["family"], x["seed"]) for x in controls} == set(tasks),
        "Direct-source control missing or duplicated",
    )
    checked_controls = []
    for item in controls:
        key = item["family"], item["seed"]
        case = tasks[key]
        source = case["source"]
        rebuilt = prepare_reap_model_case(source["admission"], source["features"], source["baseline"], *key)
        partition = protocol["partitions"][key[0] + ":" + str(key[1])]
        require(
            partition
            == {
                "scope": rebuilt[0],
                "source": rebuilt[4],
                "split": rebuilt[5],
                "data_hashes": [digest(x) for x in rebuilt[1:4]],
            },
            "Top-level target partition changed",
        )
        donor = next(entry for family, (_, entry) in donors.items() if family != key[0])
        require(donor["source_sha256"] == item["donor_source_sha256"], "Direct source identity differs")
        folder = root / "direct_source" / (key[0] + ":" + str(key[1]))
        try:
            model = generated_features.fit_candidate(
                rebuilt[1], rebuilt[0], {"source_code": donor["source_code"], "seed": 314159}
            )
            _, validation = independent_metrics(model, rebuilt[2], rebuilt[0])
        except Exception as exc:
            require(
                item["status"] == "failed" and type(exc).__name__ == item["error_type"], "Direct refit failure differs"
            )
            checked_controls.append({"family": key[0], "seed": key[1], "status": "failure_replayed"})
            continue
        require(
            digest(model) == item["model_sha256"] == digest(read(folder / "model.json")), "Direct source refit differs"
        )
        compare_metrics(item["validation"], validation)
        try:
            _, future = independent_metrics(model, rebuilt[3], rebuilt[0])
        except Exception as exc:
            require(
                item["status"] == "future_scoring_failed" and type(exc).__name__ == item["error_type"],
                "Direct future failure differs",
            )
            checked_controls.append({"family": key[0], "seed": key[1], "status": "future_failure_replayed"})
            continue
        require(item["status"] == "completed", "Direct control terminal status differs")
        compare_metrics(item["future_metrics"], future)
        checked_controls.append({"family": key[0], "seed": key[1], "status": "refit_and_metrics_verified"})
    result = {
        "status": "passed",
        "report_sha256": file_hash(root / "report.json"),
        "protocol_sha256": file_hash(root / "protocol.json"),
        "analysis_sha256": file_hash(root / "analysis.json"),
        "auditor_sha256": file_hash(Path(__file__)),
        "allocations_checked": len(rows),
        "paired_contrasts_checked": len(expected),
        "groups": results,
        "direct_source_controls": checked_controls,
        "barrier_allocations": barrier["allocations"],
        "scope": "Independent source/partition/ancestry, numerical refit, metric, global scoring barrier and all-allocation cost audit. Not an independent efficacy replicate or evidence of improver evolution.",
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    shutil.copy2(Path(__file__), output / "auditor.py")
    helper = Path(sys.modules[audit_group.__module__].__file__)
    shutil.copy2(helper, output / helper.name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign, args.implementation_root, args.output)), flush=True)
