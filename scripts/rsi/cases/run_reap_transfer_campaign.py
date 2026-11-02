"""Repeated, frozen cross-protein REAP experience ablation.

Nine condition/partition groups contain 36 allocations. All scientific future
scoring waits until every allocated development/reuse episode has ended.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import asyncio
import json
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from scripts.rsi.cases import run_reap_generated_matrix as component

from agent.harness import Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.reap_experience import design, load_library
from agent.rsi.cases.reap_model_case import prepare_reap_model_case

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def score_group(folder):
    report, protocol = read(folder / "report.json"), read(folder / "protocol.json")
    if report["status"] != "development_completed" or not protocol["defer_future_scoring"]:
        raise ValueError("All allocated development phases must finish before scoring")
    for arm in report["arms"]:
        if "assessment" not in arm:
            arm["future_scoring"] = {"status": "no_verified_candidate"}
        else:
            af = folder / f"arm_{arm['id']:03d}"
            try:
                model = read(af / "controller/artifacts/versions" / arm["model_id"] / "model.json")
                future = read(folder / "data" / arm["family"] / "future.json")
                arm["future_metrics"] = get_model_plugin(component.PLUGIN).evaluate(
                    model, future, protocol["cases"][arm["family"]]["scope"]
                )
                arm["future_scoring"] = {"status": "completed"}
                arm["future_metric_role"] = (
                    "Post-allocation scoring after all 36 construction/reuse allocations; includes rejected candidates"
                )
            except Exception as exc:
                arm["future_scoring"] = {"status": "failed", "error_type": type(exc).__name__}
                (af / "future_scoring_error.log").write_text(traceback.format_exc())
        write_json(folder / f"arm_{arm['id']:03d}" / "report.json", arm)
    report.update(status="campaign_completed", future_scoring_started_after_global_development=True)
    write_json(folder / "report.json", report)
    return report


def freeze_development_barrier(root, groups):
    """Bind every terminal allocation before any global future scoring."""
    children = []
    for group in groups:
        folder = root / f"group_{group['id']:02d}"
        report = read(folder / "report.json")
        if report["status"] != "development_completed" or len(report["arms"]) != 4:
            raise ValueError("Global scoring requires all four terminal allocations in every group")
        if any(a["status"] in {"allocated", "running"} or "future_metrics" in a for a in report["arms"]):
            raise ValueError("Nonterminal or prematurely scored allocation")
        target = folder / "development_terminal.json"
        with target.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        children.append(
            {
                "id": group["id"],
                "terminal_sha256": file_hash(target),
                "allocations": [
                    {k: arm[k] for k in ["id", "family", "mode", "condition", "status"]} for arm in report["arms"]
                ],
            }
        )
    barrier = {
        "groups": children,
        "allocations": sum(len(g["allocations"]) for g in children),
        "future_metrics_present": False,
        "created_at_unix": time.time(),
    }
    if barrier["allocations"] != 36:
        raise ValueError("Exactly 36 allocations required before future scoring")
    write_json(root / "development_barrier.json", barrier)
    return file_hash(root / "development_barrier.json")


def analysis_rows(reports, root=None):
    rows = []
    for group, report in reports:
        for arm in report["arms"]:
            measured = arm.get("future_metrics")
            phases = [arm[n]["run"] for n in ["development", "reuse"] if n in arm]
            model_calls = sum(x["audit"]["requests"]["model_calls"] for x in phases)
            tool_calls = sum(x["audit"]["events"]["tool_calls"] for x in phases)
            if root is not None:
                folder = Path(root) / f"group_{group['id']:02d}" / f"arm_{arm['id']:03d}"
                model_calls = tool_calls = 0
                for phase in ["develop", "reuse"]:
                    for filename, field, value, counter in [
                        ("model_requests.jsonl", "event", "request", "model"),
                        ("events.jsonl", "kind", "tool_call_start", "tool"),
                    ]:
                        path = folder / phase / "execution" / filename
                        if path.exists():
                            n = sum(json.loads(line).get(field) == value for line in path.read_text().splitlines())
                            if counter == "model":
                                model_calls += n
                            else:
                                tool_calls += n

            rows.append(
                {
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
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "matches_exposed_source": arm.get("matches_exposed_source", []),
                }
            )
    return rows


def paired_contrasts(rows):
    table = {(r["family"], r["seed"], r["mode"], r["condition"]): r for r in rows}
    results = []
    for family, seed, mode in sorted({(r["family"], r["seed"], r["mode"]) for r in rows}):
        for control in ["none", "summary"]:
            a, b = table[(family, seed, mode, "source")], table[(family, seed, mode, control)]
            results.append(
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
    return results


async def main(args):
    profile = Profile.from_dict(read(args.profile))
    if profile.model.model != "glm-5.2" or profile.harness != "langgraph":
        raise ValueError("Declared common GLM-5.2/LangGraph experiment required")
    library_manifest, donors = load_library(args.experience_library)
    groups = design(tuple(args.seeds))
    tasks = {
        (family, seed): prepare_reap_model_case(args.admission, args.features, args.baseline, family, seed)
        for seed in args.seeds
        for family in component.FAMILIES
    }
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.experience_library, args.output / "donor_library")
    protocol = {
        "schema": 1,
        "profile": profile.public(),
        "groups": groups,
        "allocations": 4 * len(groups),
        "library": library_manifest,
        "partitions": {
            family + ":" + str(seed): {
                "scope": case[0],
                "source": case[4],
                "split": case[5],
                "data_hashes": [digest(v) for v in case[1:4]],
            }
            for (family, seed), case in tasks.items()
        },
        "instruction_control": "Same experience-aware task prompt for none, summary and source; only declared projection changes",
        "order": "Cyclic condition order across seeds; Agent/Scientist order reversed between families within every group",
        "donor_selection": "First chronologically accepted and reused implementation for each family from the declared audited cohorts. Keep adverse known donor generalization; transfer only from the other family.",
        "memory_update": "Target-arm publications are archived with exposed-donor ancestry but never become input to another allocation. This isolates fixed cross-family experience availability.",
        "primary_operational_outcome": "Independent acceptance and fresh-session reuse per all allocated arms",
        "primary_scientific_utility": "Future Spearman for every independently refitted submitted model, including rejected candidates; assign predefined utility -1 when no valid score exists. The floor is a pipeline utility, not an observed correlation.",
        "secondary": [
            "conditional Spearman and MSE skill",
            "execution/candidate/acceptance/reuse separately",
            "physical requests and tool calls",
            "exact source retention",
            "source versus summary and none paired differences",
        ],
        "direct_source_control": "Refit the exact other-family donor source on each target partition with its original recipe seed; no LLM. Score only after all agent allocations, regardless of acceptance.",
        "future_scoring": "Only after all 36 agent development/reuse allocations end; no future metrics or labels supplied to agents",
        "analysis_scope": "Three overlapping partition repetitions per protein/mode/condition, not independent biological replicates. Two selected proteins and fixed donors support descriptive within-case comparisons, not broad mode or RSI superiority.",
        "exposure": "Public retrospective data already inspected by investigators; not blind or prospective. Provider alias is not an immutable model snapshot.",
        "implementation": {
            name: file_hash(ROOT / name)
            for name in [
                "scripts/rsi/cases/run_reap_transfer_campaign.py",
                "scripts/rsi/cases/run_reap_generated_matrix.py",
                "src/agent/rsi/cases/reap_experience.py",
            ]
        },
    }
    for name in protocol["implementation"]:
        destination = args.output / "implementation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(args.output / "protocol.json", protocol)
    report = {
        "status": "preflight",
        "protocol_sha256": digest(protocol),
        "groups": [],
        "direct_source_controls": [],
        "future_scoring_started": False,
    }
    write_json(args.output / "report.json", report)
    started = time.monotonic()
    if args.execute:
        report["status"] = "running"
        write_json(args.output / "report.json", report)
        for (family, seed), case in tasks.items():
            other = [entry for name, (_, entry) in donors.items() if name != family]
            if len(other) != 1:
                raise ValueError("One cross-family source required")
            donor = other[0]
            folder = args.output / "direct_source" / (family + ":" + str(seed))
            folder.mkdir(parents=True)
            item = {"family": family, "seed": seed, "donor_source_sha256": donor["source_sha256"], "status": "failed"}
            then = time.monotonic()
            try:
                plugin = get_model_plugin(component.PLUGIN)
                model = plugin.fit(case[1], case[0], {"source_code": donor["source_code"], "seed": 314159})
                write_json(folder / "model.json", model)
                item.update(
                    status="fitted", validation=plugin.evaluate(model, case[2], case[0]), model_sha256=digest(model)
                )
            except Exception as exc:
                item["error_type"] = type(exc).__name__
                (folder / "error.log").write_text(traceback.format_exc())
            item["wall_seconds"] = time.monotonic() - then
            report["direct_source_controls"].append(item)
            write_json(args.output / "report.json", report)
    for group in groups:
        folder = args.output / f"group_{group['id']:02d}"
        child = SimpleNamespace(
            profile=args.profile,
            admission=args.admission,
            features=args.features,
            baseline=args.baseline,
            output=folder,
            seed=group["seed"],
            execute=args.execute,
            credential_toml=args.credential_toml,
            condition=group["condition"],
            experience_library=args.output / "donor_library",
            defer_future_scoring=True,
        )
        item = {**group, "status": "running"}
        report["groups"].append(item)
        write_json(args.output / "report.json", report)
        print(json.dumps(item), flush=True)
        try:
            await component.main(child)
            child_report = read(folder / "report.json")
            item["status"] = child_report["status"]
            item["protocol_sha256"] = child_report["protocol_sha256"]
        except Exception as exc:
            item.update(status="controller_error", error_type=type(exc).__name__)
            (args.output / f"group_{group['id']:02d}_error.log").write_text(traceback.format_exc())
        write_json(args.output / "report.json", report)
    if not args.execute:
        report["status"] = (
            "preflight" if all(g["status"] == "preflight" for g in report["groups"]) else "preflight_failed"
        )
        write_json(args.output / "report.json", report)
        return
    if any(g["status"] != "development_completed" for g in report["groups"]):
        report.update(status="campaign_incomplete", wall_seconds=time.monotonic() - started)
        write_json(args.output / "report.json", report)
        return
    report["development_barrier_sha256"] = freeze_development_barrier(args.output, groups)
    report["future_scoring_started"] = True
    write_json(args.output / "report.json", report)
    completed = []
    for group in groups:
        completed.append((group, score_group(args.output / f"group_{group['id']:02d}")))
    for item in report["direct_source_controls"]:
        if item["status"] != "fitted":
            continue
        case = tasks[(item["family"], item["seed"])]
        folder = args.output / "direct_source" / (item["family"] + ":" + str(item["seed"]))
        try:
            model = read(folder / "model.json")
            item["future_metrics"] = get_model_plugin(component.PLUGIN).evaluate(model, case[3], case[0])
            item["status"] = "completed"
        except Exception as exc:
            item.update(status="future_scoring_failed", error_type=type(exc).__name__)
            (folder / "future_scoring_error.log").write_text(traceback.format_exc())
    rows = analysis_rows(completed, args.output)
    write_json(
        args.output / "analysis.json",
        {"rows": rows, "paired_contrasts": paired_contrasts(rows), "scope": protocol["analysis_scope"]},
    )
    report.update(
        status="campaign_completed",
        wall_seconds=time.monotonic() - started,
        completed_allocations=len(rows),
        group_reports={str(g["id"]): file_hash(args.output / f"group_{g['id']:02d}/report.json") for g in groups},
    )
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "allocations": len(rows)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["profile", "admission", "features", "baseline", "experience-library", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--execute", action="store_true")
    asyncio.run(main(p.parse_args()))
