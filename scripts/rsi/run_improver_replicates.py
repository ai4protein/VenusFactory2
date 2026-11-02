"""Three independent GLM improver revisions and 54 matched offspring goals.

All revisions/plans are frozen before any model request. All offspring complete
and independently replay before any future scoring. Failures are never replaced.
"""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path.insert(0, str(_entry_root))


import argparse
import asyncio
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.rsi.cases import audit_reap_improver_comparison as independent
import run_improver_revision as revision
from scripts.rsi.cases import run_reap_goal_comparison as goals
from scripts.rsi.cases import run_reap_improver_comparison as comparison
from scripts.rsi.cases.audit_reap_science_goal import audit as audit_goal
from run_harness_probe import ROOT

from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json as _write_json
from agent.rsi.engine.improver_replicates import freeze_batch_barrier, interleaved_schedule
from agent.rsi.engine.source_snapshot import research_implementation_paths

read = revision.read


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def prepare(args):
    root, audits = args.output.resolve(), args.audit_root.resolve()
    if audits.is_relative_to(ROOT) or audits.is_relative_to(root):
        raise ValueError("Independent audits must be outside the implementation and experiment roots")
    root.mkdir(parents=True, exist_ok=False)
    audits.mkdir(parents=True, exist_ok=False)
    bindings, schedules = {}, {}
    for group in ("r00", "r01", "r02"):
        rev = root / "revisions" / group
        offspring = root / "offspring" / group
        revision.prepare(
            SimpleNamespace(
                output=rev, profile=args.revision_profile, parent_policy=None, history_goal=args.history_goal
            )
        )
        comparison.plan(
            SimpleNamespace(
                output=offspring,
                revision=rev,
                profile=args.profile,
                admission=args.admission,
                features=args.features,
                baseline=args.baseline,
                seeds=[7, 8, 9],
            )
        )
        preflight = independent.preflight(offspring, ROOT)
        pre = audits / "preflight" / (group + ".json")
        write_json(pre, preflight)
        schedules[group] = read(offspring / "plan.json")["design"]["schedule"]
        bindings[group] = {
            "revision_protocol": file_hash(rev / "protocol.json"),
            "comparison_plan": file_hash(offspring / "plan.json"),
            "preflight": file_hash(pre),
        }
    histories = [read(root / "revisions" / group / "protocol.json")["history_sha256"] for group in bindings]
    if len(set(histories)) != 1:
        raise ValueError("Independent revisions must receive the identical development history")
    source = {str(p.relative_to(ROOT)): file_hash(p) for p in research_implementation_paths(ROOT)}
    plan = {
        "schema": 1,
        "bindings": bindings,
        "history_sha256": histories[0],
        "schedule": interleaved_schedule(schedules),
        "audit_root": str(audits),
        "implementation": source,
        "revision_attempts": 3,
        "offspring_allocations": 54,
        "runtime_concurrency": 1,
        "revision_failure_policy": "Stop before all offspring if any of the three predeclared revisions fails; no replacement",
        "future_policy": "All 54 offspring terminal and independently replayed before any future metric",
        "scope": "Three stochastic API revision attempts on shared retrospective proteins/partitions; no biological replication or policy promotion",
    }
    write_json(root / "plan.json", plan)
    write_json(
        root / "report.json", {"status": "prepared", "plan_sha256": digest(plan), "revisions": [], "allocations": []}
    )
    return plan


def verify(root):
    plan = read(root / "plan.json")
    if read(root / "report.json")["plan_sha256"] != digest(plan):
        raise ValueError("Batch plan changed")
    for name, sha in plan["implementation"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError("Frozen batch source changed")
    for group, hashes in plan["bindings"].items():
        if (
            file_hash(root / "revisions" / group / "protocol.json") != hashes["revision_protocol"]
            or file_hash(root / "offspring" / group / "plan.json") != hashes["comparison_plan"]
            or file_hash(Path(plan["audit_root"]) / "preflight" / (group + ".json")) != hashes["preflight"]
        ):
            raise ValueError("Predeclared revision or comparison changed")
    return plan


async def run(root, credential_toml):
    root = root.resolve()
    with (root / ".batch.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify(root)
        report = read(root / "report.json")
        if report["status"] != "prepared":
            raise ValueError("Batch already attempted; inspect its live handles or retained failure")
        audits = Path(plan["audit_root"])
        stage = "revision"
        try:
            report.update(status="revising", started_unix=time.time())
            write_json(root / "report.json", report)
            for group in plan["bindings"]:
                result = await revision.run(root / "revisions" / group, credential_toml)
                report["revisions"].append(
                    {"replicate": group, "status": result["status"], "policy_id": result.get("policy_id")}
                )
                write_json(root / "report.json", report)
            if any(row["status"] != "candidate_submitted" for row in report["revisions"]):
                report["status"] = "revision_failed"
                write_json(root / "report.json", report)
                return report
            stage = "materialization"
            group_reports = {}
            for group in plan["bindings"]:
                folder = root / "offspring" / group
                comparison.materialize(folder)
                protocol = comparison.verify_campaign(folder)
                group_reports[group] = {
                    "status": "running",
                    "allocations": [],
                    "planned": 18,
                    "design_sha256": protocol["design_sha256"],
                }
                write_json(folder / "report.json", group_reports[group])
            stage = "development"
            report["status"] = "developing"
            write_json(root / "report.json", report)
            env = {
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "PYTHONPATH": str(ROOT / "src"),
                "HOME": str(Path.home()),
                "LANG": "C.UTF-8",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            for allocation in plan["schedule"]:
                verify(root)
                group, aid = allocation["replicate"], allocation["allocation"]
                folder = root / "offspring" / group
                arm = folder / "arms" / aid
                row = next(r for r in read(folder / "protocol.json")["design"]["schedule"] if r["id"] == aid)
                item = {**row, "status": "running"}
                group_reports[group]["allocations"].append(item)
                batch_item = {**allocation, "status": "running"}
                report["allocations"].append(batch_item)
                started = time.monotonic()
                with (arm / "worker.log").open("wb") as log:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/rsi/cases/run_reap_improver_comparison.py"),
                        "worker",
                        "--output",
                        str(folder),
                        "--allocation",
                        aid,
                        "--credential-toml",
                        str(credential_toml),
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    identity = {
                        "pid": proc.pid,
                        "start_ticks": goals.process_start(proc.pid),
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    }
                    item.update(pid=proc.pid, process=identity)
                    batch_item["process"] = identity
                    write_json(folder / "report.json", group_reports[group])
                    write_json(root / "report.json", report)
                    code = await proc.wait()
                result_path = arm / "comparison_result.json"
                status = read(result_path)["status"] if result_path.exists() else "worker_missing_result"
                item.update(status=status, exit_code=code, worker_wall_seconds=time.monotonic() - started)
                batch_item.update(status=status, exit_code=code)
                write_json(folder / "report.json", group_reports[group])
                write_json(root / "report.json", report)
                print(json.dumps(batch_item), flush=True)
            if any(
                a["status"] not in {"completed", "budget_exhausted"} or a["exit_code"] != 0
                for a in report["allocations"]
            ):
                report["status"] = "development_incomplete"
                write_json(root / "report.json", report)
                return report
            stage = "independent_replay"
            report["status"] = "auditing"
            write_json(root / "report.json", report)
            for row in plan["schedule"]:
                arm = root / "offspring" / row["replicate"] / "arms" / row["allocation"]
                bindings = {
                    name: file_hash(arm / name) for name in ("goal.json", "protocol.json", "comparison_result.json")
                }
                report["audit_current"] = row
                write_json(root / "report.json", report)
                result = audit_goal(arm, ROOT, defer_future_scoring=True)
                if result.get("status") != "passed" or any(
                    file_hash(arm / name) != sha for name, sha in bindings.items()
                ):
                    raise ValueError("Independent replay failed or development evidence changed during audit")
                evidence = {
                    "status": "passed",
                    "future_scoring_deferred": True,
                    "source_bindings": bindings,
                    "audit": result,
                }
                write_json(audits / "development" / row["replicate"] / (row["allocation"] + ".json"), evidence)
            freeze_batch_barrier(root, plan["schedule"], audits, read)
            report["global_barrier_sha256"] = file_hash(root / "development_barrier.json")
            stage = "scoring"
            report["status"] = "scoring"
            write_json(root / "report.json", report)
            for group in plan["bindings"]:
                folder = root / "offspring" / group
                rows = read(folder / "protocol.json")["design"]["schedule"]
                current = group_reports[group]
                current.update(
                    all_allocations_attempted=True,
                    reported_complete_objectives=sum(r["status"] == "completed" for r in current["allocations"]),
                    independent_audit_required=True,
                )
                goals.finish_campaign(
                    folder, rows, current, analyze=comparison.score_offspring, barrier=goals.freeze_goal_barrier
                )
                result = independent.audit(
                    folder, ROOT, audits / "terminal" / group, audits / "preflight" / (group + ".json")
                )
                write_json(audits / "terminal" / group / "audit.json", result)
            verify(root)
            report.update(status="completed", independent_audits_passed=True, promoted=False)
        except BaseException as exc:
            report.update(status="failed", failed_stage=stage, error_type=type(exc).__name__)
            raise
        finally:
            report["finished_unix"] = time.time()
            write_json(root / "report.json", report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--revision-profile", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--history-goal", action="append", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--credential-toml", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        print(json.dumps({"status": "prepared", "revisions": 3, "offspring_allocations": 54}))
    else:
        result = asyncio.run(run(args.output, args.credential_toml))
        print(json.dumps({"status": result["status"]}))
