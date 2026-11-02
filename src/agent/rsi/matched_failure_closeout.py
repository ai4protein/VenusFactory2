"""Close matched studies without converting unpublished failures into successes."""

import datetime
import hashlib
import math
from pathlib import Path

from .matched_history_analysis import require, validate_design
from .matched_history_evidence import Evidence
from .protein_model import digest
from .scientist_model_campaign import process_live as _process_live
from .shared_sequence_evaluation import require_terminal

SCHEMA = "matched_study_terminal_closeout.v1"
RESOURCES = ("model_calls", "tool_calls", "wall_seconds")


def process_live(identity):
    """External runner manifests may encode Linux start ticks as JSON numbers."""
    require(
        isinstance(identity, dict) and type(identity.get("pid")) is int and identity["pid"] > 0, "Malformed runner PID"
    )
    ticks = identity.get("start_ticks")
    require(type(ticks) in (int, str) and str(ticks).isascii() and str(ticks).isdigit(), "Malformed runner start ticks")
    require(isinstance(identity.get("boot_id"), str) and bool(identity["boot_id"]), "Missing runner boot identity")
    return _process_live({**identity, "start_ticks": str(ticks)})


def empty_library(evidence, path):
    """Even unindexed packets/origins prevent a zero-publication certificate."""
    path = evidence.path(path)
    require(path.parent.is_dir(), "Failed library parent is unavailable")
    if not path.exists():
        return [{"path": ".", "kind": "absent"}]
    require(path.is_dir(), "Failed library path is not a directory")
    inventory = []
    for item in sorted(path.rglob("*")):
        item = evidence.path(item, inside=path)
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            inventory.append({"path": relative, "kind": "directory"})
        else:
            require(relative == ".library.lock", "Failed unit contains a possible partial publication")
            require(evidence.read_bytes(item) == b"", "Unexpected failed library lock content")
            inventory.append({"path": relative, "kind": "empty_lock"})
    return inventory


def unpublished_failure(evidence, unit, plan, campaign):
    """Admit only a closed single-batch failure with an independently empty library."""
    require(len(plan["batches"]) == len(campaign["batches"]) == 1, "Single matched batch required")
    specification = plan["batches"][0]
    name = specification["id"]
    require(set(campaign["batches"]) == {name}, "Failed batch identity differs")
    batch = campaign["batches"][name]
    require(batch["status"] == "failed" and batch["attempts"], "Recorded failed attempt required")
    require(
        all(a["status"] == "failed" and a.get("publication") is None for a in batch["attempts"]),
        "Failure contains a publication or mixed attempts",
    )
    for key in RESOURCES:
        amounts = [a["charged"][key] for a in batch["attempts"]]
        require(
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in amounts),
            "Invalid failed unit charge",
        )
        require(
            campaign["usage"][key] == sum(amounts),
            "Failed unit reservation accounting differs",
        )
    folder = Path(unit["plan"]).parent
    library = evidence.path(unit["library"], inside=folder)
    require(
        campaign["definition"]["library_root"] == str(library) and library == folder / "model_library",
        "Failed unit library binding differs",
    )
    inventory = empty_library(evidence, library)
    goal = evidence.path(folder / "campaign/goals" / name, inside=folder)
    state = evidence.read(goal / "goal.json")
    protocol = evidence.read(goal / "protocol.json")
    require(
        state["inflight"] is None and state["status"] in {"completed", "budget_exhausted"},
        "Failed unit goal has not finished acquisition",
    )
    require(
        state["definition_sha256"] == digest(state["definition"])
        and state["definition"]["admission"] == protocol["admission"],
        "Failed goal admission changed",
    )
    tasks = {t["id"]: t for t in specification["tasks"]}
    require(
        {t["id"]: t["task_sha256"] for t in protocol["targets"]} == {k: digest(t) for k, t in tasks.items()},
        "Failed goal task allocation differs",
    )
    require(
        protocol["admission"]["shared_training_history"] == specification["shared_training_history"],
        "Failed goal history policy differs",
    )
    return {
        "classification": "unpublished_failure",
        "allocated_tasks": list(tasks),
        "published_models": 0,
        "original_batch_status": batch["status"],
        "original_attempts": batch["attempts"],
        "charged_usage": campaign["usage"],
        "goal_checkpoint_usage": state["usage"],
        "goal_status": state["status"],
        "library_inventory": inventory,
        "accepted_checkpoint_targets_not_counted_as_publications": len(state.get("progress", {}).get("targets", {})),
        "failure_metric_policy": "Keep all allocated tasks; no Scientist predictions or imputed quality scores.",
    }


def inspect_study(study, launch_sha256, *, repository):
    """Inspect immutable allocations and actual process/registry state before scoring."""
    evidence = Evidence(repository)
    study = evidence.path(study)
    launch = evidence.read(study / "launch_basis.json", launch_sha256)
    design = evidence.read(study / "study_design.json", launch["study_design_sha256"])
    require(launch["final_protocol_sha256"] == design["final_protocol_sha256"], "Final protocol pin differs")
    runners = sorted(study.glob("runner_identity*.json"))
    require(runners and study / "runner_identity.json" in runners, "Missing original runner identity")
    for path in runners:
        identity = evidence.read(path)
        require(not process_live(identity), "Acquisition study runner is live")
    snapshot = evidence.read(study / "snapshot.json")
    require(
        digest(snapshot["files"]) == snapshot["sha256"] == design["snapshot_sha256"],
        "Frozen runtime manifest differs",
    )
    runtime = evidence.path(design["repository"], inside=study)
    for relative, pin in snapshot["files"].items():
        evidence.read_bytes(runtime / relative, pin, inside=runtime)
    plans, contexts, units = {}, {}, {}
    for unit in design["units"]:
        folder = evidence.path(study / unit["id"], inside=study)
        require(folder.parent == study, "Unit must be a direct study child")
        require(
            unit["campaign"] == str(folder / "campaign") and unit["plan"] == str(folder / "plan.json"),
            "Matched unit paths differ",
        )
        plan = evidence.read(unit["plan"], unit["plan_sha256"], inside=folder)
        runner = evidence.read(folder / "runner_identity.json")
        require(runner["campaign"] == unit["campaign"], "Unit runner binding differs")
        campaign = evidence.read(folder / "campaign/campaign.json")
        require(
            not process_live(runner)
            and campaign["owner"] is None
            and campaign["inflight"] is None
            and campaign["status"] in {"completed", "needs_tasks", "budget_exhausted"},
            "Matched acquisition unit remains active",
        )
        failed = any(b["status"] == "failed" for b in campaign["batches"].values())
        if failed:
            fact = unpublished_failure(evidence, unit, plan, campaign)
            terminal = {
                "status": campaign["status"],
                "audit_id": campaign["last_audit"],
                "usage": campaign["usage"],
                "unpublished_failure": True,
                "classification": fact["classification"],
            }
        else:
            require_terminal(campaign, runner_live=False)
            terminal = evidence.read(folder / "terminal_summary_v1.json")
            evidence.read_bytes(folder / "campaign/campaign.json", terminal["campaign_sha256"])
            require(campaign["last_audit"] == terminal["audit_id"], "Unit audit binding differs")
            fact = {"classification": "acquisition_terminal", "original_status": campaign["status"]}
        require(campaign["last_audit"] is not None, "Missing publication audit")
        audit = evidence.read(folder / "campaign/audits" / (campaign["last_audit"] + ".json"))
        require(digest(audit) == campaign["last_audit"], "Publication audit identity differs")
        if failed:
            require(audit["verified_scope_count"] == 0 and not audit["batches"], "Failed audit includes publications")
        else:
            evidence.read_bytes(
                folder / "campaign/audits" / (campaign["last_audit"] + ".json"), terminal["audit_sha256"]
            )
        plans[unit["id"]] = plan
        contexts[unit["id"]] = (folder, campaign, terminal)
        units[unit["id"]] = fact
    tasks = validate_design(design, plans)
    protocol = evidence.read(study / "final_protocol.json", design["final_protocol_sha256"])
    require(protocol["task_bindings"] == {k: digest(t) for k, t in tasks.items()}, "Prospective tasks differ")
    declared = evidence.read(study / "tasks.json", design["tasks_sha256"])
    require(len(declared) == len(tasks) and {t["id"]: t for t in declared} == tasks, "Declared tasks differ")
    require(
        protocol["fit_partition"] == "training"
        and protocol["score_partition"] == "future"
        and protocol["hyperparameter_search"] is False,
        "Frozen scoring policy differs",
    )
    evidence.unchanged()
    for unit in design["units"]:
        if units[unit["id"]]["classification"] == "unpublished_failure":
            require(
                empty_library(evidence, unit["library"]) == units[unit["id"]]["library_inventory"],
                "Failed library changed during closeout",
            )
    facts = {
        "schema": SCHEMA,
        "study": str(study),
        "launch_basis_sha256": launch_sha256,
        "units": units,
        "allocated_scientific_scopes": len(tasks),
        "allocated_model_attempts": design["allocated_model_attempts"],
        "source_bindings": evidence.bindings,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_experiments_modified": False,
        "final_labels_scored": False,
        "external_model_calls": 0,
    }
    return facts, design, plans, contexts


def prepare(study, launch_sha256, *, repository):
    facts, _, _, _ = inspect_study(study, launch_sha256, repository=repository)
    return {"utc": datetime.datetime.now(datetime.UTC).isoformat(), **facts}


def verify(study, reference, *, repository):
    evidence = Evidence(repository)
    saved = evidence.read(reference["path"], reference["sha256"], inside=study)
    facts, design, plans, contexts = inspect_study(study, saved["launch_basis_sha256"], repository=repository)
    require({k: v for k, v in saved.items() if k != "utc"} == facts, "Terminal closeout evidence changed")
    evidence.unchanged()
    return facts, design, plans, contexts
