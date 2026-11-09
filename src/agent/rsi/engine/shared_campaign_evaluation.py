"""Descriptive assessment of every scope in a frozen shared Scientist campaign."""

from pathlib import Path

from agent.harness.contracts import digest, file_hash
from agent.rsi.campaigns.shared_model_campaign import SharedCampaignStore, bind_goal
from agent.rsi.datasets import model_tasks

from .campaign_evidence import observed_usage, read, require, tree
from .campaign_scoring import context_controls, score_model
from .scientist_model_library import ScientistModelLibrary
from .shared_campaign_study import load, outside, publication_binding, verify
from .storage import atomic_json, safe_child


def assess_batch(definition, store, library, batch_id, output):
    frozen = definition["snapshot"]
    specification = frozen["specifications"][batch_id]
    record = store.read()["batches"][batch_id]
    root = safe_child(store.root, "goals/" + batch_id)
    models = frozen["publications"][batch_id]["models"]
    result = {
        "batch_id": batch_id,
        "original_batch_status": record["status"],
        "original_goal_status": frozen["goals"][batch_id]["status"],
        "cost": observed_usage(root, record),
        "rows": [],
    }
    verified, evidence_error = {}, None
    try:
        if (root / "goal.json").exists():
            bind_goal(store, specification, root)
        else:
            require(not models, "Missing goal has published models")
        if models:
            verified = library.resolve_many([row["entry_id"] for row in models.values()], replay_training=True)
            for binding in models.values():
                require(
                    publication_binding(verified[binding["entry_id"]]) == binding,
                    "Numerical verification differs from frozen publication",
                )
    except Exception as exc:
        evidence_error = {"error_type": type(exc).__name__, "error": str(exc)}
    for task in specification["tasks"]:
        identity = task["id"]
        binding = models.get(identity)
        row = {
            "batch_id": batch_id,
            "task_id": identity,
            "stage": task["stage"],
            "protein": task["scope"]["protein"],
            "target": task["scope"]["target"],
            "assay": task["scope"]["assay"],
            "study": task["paper"]["doi"],
            "harness": specification["profile"]["harness"],
            "entry_id": binding["entry_id"] if binding else None,
            "verified_callable": False,
            "evaluation_status": "no_published_model",
            "final_metrics": None,
        }
        try:
            data = model_tasks.load_task(task)
            if evidence_error:
                row.update(evaluation_status="evidence_failed", **evidence_error)
            elif binding:
                model = verified[binding["entry_id"]]["packet"]["model"]
                scored = score_model(task, data, model, output, output / "predictions" / (identity + ".json"))
                row.update(
                    verified_callable=True,
                    evaluation_status=scored.pop("status"),
                    final_metrics=scored.pop("metrics"),
                    **scored,
                )
        except Exception as exc:
            row.update(evaluation_status="evidence_failed", error_type=type(exc).__name__, error=str(exc))
            data = None
        # Controls cover failed and unpublished acquisitions too. They cannot repair publication evidence.
        try:
            require(data is not None, "Task data failed validation")
            row["context_controls"] = context_controls(task, data, output, identity, independent_refit=True)
        except Exception as exc:
            row["context_controls"] = {"status": "evidence_failed", "error_type": type(exc).__name__, "error": str(exc)}
        result["rows"].append(row)
    result["all_allocated_scopes_verified"] = all(row["verified_callable"] for row in result["rows"])
    result["completed_goal_with_all_scopes_verified"] = (
        result["original_goal_status"] == "completed"
        and result["original_batch_status"] == "completed"
        and result["all_allocated_scopes_verified"]
    )
    return result


def artifact_bindings(record):
    result = []
    for row in record["rows"]:
        if "predictions" in row:
            result.append(row["predictions"])
        result.extend(
            item["predictions"] for item in row["context_controls"].get("outcomes", []) if "predictions" in item
        )
    return result


def checkpoint(path, definition, batch_id, output):
    saved = read(path)
    record = saved["record"]
    require(
        saved["sha256"] == digest(record) and saved["study_sha256"] == digest(definition),
        "Shared checkpoint identity changed",
    )
    require(
        record["batch_id"] == batch_id
        and [row["task_id"] for row in record["rows"]]
        == [task["id"] for task in definition["snapshot"]["specifications"][batch_id]["tasks"]],
        "Checkpoint allocation coverage changed",
    )
    for artifact in artifact_bindings(record):
        require(file_hash(safe_child(output, artifact["path"])) == artifact["sha256"], "Checkpoint predictions changed")
    return record


def summarize(definition, batches, complete):
    rows = [row for batch in batches for row in batch["rows"]]
    failed = any(
        row["evaluation_status"] == "evidence_failed" or row["context_controls"]["status"] == "evidence_failed"
        for row in rows
    )
    return {
        "schema": "shared_campaign_descriptive_evaluation.v1",
        "study_sha256": digest(definition),
        "status": ("evidence_failed" if failed else "completed") if complete else "running",
        "allocated": len(definition["snapshot"]["tasks"]),
        "assessed": len(rows),
        "verified_callable": sum(row["verified_callable"] for row in rows),
        "future_scored": sum(row["final_metrics"] is not None for row in rows),
        "without_scored_model": sum(row["final_metrics"] is None for row in rows),
        "controls_completed": sum(row["context_controls"]["status"] == "completed" for row in rows),
        "original_campaign_status": definition["snapshot"]["original_campaign_status"],
        "batches": batches,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "cost_attribution": definition["cost_attribution"],
        "interpretation": definition["interpretation"],
    }


def evaluate(study, output, *, resume=False, progress=None):
    definition = load(study)
    store = SharedCampaignStore(definition["campaign"])
    library = ScientistModelLibrary(definition["library"])
    output = outside(output, store.root, library.root, Path(study))
    with store.lease():
        verify(definition, store, library)
        barrier = {"schema": 1, "study_sha256": digest(definition), "final_scoring_started": False}
        if resume:
            require(read(output / "development_barrier.json") == barrier, "Shared scoring barrier changed")
        else:
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(output / "development_barrier.json", barrier)
        batches = []
        for batch_id in definition["snapshot"]["specifications"]:
            path = output / "batches" / (batch_id + ".json")
            if path.exists():
                require(resume, "Unexpected existing batch checkpoint")
                result = checkpoint(path, definition, batch_id, output)
            else:
                result = assess_batch(definition, store, library, batch_id, output)
                atomic_json(path, {"study_sha256": digest(definition), "record": result, "sha256": digest(result)})
            batches.append(result)
            report = summarize(definition, batches, False)
            atomic_json(output / "report.json", report)
            if progress is not None:
                progress(
                    {
                        "batch": batch_id,
                        "assessed": report["assessed"],
                        "allocated": report["allocated"],
                        "future_scored": report["future_scored"],
                    }
                )
        verify(definition, store, library)
        require(load(study) == definition, "Shared study changed during scoring")
        report = summarize(definition, batches, True)
        require(report["assessed"] == report["allocated"], "Not every shared allocation was assessed")
        atomic_json(output / "report.json", report)
        return report


def audit(study, evaluation, output):
    """Recompute in a new folder; failed evidence remains failed, never silently repaired."""
    expected = read(Path(evaluation) / "report.json")
    require(expected["status"] == "completed", "Only complete evidence can pass replay")
    actual = evaluate(study, output)
    require(actual == expected, "Shared numerical replay differs")
    require(tree(Path(evaluation)) == tree(Path(output)), "Shared prediction artifacts differ")
    return {
        "status": "passed",
        "allocated": actual["allocated"],
        "future_scored": actual["future_scored"],
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
