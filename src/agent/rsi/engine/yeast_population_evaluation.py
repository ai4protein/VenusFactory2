"""Six predeclared expression controls, scored only after campaign completion."""

from pathlib import Path

from agent.rsi import campaign_evaluation as core
from agent.rsi import model_tasks, yeast_population_controls
from .prediction_metrics import measure_context
from .protein_model import digest
from agent.rsi.campaigns.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore, allocation_execution, execution_policy
from .storage import atomic_json, safe_child, sha256_file

METHODS = (
    "training_mean",
    "condition_ridge",
    "condition_random_forest",
    "condition_svr",
    "transcript_only_ols",
    "culture_metadata_ols",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_selected_tasks(definition, protocol):
    """Bind unique selected tasks to either single runs or complete disjoint pairs."""
    selected = protocol["selection"]["selected_tasks"]
    require(len(selected) == len(set(selected)), "Selected task identities must be unique")
    allocations = definition["allocations"]
    pairs = definition.get("pairs", [])
    if not pairs:
        observed = [a["task"]["id"] for a in allocations.values()]
    else:
        core.paired_design(allocations, pairs)
        seen = set()
        observed = []
        for pair in pairs:
            members = (pair["left"], pair["right"])
            require(not seen.intersection(members), "An allocation belongs to multiple pairs")
            seen.update(members)
            tasks = [allocations[name]["task"]["id"] for name in members]
            require(tasks[0] == tasks[1], "A matched pair must share its task")
            observed.append(tasks[0])
        require(seen == set(allocations), "Every allocation must belong to a declared pair")
    require(sorted(observed) == sorted(selected), "Selected task membership changed")


def load_binding(pilot):
    pilot = Path(pilot).absolute()
    preparation = core.read(pilot / "preparation.json")
    require(sha256_file(pilot / "plan.json") == preparation["plan_sha256"], "Prepared plan changed")
    require(sha256_file(pilot / "protocol.json") == preparation["protocol_sha256"], "Prepared protocol changed")
    protocol = core.read(pilot / "protocol.json")
    declared = protocol["final_controls"]
    require(tuple(declared["methods"]) == METHODS, "Six predeclared expression controls required")
    require(
        declared["implementation"] == "src/agent/rsi/engine/yeast_population_controls.py"
        and declared["sha256"] == sha256_file(Path(yeast_population_controls.__file__)),
        "Use the predeclared numerical control implementation",
    )
    snapshot = core.read(pilot / "snapshot.json")["files"]
    repository = pilot / "repository"
    require(snapshot == core.source_identity(), "Run with the frozen campaign implementation")
    require(
        all(sha256_file(safe_child(repository, name)) == expected for name, expected in snapshot.items()),
        "Frozen source snapshot changed",
    )
    study = pilot / "evaluation_study"
    definition = core.load_study(study)
    require(str(study) == preparation["study"], "Study location differs from preparation")
    require(definition["campaign"] == preparation["campaign"], "Campaign location differs from preparation")
    require(
        len(definition["allocations"]) == protocol["allocation_count"] == preparation["allocation_count"],
        "Allocation count changed",
    )
    verify_selected_tasks(definition, protocol)
    for allocation in definition["allocations"].values():
        task = allocation["task"]
        require(
            task["plugin_id"] == "generated_context.v1"
            and task["scope"]["assay"] == "population_protein_expression"
            and task["scope"]["grouping"]["level"] == "genetic_population",
            "Only declared population-expression tasks are supported",
        )
    return definition, ModelCampaignStore(definition["campaign"]), protocol


def require_terminal(definition, state):
    require(
        state["definition_sha256"] == definition["campaign_definition_sha256"]
        and state["plans"] == definition["plans"],
        "Campaign design changed since admission",
    )
    require(state["owner"] is None and state["inflight"] is None, "Campaign execution is unresolved")
    require(
        execution_policy(state["definition"]) == definition["execution_policy"] == ALL_ALLOCATIONS
        and allocation_execution(state)["execution_complete"],
        "Every allocated task needs a terminal execution attempt before final scoring",
    )


def score_task(task, core_row, core_output, output):
    data = model_tasks.load_task(task)
    training, scoring = data["training"], data["future"]
    outcomes = yeast_population_controls.controls(training, scoring, task["scope"])
    require(tuple(row["method"] for row in outcomes) == METHODS, "Control method membership changed")
    require(all(row["status"] == "completed" for row in outcomes), "Every declared control must be evaluated")
    yeast_population_controls.verify_metrics(training, scoring, outcomes)
    archived = core_row["context_controls"]["outcomes"]
    require(
        len(archived) == 4 and {row["method"] for row in archived} == set(METHODS[:4]),
        "Core evaluation lacks the four fixed controls",
    )
    archived = {row["method"]: row for row in archived}
    results = []
    for outcome in outcomes:
        predictions = [
            {
                "id": row["id"],
                "protein": row["protein"],
                "group": row["group"],
                "values_sha256": digest(row["values"]),
                "prediction": item["prediction"],
            }
            for row, item in zip(scoring, outcome["predictions"], strict=True)
        ]
        metrics = measure_context(training, scoring, predictions, scope=task["scope"])
        if outcome["method"] in archived:
            previous = archived[outcome["method"]]
            require(previous["status"] == "completed", "Core control did not complete")
            reference = previous["predictions"]
            saved = safe_child(core_output, reference["path"])
            require(sha256_file(saved) == reference["sha256"], "Core predictions changed")
            require(core.read(saved) == predictions, "Core control predictions differ from replay")
            require(previous["metrics"] == metrics, "Core control metrics differ from replay")
        file = output / "control_predictions" / core_row["allocation_id"] / (outcome["method"] + ".json")
        atomic_json(file, predictions)
        results.append(
            {
                "method": outcome["method"],
                "status": "completed",
                "metrics": metrics,
                "predictions": {"path": str(file.relative_to(output)), "sha256": sha256_file(file)},
            }
        )
    best = min(results, key=lambda row: (row["metrics"]["rmse"], row["method"]))
    final_metrics = core_row["final_metrics"]
    difference = final_metrics["rmse"] - best["metrics"]["rmse"] if final_metrics is not None else None
    return {
        "allocation_id": core_row["allocation_id"],
        "task_id": task["id"],
        "original_status": core_row["original_status"],
        "publication_and_reuse_success": core_row["publication_and_reuse_success"],
        "final_metrics": final_metrics,
        "controls": results,
        "best_control": best["method"],
        "final_model_minus_best_rmse": difference,
        "final_model_better_than_all_controls": difference < -1e-10 if difference is not None else None,
    }


def evaluate(pilot, output):
    pilot, output = Path(pilot).absolute(), Path(output).absolute()
    definition, store, protocol = load_binding(pilot)
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Symlinked evaluation path")
    require(
        output != store.root and store.root not in output.parents and output not in store.root.parents,
        "Evaluation must be outside the campaign",
    )
    # Fail before creating output or opening numerical task data for live campaigns.
    with store.lease():
        state = store.read()
        require_terminal(definition, state)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "yeast_population_final_controls.v1",
        "status": "running",
        "study_sha256": digest(definition),
        "protocol_sha256": sha256_file(pilot / "protocol.json"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "numeric_implementation_sha256": protocol["final_controls"]["sha256"],
        "fit_partition": "training",
        "score_partition": "future",
        "rows": [],
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    atomic_json(output / "report.json", report)
    try:
        core_output = output / "core_evaluation"
        scored = core.evaluate(pilot / "evaluation_study", core_output)
        require(scored["status"] == "completed", "Core evaluation failed evidence verification")
        audited = core.audit(pilot / "evaluation_study", core_output, output / "core_audit")
        require(audited["status"] == "passed", "Core publication and prediction replay failed")
        barrier = core.read(core_output / "development_barrier.json")
        require(barrier["campaign_state"] == state, "Campaign changed before final evaluation")
        rows = scored["rows"]
        require(
            len(rows) == len(definition["allocations"])
            and {row["allocation_id"] for row in rows} == set(definition["allocations"]),
            "Core evaluation must retain every allocated task",
        )
        with store.lease():
            require(store.read() == state, "Campaign changed after evaluation")
            for row in rows:
                allocation = definition["allocations"][row["allocation_id"]]
                report["rows"].append(score_task(allocation["task"], row, core_output, output))
                atomic_json(output / "report.json", report)
            require(store.read() == state, "Campaign changed during control evaluation")
            require(
                all(core.tree(store.goal(name)) == saved for name, saved in barrier["goal_files"].items()),
                "Goal evidence changed during control evaluation",
            )
        report.update(
            status="completed",
            allocated=len(rows),
            completed_controls=6 * len(rows),
            scored_models=sum(row["final_metrics"] is not None for row in rows),
            publication_and_reuse_successes=sum(row["publication_and_reuse_success"] for row in rows),
            core_report_sha256=sha256_file(core_output / "report.json"),
            core_audit_sha256=sha256_file(output / "core_audit/audit.json"),
            interpretation="Shared population cohort. Missing models retain null scores; failed tasks remain in the denominator. Control fits are not autonomous models.",
        )
        atomic_json(output / "report.json", report)
        return report
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output / "report.json", report)
        raise
