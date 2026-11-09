"""Prospective, all-allocation evaluation of persistent Scientist campaigns.

Publication proofs and prediction interfaces are reused. Future metrics are
calculated separately, after every campaign allocation reaches a terminal state.
No credentials, model service, training decision or library write is needed.
"""

from __future__ import annotations

import copy
from pathlib import Path

from agent.harness.contracts import digest, file_hash
from agent.rsi import context_feedback_dispatch as context_feedback
from agent.rsi import library_experience, model_tasks, training_attempt_memory
from agent.rsi.campaigns.scientist_model_campaign import (
    ALL_ALLOCATIONS,
    TERMINAL,
    ModelCampaignStore,
    allocation_execution,
    bind_goal,
    execution_policy,
)

from .campaign_evidence import (
    observed_usage as observed_usage,
)
from .campaign_evidence import (
    read as read,
)
from .campaign_evidence import (
    require as require,
)
from .campaign_evidence import (
    source_identity as source_identity,
)
from .campaign_evidence import (
    tree as tree,
)
from .campaign_scoring import context_controls as context_controls
from .campaign_scoring import score_model
from .model_plugins import get_model_plugin
from .scientist_model_library import ScientistModelLibrary, validate_model_goal
from .storage import atomic_json, safe_child

LIBRARY_RETRIEVAL = "library_retrieval.v1"


def paired_design(specifications, pairs):
    require(isinstance(pairs, list), "Comparisons must be explicitly listed")
    seen = set()
    for pair in pairs:
        require(set(pair) == {"id", "left", "right", "factor"}, "Invalid comparison fields")
        require(isinstance(pair["id"], str) and pair["id"] and pair["id"] not in seen, "Duplicate comparison")
        seen.add(pair["id"])
        require(
            pair["left"] != pair["right"] and {pair["left"], pair["right"]} <= set(specifications),
            "Unallocated comparison",
        )
        require(
            pair["factor"]
            in {"library_experience", LIBRARY_RETRIEVAL, context_feedback.FIELD, training_attempt_memory.FIELD},
            "Unsupported matched comparison factor",
        )
        left, right = [copy.deepcopy(specifications[pair[k]]) for k in ("left", "right")]
        if pair["factor"] == training_attempt_memory.FIELD:
            require(
                all(training_attempt_memory.FIELD in row for row in (left, right)),
                "Both memory conditions must be explicitly declared",
            )
            require(
                len({training_attempt_memory.policy_for(row) for row in (left, right)}) == 2,
                "Compare two distinct training-attempt memory policies",
            )
            for row in (left, right):
                row.pop("id")
                row.pop(training_attempt_memory.FIELD)
        elif pair["factor"] == LIBRARY_RETRIEVAL:
            require("experience" in left and "experience" in right, "Both retrieval arms require explicit experience")
            exposures = []
            for row in (left, right):
                policy = library_experience.campaign_policy(row["experience"])
                require(policy["selection"] == "explicit", "Retrieval comparisons require frozen source selections")
                require(policy["condition"] in {"none", "source"}, "Retrieval arms expose source or no source")
                exposures.append(set(policy["entry_ids"]) if policy["condition"] == "source" else set())
            require(exposures[0] != exposures[1], "Retrieval source exposures are identical")
            for row in (left, right):
                row.pop("id")
                row["experience"].pop("condition")
                row["experience"].pop("entry_ids")
        elif pair["factor"] == context_feedback.FIELD:
            policies = [context_feedback.policy_for(row, plugin_id=row["task"]["plugin_id"]) for row in (left, right)]
            require(
                len(set(policies)) == 2,
                "Compare two distinct declared training-baseline policies",
            )
            for row in (left, right):
                row.pop("id")
                row.pop(context_feedback.FIELD, None)
        else:
            require("experience" in left and "experience" in right, "Both conditions require admitted experience")
            require(
                left["experience"]["selection"] == right["experience"]["selection"] == "explicit",
                "Matched comparisons require fixed donors",
            )
            require(
                left["experience"]["condition"] != right["experience"]["condition"],
                "Comparison conditions are identical",
            )
            for row in (left, right):
                row.pop("id")
                row["experience"].pop("condition")
        require(left == right, "Task, donors, profile, lineage or budgets differ within comparison")
    return copy.deepcopy(pairs)


def admit(campaign, output, *, pairs=(), allocation_ids=None):
    store = ModelCampaignStore(campaign)
    output = Path(output).absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Study path traverses a symlink")
    require(
        store.root not in output.parents and output not in store.root.parents and output != store.root,
        "Study must be outside the campaign",
    )
    with store.lease():
        state = store.read()
        require(state["owner"] is None and state["inflight"] is None, "Admit a quiescent campaign")
        require(
            execution_policy(state["definition"]) == ALL_ALLOCATIONS,
            "Prospective studies require all_allocations.v1 execution; coverage-target stopping can omit controls",
        )
        specifications = store.specifications(state)
        ids = list(specifications) if allocation_ids is None else list(allocation_ids)
        require(ids and len(ids) == len(set(ids)) and set(ids) <= set(specifications), "Invalid study allocations")
        for identity in ids:
            item = state["allocations"][identity]
            model_tasks.load_task(specifications[identity]["task"])
            require(
                item["status"] == "queued" and not item["attempts"] and not item["entries"],
                "Study admission must precede selected executions",
            )
            require(not store.goal(identity).exists(), "Study allocation already has a goal")
        required_resources = dict.fromkeys(("model_calls", "tool_calls", "wall_seconds"), 0)
        for identity, allocation in specifications.items():
            if state["allocations"][identity]["status"] in TERMINAL | {"cancelled"}:
                continue
            root = store.goal(identity)
            before = (
                bind_goal(allocation, root)["usage"]
                if (root / "goal.json").exists()
                else dict.fromkeys(required_resources, 0)
            )
            remaining = {k: max(0, allocation["goal_budget"][k] - before[k]) for k in required_resources}
            require(all(n > 0 for n in remaining.values()), "Unfinished study allocation has no remaining budget")
            for key, amount in remaining.items():
                required_resources[key] += amount
        available_resources = {k: state["limits"][k] - state["usage"][k] for k in required_resources}
        require(
            all(required_resources[k] <= available_resources[k] for k in required_resources),
            "Study budget cannot reserve all unfinished allocations",
        )
        selected = {identity: specifications[identity] for identity in ids}
        definition = {
            "schema": 1,
            "campaign": str(store.root),
            "campaign_definition_sha256": state["definition_sha256"],
            "execution_policy": ALL_ALLOCATIONS,
            "execution_resources": {
                "required_remaining_reservations": required_resources,
                "available_at_admission": available_resources,
            },
            "plans": state["plans"],
            "allocations": selected,
            "pairs": paired_design(selected, list(pairs)),
            **(
                {"include_context_controls": True}
                if any(
                    a["task"]["plugin_id"] in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}
                    for a in selected.values()
                )
                else {}
            ),
            "evaluator_sources": source_identity(),
            "plugin_sources": {
                a["task"]["plugin_id"]: get_model_plugin(a["task"]["plugin_id"]).source_sha256
                for a in selected.values()
            },
            "selection": "Final independently published generation in chronological lineage order; no selection by future scores",
            "primary_feasibility": "Completed allocation with independently verified publication and fresh-session reuse",
            "performance": "Conditional future metrics; failed or missing models retain null scores and remain in the feasibility denominator",
        }
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "study.json", {"definition": definition, "sha256": digest(definition)})
    return definition


def load_study(output):
    bound = read(Path(output) / "study.json")
    definition = bound["definition"]
    require(digest(definition) == bound["sha256"] and definition["schema"] == 1, "Study definition changed")
    require(definition["evaluator_sources"] == source_identity(), "Use the admitted evaluation source")
    for plugin, expected in definition["plugin_sources"].items():
        require(get_model_plugin(plugin).source_sha256 == expected, "Prediction implementation changed")
    paired_design(definition["allocations"], definition["pairs"])
    return definition


def admit_descriptive(campaign, output):
    """Freeze all terminal acquisitions for descriptive scoring, not prospective inference.

    This path cannot select arms, register contrasts, resume development or replace
    the original prospective-admission rule. Every allocated run must have finished.
    """
    store = ModelCampaignStore(campaign)
    output = Path(output).absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Study path traverses a symlink")
    require(
        store.root not in output.parents and output not in store.root.parents and output != store.root,
        "Study must be outside the campaign",
    )
    with store.lease():
        state = store.read()
        require(state["owner"] is None and state["inflight"] is None, "Campaign execution is unresolved")
        require(
            all(r["status"] in TERMINAL and bool(r["attempts"]) for r in state["allocations"].values()),
            "Descriptive admission requires every allocated execution to be terminal",
        )
        specifications = store.specifications(state)
        require(bool(specifications), "No allocated tasks")
        for allocation in specifications.values():
            model_tasks.load_task(allocation["task"])
        definition = {
            "schema": 1,
            "admission_protocol": "terminal_descriptive.v1",
            "campaign": str(store.root),
            "campaign_definition_sha256": state["definition_sha256"],
            "execution_policy": execution_policy(state["definition"]),
            "plans": state["plans"],
            "allocations": specifications,
            "pairs": [],
            "evaluator_sources": source_identity(),
            "plugin_sources": {
                a["task"]["plugin_id"]: get_model_plugin(a["task"]["plugin_id"]).source_sha256
                for a in specifications.values()
            },
            "terminal_campaign_state_sha256": digest(state),
            "terminal_goal_files": {identity: tree(store.goal(identity)) for identity in specifications},
            "include_context_controls": True,
            "selection": "Final independently published generation in chronological lineage order; all published generations scored",
            "primary_feasibility": "Completed allocation with all requested generations published and freshly reused",
            "performance": "Post-acquisition descriptive evaluation of every allocation; no prospective or causal RSI comparison",
        }
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "study.json", {"definition": definition, "sha256": digest(definition)})
    return definition


def published_models(store, identity, allocation, record):
    """Score this allocation's model, even when publication deduplicated its key."""
    root = store.goal(identity)
    publications = record.get("generation_publications", []) + [
        a["publication"] for a in record["attempts"] if a.get("publication")
    ]
    if not publications:
        require(
            record["status"] != "completed" and not record["entries"], "Completed allocation lacks publication evidence"
        )
        return []
    state = bind_goal(allocation, root)
    library = ScientistModelLibrary(store.read()["definition"]["library_root"])
    outcomes = {}
    for report in publications:
        stored = library._object("publication_attempts", report["report_id"])
        require(stored == {k: v for k, v in report.items() if k != "report_id"}, "Publication attempt changed")
        require(
            stored["kind"] in {"model_goal_publication_attempt", "model_generation_publication_attempt"},
            "Unknown publication kind",
        )
        origin = library.goal(safe_child(library.owner, stored["origin_goal"]))
        if stored["kind"] == "model_generation_publication_attempt":
            from .generation_publication import bind_parent

            bind_parent(origin, root)
        else:
            require(origin == root, "Publication belongs to another allocation")
        require(stored["goal_sha256"] == file_hash(origin / "goal.json"), "Publication goal changed")
        for outcome in stored["outcomes"]:
            if outcome["status"] == "published":
                mid = outcome["model_id"]
                if mid in outcomes:
                    previous, _ = outcomes[mid]
                    require(
                        all(previous[k] == outcome[k] for k in ("model_id", "entry_id", "key_id")),
                        "Conflicting publication outcomes",
                    )
                outcomes[mid] = (outcome, origin)
    require(
        set(record["entries"]) == {x["entry_id"] for x, _ in outcomes.values()},
        "Recorded publication inventory changed",
    )
    lineage = state["progress"].get("generations") or [state["progress"]["accepted_model"]]
    require(set(outcomes) <= {r["model_id"] for r in lineage}, "Published model is outside the goal lineage")
    if record["status"] == "completed":
        require(
            len(outcomes) == len(lineage) == allocation["generations"],
            "Completed allocation has incomplete publications",
        )
    result = []
    for generation, row in enumerate(lineage, 1):
        mid = row["model_id"]
        if mid not in outcomes:
            continue
        outcome, origin = outcomes[mid]
        own = validate_model_goal(origin, mid, replay_training=True)
        canonical = library.resolve(outcome["entry_id"], replay_training=True)
        require(own["key"] == canonical["key"], "Deduplicated publication has another model key")
        key_id = digest(own["key"])
        expected = {
            **own["origin"],
            "goal": str(origin.relative_to(library.owner)),
            "key_id": key_id,
            "packet_id": digest(own["packet"]),
        }
        require(outcome["key_id"] == key_id, "Publication key differs")
        require(library._object("origins/" + key_id, outcome["origin_id"]) == expected, "Supporting origin changed")
        require(library._object("packets", expected["packet_id"]) == own["packet"], "Published fitted artifact differs")
        result.append(
            {
                "generation": generation,
                "model_id": mid,
                "entry_id": outcome["entry_id"],
                "canonical_origin_is_this_goal": canonical["origin"]
                .get("generation_milestone", {})
                .get("parent_goal", canonical["origin_goal"])
                == str(root),
                "model": own["packet"]["model"],
            }
        )
    return result


def contrasts(rows, pairs):
    table = {row["allocation_id"]: row for row in rows}
    result = []
    for pair in pairs:
        left, right = table[pair["left"]], table[pair["right"]]
        lm, rm = left.get("final_metrics"), right.get("final_metrics")
        delta = {}
        for name in (
            "mse_skill_vs_training_mean",
            "mae",
            "spearman",
            "brier_skill_vs_training_prior",
            "log_loss",
            "balanced_accuracy",
        ):
            a, b = (m.get(name) if m else None for m in (lm, rm))
            delta[name] = a - b if type(a) in (int, float) and type(b) in (int, float) else None
        complete_cost = all(r["cost"]["call_accounting_complete"] for r in (left, right))
        result.append(
            {
                **pair,
                "left_status": left["original_status"],
                "right_status": right["original_status"],
                "feasibility_difference": int(left["publication_and_reuse_success"])
                - int(right["publication_and_reuse_success"]),
                "both_scored": lm is not None and rm is not None,
                "metric_differences": delta,
                "physical_request_difference": left["cost"]["physical_requests_observed"]
                - right["cost"]["physical_requests_observed"]
                if complete_cost
                else None,
            }
        )
    return result


def evaluate(study, output):
    study, output = Path(study), Path(output).absolute()
    definition = load_study(study)
    store = ModelCampaignStore(definition["campaign"])
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Evaluation path traverses a symlink")
    require(
        output != store.root and store.root not in output.parents and output not in store.root.parents,
        "Evaluation must be outside the campaign",
    )
    with store.lease():
        state = store.read()
        require(
            state["definition_sha256"] == definition["campaign_definition_sha256"]
            and state["plans"] == definition["plans"],
            "Campaign design changed since study admission",
        )
        require(state["owner"] is None and state["inflight"] is None, "Campaign execution is unresolved")
        require(
            all(r["status"] in TERMINAL | {"cancelled"} for r in state["allocations"].values()),
            "Every campaign allocation must be terminal before future scoring",
        )
        if definition.get("execution_policy") == ALL_ALLOCATIONS:
            require(
                execution_policy(state["definition"]) == ALL_ALLOCATIONS
                and allocation_execution(state)["execution_complete"],
                "Every predeclared allocation requires a terminal execution attempt before future scoring",
            )
        specifications = store.specifications(state)
        for identity, allocated in definition["allocations"].items():
            require(specifications[identity] == allocated, "Study allocation changed")
            model_tasks.load_task(allocated["task"])
        frozen = {identity: tree(store.goal(identity)) for identity in definition["allocations"]}
        if definition.get("admission_protocol") == "terminal_descriptive.v1":
            require(
                digest(state) == definition["terminal_campaign_state_sha256"]
                and frozen == definition["terminal_goal_files"],
                "Terminal acquisition evidence changed after descriptive admission",
            )
        barrier = {
            "schema": 1,
            "study_sha256": digest(definition),
            "campaign_state": state,
            "goal_files": frozen,
            "future_scoring_started": False,
        }
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "development_barrier.json", barrier)
        report = {
            "schema": 1,
            "status": "running",
            "study_sha256": digest(definition),
            "barrier_sha256": digest(barrier),
            "rows": [],
            "external_model_calls": 0,
            "new_autonomous_models": 0,
        }
        if "admission_protocol" in definition:
            report["admission_protocol"] = definition["admission_protocol"]
        atomic_json(output / "report.json", report)
        for identity, allocation in definition["allocations"].items():
            record = state["allocations"][identity]
            root = store.goal(identity)
            row = {
                "allocation_id": identity,
                "task_id": allocation["task"]["id"],
                "stage": allocation["task"]["stage"],
                "harness": allocation["profile"]["harness"],
                "condition": allocation.get("experience", {}).get("condition"),
                "original_status": record["status"],
                "publication_and_reuse_success": False,
                "evaluation_status": "no_published_model",
                "generations": [],
                "final_metrics": None,
            }
            try:
                row["cost"] = observed_usage(root, record)
                models = published_models(store, identity, allocation, record)
                row["publication_and_reuse_success"] = (
                    record["status"] == "completed" and len(models) == allocation["generations"]
                )
                data = model_tasks.load_task(allocation["task"])
                plugin = get_model_plugin(allocation["task"]["plugin_id"])
                for item in models:
                    model = item.pop("model")
                    artifact = output / "predictions" / identity / (str(item["generation"]) + ".json")
                    item.update(score_model(allocation["task"], data, model, output, artifact))
                    row["generations"].append(item)
                if models:
                    row["evaluation_status"] = row["generations"][-1]["status"]
                    row["final_metrics"] = row["generations"][-1]["metrics"]
                    first, last = row["generations"][0]["metrics"], row["generations"][-1]["metrics"]
                    metric = plugin.acceptance_metric
                    a, b = (m.get(metric) if m else None for m in (first, last))
                    row["future_lineage_skill_change"] = (
                        b - a if len(models) > 1 and type(a) in (int, float) and type(b) in (int, float) else None
                    )
                if definition.get("include_context_controls") and plugin.id in {
                    "generated_context.v1",
                    "generated_context.v2",
                    "generated_context.v3",
                }:
                    row["context_controls"] = context_controls(allocation["task"], data, output, identity)
            except Exception as exc:
                row.update(
                    evaluation_status="evidence_failed",
                    error_type=type(exc).__name__,
                    publication_and_reuse_success=False,
                    final_metrics=None,
                )
            report["rows"].append(row)
            atomic_json(output / "report.json", report)
        require(store.read() == state, "Campaign changed during evaluation")
        require(
            all(tree(store.goal(i)) == files for i, files in frozen.items()), "Goal evidence changed during evaluation"
        )
        for allocation in definition["allocations"].values():
            model_tasks.load_task(allocation["task"])
        failed = any(r["evaluation_status"] == "evidence_failed" for r in report["rows"])
        report.update(
            status="evidence_failed" if failed else "completed",
            allocated=len(report["rows"]),
            publication_and_reuse_successes=sum(r["publication_and_reuse_success"] for r in report["rows"]),
            future_scored=sum(r["final_metrics"] is not None for r in report["rows"]),
            paired_contrasts=[] if failed else contrasts(report["rows"], definition["pairs"]),
            interpretation="All allocated runs retained. Scores are conditional on valid published models; missing scores are not zeros. Descriptive matched comparisons, not additional independent replicates.",
        )
        if definition.get("admission_protocol") == "terminal_descriptive.v1":
            report["interpretation"] = (
                definition["performance"]
                + ". Failed allocations retain their status; published partial lineages are scored without counting full-goal success."
            )
        atomic_json(output / "report.json", report)
        return report


def audit(study, evaluation, output):
    """Revalidate publications and independently replay a saved score report."""
    evaluation, output = Path(evaluation), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "external_model_calls": 0, "new_autonomous_models": 0}
    atomic_json(output / "audit.json", result)
    try:
        replay = evaluate(study, output / "replay")
        original = read(evaluation / "report.json")
        require(replay["status"] == "completed" and original == replay, "Saved evaluation differs from replay")
        require(
            read(evaluation / "development_barrier.json") == read(output / "replay/development_barrier.json"),
            "Development barrier changed",
        )
        for row in original["rows"]:
            artifacts = row["generations"] + row.get("context_controls", {}).get("outcomes", [])
            for item in artifacts:
                if "predictions" in item:
                    asset = item["predictions"]
                    path = safe_child(evaluation, asset["path"])
                    require(file_hash(path) == asset["sha256"], "Archived predictions changed")
                    require(
                        read(path) == read(safe_child(output / "replay", asset["path"])), "Prediction replay differs"
                    )
        result.update(
            status="passed",
            allocations=original["allocated"],
            paired_contrasts=len(original["paired_contrasts"]),
            report_sha256=file_hash(evaluation / "report.json"),
        )
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        atomic_json(output / "audit.json", result)
        raise
    atomic_json(output / "audit.json", result)
    return result
