"""Freeze every shared allocation and its current publication before scoring.

Admission hashes dataset files without decoding labels. Numerical/native replay
belongs to evaluation, after the complete cohort and evaluator are immutable.
"""

import platform
from importlib.metadata import version
from pathlib import Path

from agent.harness.contracts import digest
from agent.rsi.campaigns.shared_model_campaign import TERMINAL, SharedCampaignStore
from agent.rsi.datasets import model_tasks

from .campaign_evidence import read, require, source_identity, tree
from .campaign_scoring import SUPPORTED_SHARED_PLUGINS
from .model_plugins import get_model_plugin
from .scientist_model_library import ScientistModelLibrary
from .storage import atomic_json, safe_child

SCHEMA = "shared_campaign_descriptive_study.v1"


def versions():
    return {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")}}


def outside(output, *roots):
    output = Path(output).absolute()
    require(not any(p.is_symlink() for p in (output, *output.parents)), "Output path traverses a symlink")
    for root in roots:
        root = Path(root).absolute()
        require(
            output != root and root not in output.parents and output not in root.parents,
            "Output must be outside acquisition and library evidence",
        )
    return output


def publication_binding(value):
    """Keep a compact identity, not a second copy of the numerical model."""
    return {key: value[key] for key in ("entry_id", "key_id", "origin_goal", "key", "origin")} | {
        "packet_sha256": digest(value["packet"]),
    }


def publications(library, store, specifications):
    goals = {str(safe_child(store.root, "goals/" + batch)): batch for batch in specifications}
    result = {batch: {"reports": {}, "models": {}} for batch in specifications}
    expected = {batch: {} for batch in specifications}
    for path in sorted((library.root / "publication_attempts").glob("*.json")):
        report = read(path)
        goal = str(library.goal(safe_child(library.owner, report["origin_goal"])))
        if goal not in goals:
            continue
        require(digest(report) == path.stem, "Shared publication identity changed")
        require(report["kind"] == "shared_model_goal_publication_attempt", "Expected shared publication kind")
        batch = goals[goal]
        result[batch]["reports"][path.stem] = report
        for row in report["outcomes"]:
            if row["status"] != "published":
                continue
            task_id = row["task_id"]
            require(
                task_id not in expected[batch] or expected[batch][task_id] == row["entry_id"],
                "Ambiguous shared publication for one task",
            )
            expected[batch][task_id] = row["entry_id"]
    for path in sorted((library.root / "index").glob("*.json")):
        entry_id = read(path)["entry_id"]
        value = library.inspect_publication(entry_id)
        require(value["key_id"] == path.stem, "Library index identity changed")
        batch = goals.get(value["origin_goal"])
        if batch is None:
            continue
        task_id = value["origin"]["task_id"]
        tasks = {task["id"]: task for task in specifications[batch]["tasks"]}
        require(task_id in tasks and task_id not in result[batch]["models"], "Foreign or duplicate shared task")
        task = tasks[task_id]
        require(
            value["key"]["scope"] == task["scope"] and value["key"]["plugin_id"] == task["plugin_id"],
            "Published model differs from allocated scope",
        )
        require(
            value["origin"]["usage_attribution"] == "shared_goal_total_count_once", "Shared usage attribution required"
        )
        require(value["origin"]["future_labels_scored"] is False, "Model selection already used final labels")
        result[batch]["models"][task_id] = publication_binding(value)
    for batch, record in result.items():
        require(
            {task: model["entry_id"] for task, model in record["models"].items()} == expected[batch],
            "Published shared inventory is incomplete",
        )
        for report in record["reports"].values():
            for outcome in report["outcomes"]:
                if outcome["status"] != "published":
                    continue
                bound = record["models"][outcome["task_id"]]
                origin = bound["origin"]
                require(
                    origin["model_id"] == outcome["model_id"]
                    and bound["key_id"] == outcome["key_id"]
                    and origin["goal_sha256"] == report["goal_sha256"]
                    and origin["shared_execution_id"] == report["shared_execution_id"],
                    "Shared report differs from canonical origin",
                )
    return result


def snapshot(store, library):
    """Read acquisition metadata and hash asset bytes, without deserializing data."""
    state = store.read()
    require(state["owner"] is None and state["inflight"] is None, "Campaign execution is unresolved")
    specifications = store.specifications(state)
    require(bool(specifications), "No allocated shared batches")
    require(
        all(row["status"] in TERMINAL and row["attempts"] for row in state["batches"].values()),
        "Every shared batch must have a terminal attempt",
    )
    registered = publications(library, store, specifications)
    tasks, assets, goals = {}, {}, {}
    for batch, spec in specifications.items():
        root = safe_child(store.root, "goals/" + batch)
        goal = read(root / "goal.json") if (root / "goal.json").exists() else None
        if goal is None:
            require(state["batches"][batch]["status"] != "completed", "Completed batch lacks goal evidence")
        else:
            require(goal["status"] in TERMINAL and goal.get("inflight") is None, "Shared goal is not terminal")
        goals[batch] = {"status": goal["status"] if goal else None, "files": tree(root)}
        for task in spec["tasks"]:
            require(task["id"] not in tasks, "Duplicate allocated task identity")
            require(
                task["plugin_id"] in SUPPORTED_SHARED_PLUGINS,
                "Shared evaluation requires context-response or sequence-regression tasks",
            )
            if goal is not None:
                require(
                    read(safe_child(root, "targets/" + task["id"] + "/model_task.json")) == task,
                    "Shared target metadata changed",
                )
            require(set(task["data"]) == {"training", "validation", "future"}, "Three fixed partitions required")
            tasks[task["id"]] = {"batch": batch, "task_sha256": digest(task)}
            for ref in [*task["data"].values(), *task["references"]]:
                path = model_tasks.bound_file(ref)
                assets[str(path)] = ref["sha256"]
    return {
        "campaign_state_sha256": digest(state),
        "original_campaign_status": state["status"],
        "specifications": specifications,
        "tasks": tasks,
        "assets": assets,
        "goals": goals,
        "publications": registered,
    }


def admit(campaign, library, output):
    store, registry = SharedCampaignStore(campaign), ScientistModelLibrary(library)
    output = outside(output, store.root, registry.root)
    with store.lease():
        frozen = snapshot(store, registry)
        definition = {
            "schema": SCHEMA,
            "campaign": str(store.root),
            "library": str(registry.root),
            "snapshot": frozen,
            "evaluator_sources": source_identity(),
            "versions": versions(),
            "plugin_sources": {
                task["plugin_id"]: get_model_plugin(task["plugin_id"]).source_sha256
                for batch in frozen["specifications"].values()
                for task in batch["tasks"]
            },
            "selection": "All current indexed publications from every allocated terminal batch; no score-based selection",
            "controls": "Fixed training-only controls on every allocation; context and applicable positional controls use independent refits",
            "cost_attribution": "Once per original batch; per-task rows reference batch_id",
            "interpretation": "Post-acquisition descriptive assessment; no causal RSI effect or new acquisition",
            "final_labels_decoded": False,
        }
        require(snapshot(store, registry) == frozen, "Acquisition changed during admission")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "study.json", {"definition": definition, "sha256": digest(definition)})
    return definition


def load(study):
    saved = read(Path(study) / "study.json")
    definition = saved["definition"]
    require(digest(definition) == saved["sha256"] and definition["schema"] == SCHEMA, "Shared study identity changed")
    require(definition["evaluator_sources"] == source_identity(), "Evaluator implementation changed")
    require(definition["versions"] == versions(), "Numerical environment changed")
    for name, expected in definition["plugin_sources"].items():
        require(get_model_plugin(name).source_sha256 == expected, "Model implementation changed")
    return definition


def verify(definition, store, library):
    require(snapshot(store, library) == definition["snapshot"], "Frozen acquisition or publication inventory changed")
