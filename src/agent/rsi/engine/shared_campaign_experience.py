"""Select a bounded history from completed earlier batches of one campaign."""

from __future__ import annotations

import copy
import json

from agent.harness.contracts import digest

from agent.rsi import context_interfaces, context_transfer, shared_library_experience
from .scientist_model_library import ScientistModelLibrary
from .storage import atomic_json, safe_child

FIELD = "program_memory"
POLICY = "prior_shared_batch_programs.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def policy(value, tasks):
    base = {"policy", "condition", "max_candidates", "keep", "ranking"}
    require(
        isinstance(value, dict) and base <= set(value) <= base | {"overlap_policy", "adaptation_policy"},
        "Invalid shared campaign program-memory fields",
    )
    require(
        value["policy"] == POLICY and value["condition"] in {"none", "summary", "source"},
        "Invalid program-memory policy",
    )
    require(
        type(value["max_candidates"]) is int
        and 1 <= value["max_candidates"] <= 5
        and type(value["keep"]) is int
        and 1 <= value["keep"] <= value["max_candidates"],
        "Invalid bounded program-memory capacity",
    )
    require(value["ranking"] in shared_library_experience.RANKINGS, "Invalid program-memory ranking")
    if value["ranking"] == "training_holdout_mean_skill":
        require(
            all(task["plugin_id"] in context_interfaces.PLUGINS for task in tasks),
            "Training screening requires contextual tasks",
        )
    for task in tasks:
        context_transfer.validate_declaration(value, plugin_id=task["plugin_id"])
        if task["plugin_id"] in context_interfaces.PLUGINS:
            require(
                value.get("overlap_policy") == context_transfer.POLICY,
                "Contextual program memory requires study-disjoint admission",
            )
    return copy.deepcopy(value)


def prior_pool(store, batch):
    """Newest distinct source programs; only immutable prior-batch publications."""
    state = store.read()
    specs = store.specifications(state)
    require(batch["id"] in specs and digest(specs[batch["id"]]) == digest(batch), "Program-memory batch changed")
    declaration = policy(batch[FIELD], batch["tasks"])
    library = ScientistModelLibrary(state["definition"]["library_root"])
    publications, inventory = [], []
    for name in specs:
        if name == batch["id"]:
            break
        item = state["batches"][name]
        require(
            item["status"] in {"completed", "budget_exhausted", "failed", "interrupted", "cancelled"},
            "Earlier batch is not terminal",
        )
        for attempt in item["attempts"]:
            report = attempt.get("publication")
            if report is None:
                continue
            stored = library._object("publication_attempts", report["report_id"])
            require(
                stored == {key: value for key, value in report.items() if key != "report_id"},
                "Prior publication changed",
            )
            require(
                stored["kind"] == "shared_model_goal_publication_attempt"
                and stored["origin_goal"] == str(store.goal(name).relative_to(library.owner))
                and stored["all_published"],
                "Program memory contains a foreign or incomplete publication",
            )
            publications.append({"batch": name, "report_id": report["report_id"]})
            for outcome in stored["outcomes"]:
                require(outcome["status"] == "published", "Unpublished program in prior pool")
                entry = library._object("entries", outcome["entry_id"])
                require(digest(entry["key"]) == outcome["key_id"], "Prior entry key changed")
                packet = library._object("packets", entry["packet_id"])
                inventory.append(
                    {
                        "entry_id": outcome["entry_id"],
                        "source_sha256": packet["model"]["source_sha256"],
                        "batch": name,
                        "publication_report_id": report["report_id"],
                    }
                )
    selected, seen = [], set()
    for row in reversed(inventory):
        if row["source_sha256"] in seen:
            continue
        seen.add(row["source_sha256"])
        if len(selected) < declaration["max_candidates"]:
            selected.append(row)
    return {
        "policy": declaration,
        "prefix_publications": publications,
        "prior_entry_records": len(inventory),
        "prior_distinct_sources": len(seen),
        "prior_inventory_sha256": digest(inventory),
        "candidates": selected,
        "selection_rule": "Newest distinct programs from earlier terminal batches, bounded before recipient training screening",
    }


def selection_path(store, batch):
    return safe_child(store.root, "source_selections/" + batch["id"] + ".json")


def _request(store, pool):
    declaration = pool["policy"]
    return {
        "condition": declaration["condition"],
        "library_root": store.read()["definition"]["library_root"],
        "entry_ids": [row["entry_id"] for row in pool["candidates"]],
        **{key: declaration[key] for key in ["overlap_policy", "adaptation_policy"] if key in declaration},
    }


def select_for_batch(store, batch):
    path = selection_path(store, batch)
    if path.exists():
        record = verify_selection(store, batch)
        return record["screen"]["selected"]
    pool = prior_pool(store, batch)
    report, _ = shared_library_experience.select_sources(
        _request(store, pool), batch["tasks"], keep=pool["policy"]["keep"], ranking=pool["policy"]["ranking"]
    )
    record = {"schema": POLICY, "batch": batch["id"], "pool": pool, "screen": report}
    atomic_json(path, record)
    path.chmod(0o444)
    return report["selected"]


def verify_selection(store, batch, *, goal=None, replay=False):
    path = selection_path(store, batch)
    require(path.is_file() and not path.is_symlink(), "Missing shared source selection")
    record = json.loads(path.read_text())
    require(
        set(record) == {"schema", "batch", "pool", "screen"}
        and record["schema"] == POLICY
        and record["batch"] == batch["id"],
        "Shared source selection identity changed",
    )
    pool = prior_pool(store, batch)
    require(record["pool"] == pool, "Shared source pool changed or includes later batches")
    request = _request(store, pool)
    expected = {"selection": request, "keep": pool["policy"]["keep"], "ranking": pool["policy"]["ranking"]}
    require(record["screen"]["request"] == expected, "Shared source screening policy changed")
    chosen = record["screen"]["selected"]
    require(len(chosen["entry_ids"]) <= pool["policy"]["keep"], "Too many selected historical programs")
    require(set(chosen["entry_ids"]) <= set(request["entry_ids"]), "Selected source was outside prior batches")
    require(
        {key: value for key, value in chosen.items() if key != "entry_ids"}
        == {key: value for key, value in request.items() if key != "entry_ids"},
        "Selected source transfer contract changed",
    )
    if replay:
        actual, _ = shared_library_experience.select_sources(
            request, batch["tasks"], keep=pool["policy"]["keep"], ranking=pool["policy"]["ranking"]
        )
        require(actual == record["screen"], "Shared source screening replay differs")
    if goal is not None:
        state = json.loads((goal / "goal.json").read_text())
        protocol = json.loads((goal / "protocol.json").read_text())
        frozen = shared_library_experience.read_frozen(goal, state, protocol)
        require(
            frozen is not None
            and frozen["manifest"]["selection"] == chosen
            and digest(frozen) == record["screen"]["selected_snapshot_sha256"],
            "Shared child historical source differs from its frozen campaign selection",
        )
    return record
