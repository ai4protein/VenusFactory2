"""Retain coverage and validation/resource tradeoffs without new fitting."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from agent.harness.contracts import digest, file_hash

from .model_curation import current, effective_archived
from .model_scope_inventory import canonical_scope
from .numerical_evidence import verify_assessment
from .storage import safe_child

SCHEMA = "model_retention_plan.v1"
DIMENSIONS = ("scope_id", "protein", "property", "scenario", "function", "interface")


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"schema", "metric_directions", "archive_validation_dominated"}:
        raise ValueError("Invalid model retention policy")
    if policy["schema"] != "model_retention_policy.v1" or type(policy["archive_validation_dominated"]) is not bool:
        raise ValueError("Invalid model retention policy")
    directions = policy["metric_directions"]
    if not isinstance(directions, dict) or not directions or any(v not in {"min", "max"} for v in directions.values()):
        raise ValueError("Declare metric directions")


def _objectives(metrics, directions, prefix=""):
    result = {}
    for name, value in metrics.items():
        if isinstance(value, dict):
            result.update(_objectives(value, directions, prefix + name + "."))
        elif name in directions:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Missing finite comparison metric")
            result[prefix + name] = value if directions[name] == "min" else -value
    return result


def _comparison(library, entry_id, policy):
    publication = library.inspect_publication(entry_id)
    packet, origin = publication["packet"], publication["origin"]
    report, model = packet.get("assessment", {}), packet["model"]
    from .model_plugins import get_model_plugin

    if report.get("plugin_sha256") != get_model_plugin(publication["key"]["plugin_id"]).source_sha256:
        return {"status": "runtime_implementation_requires_review"}
    if report.get("passed") is not True or report.get("training_replayed") is not True:
        return {"status": "insufficient_evidence"}
    # Runtime-migrated archives need their own comparison adapter; retain them.
    if "runtime_migration" in origin:
        return {"status": "runtime_migration_requires_review"}
    folder = Path(publication["origin_goal"])
    model_id = packet["model_id"]
    matches = sorted(folder.glob(f"episodes/**/assessment_{model_id}.json"))
    for item in matches:
        item = safe_child(folder, str(item.relative_to(folder)))
        if json.loads(item.read_text()) != report:
            continue
        controller = item.parent
        assessment_policy = json.loads(safe_child(controller, "policy.json").read_text())
        if digest(assessment_policy) != report["policy_sha256"]:
            raise ValueError("Assessment policy changed")
        verified = verify_assessment(controller, model_id, model, assessment_policy, report)
        if verified["status"] != "verified":
            continue
        objectives = _objectives(report["metrics"], policy["metric_directions"])
        if not objectives:
            return {"status": "unsupported_metrics"}
        key = publication["key"]
        if (
            assessment_policy["scope"] != key["scope"]
            or assessment_policy["training_sha256"] != key["training_sha256"]
            or assessment_policy["plugin_id"] != key["plugin_id"]
        ):
            raise ValueError("Comparison scope changed")
        cohort = {
            k: assessment_policy[k]
            for k in ("scope", "training_sha256", "validation_sha256", "plugin_id", "plugin_sha256")
        }
        # No runtime/memory claim is made from serialized payload size.
        size = len(json.dumps(model.get("model_state"), sort_keys=True, separators=(",", ":")).encode())
        size += len(model.get("recipe", {}).get("source_code", "").encode())
        return {
            "status": "comparable",
            "cohort": digest(cohort),
            "objectives": objectives,
            "payload_bytes": size,
            "predictions_sha256": verified["evidence"]["predictions_sha256"],
            "assessment_sha256": digest(report),
            "evidence_sha256": digest(verified["evidence"]),
            "policy_sha256": digest(assessment_policy),
            "goal_sha256": origin["goal_sha256"],
            "training_sha256": key["training_sha256"],
            "validation_sha256": assessment_policy["validation_sha256"],
            "assessment_path": str(item.relative_to(library.owner)),
            "assessment_file_sha256": file_hash(item),
        }
    return {"status": "comparison_evidence_unavailable"}


def collect_records(library, policy):
    validate_policy(policy)
    catalog = library.catalog(verify=False)
    archived = effective_archived(library)
    rows, groups = [], defaultdict(list)
    for row in catalog["entries"]:
        key = row["key"]
        scope = key["scope"]
        entry = library._object("entries", row["entry_id"])
        origin = library._object("origins/" + digest(key), entry["canonical_origin"])
        record = {
            "entry_id": row["entry_id"],
            "scope_id": canonical_scope(key),
            "protein": scope["protein"],
            "property": scope["target"],
            "scenario": digest(scope.get("context", {"assay": scope["assay"]})),
            "function": origin.get("stage", "unspecified"),
            "interface": key["plugin_id"],
            "entry_sha256": digest(entry),
            "comparison": {"status": "unique_scope"},
        }
        rows.append(record)
        if row["entry_id"] not in archived:
            # The full input contract and training partition must also agree.
            groups[
                digest(
                    {
                        "key_scope": scope,
                        "training": key["training_sha256"],
                        "interface": key["plugin_id"],
                        "function": record["function"],
                    }
                )
            ].append(record)
    for group in groups.values():
        if len(group) > 1:
            for row in group:
                row["comparison"] = _comparison(library, row["entry_id"], policy)
    if library.catalog(verify=False) != catalog:
        raise ValueError("Model catalog changed during retention audit")
    return rows, digest(catalog)


def _prefer(a, b, policy):
    x, y = a["comparison"], b["comparison"]
    if (
        x["status"] != "comparable"
        or y["status"] != "comparable"
        or x["cohort"] != y["cohort"]
        or any(a[k] != b[k] for k in DIMENSIONS)
        or set(x["objectives"]) != set(y["objectives"])
    ):
        return None
    if x["predictions_sha256"] == y["predictions_sha256"] and x["objectives"] == y["objectives"]:
        if (x["payload_bytes"], a["entry_id"]) < (y["payload_bytes"], b["entry_id"]):
            return "identical_validation_predictions"
    if (
        policy["archive_validation_dominated"]
        and x["payload_bytes"] <= y["payload_bytes"]
        and all(x["objectives"][k] <= y["objectives"][k] for k in x["objectives"])
        and any(x["objectives"][k] < y["objectives"][k] for k in x["objectives"])
    ):
        return "validation_and_payload_dominated"
    return None


def select_retention(records, policy):
    validate_policy(policy)
    by_id = {r["entry_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Repeated model identity")
    groups = defaultdict(list)
    for row in records:
        groups[row["scope_id"]].append(row)
    archive = []
    for group in groups.values():
        keepers = [r for r in group if not any(_prefer(other, r, policy) for other in group if other != r)]
        for row in group:
            if row in keepers:
                continue
            alternatives = [r for r in keepers if _prefer(r, row, policy)]
            if not alternatives:
                raise ValueError("No retained model covers archived entry")
            replacement = min(alternatives, key=lambda r: (r["comparison"]["payload_bytes"], r["entry_id"]))
            archive.append(
                {
                    "entry_id": row["entry_id"],
                    "replacement": replacement["entry_id"],
                    "reason": _prefer(replacement, row, policy),
                }
            )
    removed = {r["entry_id"] for r in archive}
    retained = [r for r in records if r["entry_id"] not in removed]
    coverage = {}
    for axis in DIMENSIONS:
        before, after = {r[axis] for r in records}, {r[axis] for r in retained}
        if before != after:
            raise ValueError("Retention would lose scientific coverage")
        coverage[axis] = {"before": len(before), "after": len(after), "lost": 0}
    return {
        "archive": sorted(archive, key=lambda x: x["entry_id"]),
        "active_ids": sorted(r["entry_id"] for r in retained),
        "coverage": coverage,
    }


def plan_retention(library, policy):
    state = current(library)
    records, catalog_hash = collect_records(library, policy)
    effective = effective_archived(library)
    active = [r for r in records if r["entry_id"] not in effective]
    selection = select_retention(active, policy)
    if current(library) != state:
        raise ValueError("Model curation changed during planning")
    return {
        "schema": SCHEMA,
        "policy": policy,
        "catalog_sha256": catalog_hash,
        "base_revision": state["revision"],
        "registered_entries": len(records),
        "previously_archived": len(records) - len(active),
        "records": active,
        **selection,
        "basis": "Recorded, hash-verified validation predictions and serialized payload bytes; no fresh performance claim.",
        "immutable_artifacts_removed": 0,
        "external_model_requests": 0,
    }
