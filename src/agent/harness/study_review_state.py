"""Versioned facts and bounded factual checks for interim SC reviews.

This validates explicit evidence-absence requests, not the truth of arbitrary
natural-language opinions. Scientific rejection remains the reviewer's decision.
"""
from __future__ import annotations

import copy

POLICY = "study-review-evidence.v1"
MISSING = (
    "data_partitions", "validation_measurement", "measured_baseline",
    "measured_revision", "model_submission",
)
REQUIREMENTS_BY_KIND = {
    "missing_evidence": MISSING,
    "execution_failure": ("execution_failure",),
    "scientific_concern": ("scientific_value",),
}
PAIRING_GUIDANCE = "Valid revision pairs: " + "; ".join(
    kind + " -> " + ", ".join(requirements)
    for kind, requirements in REQUIREMENTS_BY_KIND.items()
) + "."
LABELS = {
    "data_partitions": "Record data partitions",
    "validation_measurement": "Complete validation measurement",
    "measured_baseline": "Measure a baseline",
    "measured_revision": "Measure a justified revision",
    "model_submission": "Submit a scoped model candidate",
    "execution_failure": "Resolve the recorded execution failure",
    "scientific_value": "Address the scientific concern",
}


class StudyReviewInconsistent(ValueError):
    """A structured SC assertion contradicts the host-owned study state."""


def request_pair_diagnostic(request):
    """Describe an invalid field pairing without rewriting the review or its verdict."""
    if not isinstance(request, dict):
        return None
    kind, requirement = request.get("kind"), request.get("requirement")
    if not isinstance(kind, str) or kind not in REQUIREMENTS_BY_KIND:
        return None  # Other malformed fields retain ordinary schema diagnostics.
    allowed = REQUIREMENTS_BY_KIND[kind]
    if not isinstance(requirement, str) or requirement in allowed:
        return None
    message = ("Invalid revision pair: kind=" + repr(kind) + ", requirement=" + repr(requirement[:80]) +
               ". For this kind, requirement must be: " + ", ".join(allowed) + ".")
    if requirement in {"measured_baseline", "measured_revision", "model_submission"}:
        message += " Missing evidence uses kind='missing_evidence'; this requirement is study-wide, with experiment_id empty."
    return message


def request_schema():
    from .scientist_role_output import object_schema

    text = {"type": "string", "minLength": 1, "maxLength": 2000}
    item = object_schema({
        "kind": {"enum": list(REQUIREMENTS_BY_KIND)},
        "requirement": {"enum": [*MISSING, "execution_failure", "scientific_value"]},
        "experiment_id": {
            "type": "string", "pattern": "^([a-f0-9]{32})?$",
            "description": ("Known measured experiment ID, or empty for a study-wide request. "
                            "Must be empty for measured_baseline, measured_revision and model_submission."),
        },
        "reason": text,
        "proposed_action": text,
    })
    item["description"] = PAIRING_GUIDANCE
    item["anyOf"] = [
        object_schema({**item["properties"], "kind": {"enum": [kind]},
                       "requirement": {"enum": list(requirements)}})
        for kind, requirements in REQUIREMENTS_BY_KIND.items()
    ]
    return {
        "type": "array", "maxItems": 12,
        "description": (
            "Actionable study revisions. missing_evidence asserts a specific absent "
            "host fact; execution_failure refers to current failures; scientific_concern "
            "may question utility, generalization or split quality despite completed measurements."
        ),
        "items": item,
    }


def measurement_state(context):
    """Project already receipt-checked history; do not copy authored state."""
    history = context["development_measurements"]
    rows = []
    for row in history["experiments"]:
        counts = row["split_counts"]
        rows.append({
            "experiment_id": row["id"],
            "variant": row["variant"],
            "validation_completed": True,
            "partitions_recorded": all(type(counts.get(key)) is int and counts[key] > 0
                                       for key in ("train", "validation", "test")),
            "split_counts": copy.deepcopy(counts),
            "validation": copy.deepcopy(row["validation"]),
            "receipt_sha256": row["receipt_sha256"],
        })
    return {
        "schema": POLICY,
        "experiments": rows,
        "complete_history": not history["omitted_count"] and not history["receipt_errors"],
        "submitted_model_ids": [row["candidate_id"] for row in context["scientific_acceptance"]["models"]],
        "current_execution_failures": len(context.get("execution", {}).get("failures", [])),
        "interpretation": (
            "Completed measurements are historical host facts, not scientific acceptance. "
            "Recorded partitions do not by themselves establish scientifically adequate splits. "
            "Final test labels and scores are not disclosed."
        ),
    }


def validate(review, context):
    """Check only typed, mechanically decidable requests; never score usefulness."""
    policy = context["scientific_acceptance"].get("study_review_policy")
    if policy is None:
        return  # Legacy receipts retain their original contract.
    if policy != POLICY:
        raise StudyReviewInconsistent("Unsupported study review policy: " + str(policy))
    expected = measurement_state(context)
    if context.get("measurement_state") != expected:
        raise StudyReviewInconsistent("Host measurement state does not match the verified evidence packet.")
    requests = review.get("revision_requests")
    if not isinstance(requests, list):
        raise StudyReviewInconsistent("Return structured revision_requests for this study review.")
    if review["decision"] == "needs_revision" and not requests:
        raise StudyReviewInconsistent("A study rejection requires a structured revision request.")
    if review["decision"] == "ready_for_host_validation" and (requests or review.get("issues")):
        raise StudyReviewInconsistent("Ready cannot retain unresolved study revision requests or issues.")
    rows = {row["experiment_id"]: row for row in expected["experiments"]}
    for request in requests:
        kind, requirement = request.get("kind"), request.get("requirement")
        identity = request.get("experiment_id")
        if not isinstance(identity, str) or (identity and identity not in rows):
            raise StudyReviewInconsistent("Revision requests must reference a measured experiment from this packet.")
        if any(not isinstance(request.get(key), str) or not request[key].strip()
               for key in ("reason", "proposed_action")):
            raise StudyReviewInconsistent("Each revision needs a substantive reason and proposed action.")
        diagnostic = request_pair_diagnostic(request)
        if diagnostic:
            raise StudyReviewInconsistent(diagnostic)
        if kind == "scientific_concern" and requirement == "scientific_value":
            continue
        if kind == "execution_failure" and requirement == "execution_failure":
            if expected["current_execution_failures"]:
                continue
            raise StudyReviewInconsistent("No unresolved execution failure is recorded in this evidence packet.")
        if kind != "missing_evidence" or requirement not in MISSING:
            raise StudyReviewInconsistent("Use a requirement matching the revision kind.")
        if requirement in {"measured_baseline", "measured_revision", "model_submission"} and identity:
            raise StudyReviewInconsistent(requirement + " is study-wide; leave experiment_id empty.")
        selected = [rows[identity]] if identity else list(rows.values())
        present = {
            "data_partitions": any(row["partitions_recorded"] for row in selected),
            "validation_measurement": any(row["validation_completed"] for row in selected),
            "measured_baseline": any(row["variant"] == "baseline" for row in selected),
            "measured_revision": any(row["variant"] in {"candidate", "ablation"} for row in selected),
            "model_submission": bool(expected["submitted_model_ids"]),
        }[requirement]
        if present:
            raise StudyReviewInconsistent(
                requirement + " is already recorded by the host" +
                (" for experiment " + identity if identity else "") +
                ". Assess its scientific adequacy as scientific_concern; do not assert missing evidence."
            )
        if not identity and requirement != "model_submission" and not expected["complete_history"]:
            raise StudyReviewInconsistent(
                "The bounded history cannot establish study-wide absence of " + requirement + "."
            )


def feedback(review, context):
    """Keep facts and typed actions distinct from unverified reviewer prose."""
    validate(review, context)
    requests = copy.deepcopy(review["revision_requests"])
    return {
        "summary": "SC requests study revision; completed host measurements remain available.",
        "issues": [LABELS[row["requirement"]] for row in requests],
        "study_revision_requests": requests,
        "measurement_state": copy.deepcopy(context["measurement_state"]),
        "reviewer_commentary": {"summary": review["summary"], "issues": copy.deepcopy(review["issues"])},
    }
