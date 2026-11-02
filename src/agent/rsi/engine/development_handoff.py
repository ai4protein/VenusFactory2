"""Versioned executor-to-host handoff; submission never confers acceptance."""

from __future__ import annotations

FIELD = "development_handoff"
POLICY = "host_review.v1"


def policy_for(task):
    policy = task.get(FIELD, "none")
    if policy == "none":
        return policy
    if policy != POLICY or task.get("phase") != "develop":
        raise ValueError("Invalid development handoff policy or phase")
    return policy


def description(task):
    if policy_for(task) == "none":
        return None
    return {
        "schema": POLICY,
        "terminal_tool": "submit_model_code",
        "after_success": "return_final_summary",
        "next_actor": "host_review",
        "instruction": (
            "After submit_model_code succeeds, finish this execution with a concise evidence summary. "
            "Do not restart inspection, discovery, evaluation or submission. The candidate is awaiting "
            "independent host validation; this executor cannot register it. Submission alone does not "
            "establish predictive quality, scientific acceptance or successful reuse."
        ),
    }


def receipt_fields(task, source_sha256):
    if policy_for(task) == "none":
        return {}
    return {
        FIELD: {
            "schema": POLICY,
            "state": "awaiting_host_validation",
            "candidate_source_sha256": source_sha256,
            "next_action": "return_final_summary",
            "scientific_acceptance": "pending",
            "instruction": description(task)["instruction"],
        }
    }
