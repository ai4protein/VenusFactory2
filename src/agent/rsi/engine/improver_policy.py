"""Versioned prompt policies for the model-code improver, separate from models.

Policies are data consumed by a bounded harness. They never change evaluation,
resource limits, tools or access. A revised policy is an untested candidate;
its changed hash is not evidence of improved offspring or recursive efficacy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

PLUGIN_ID = "model_improver.prompt.v1"
METRICS = {"rows", "rmse", "training_mean_rmse", "mse_skill_vs_training_mean", "spearman"}
BASE_POLICY = {
    "schema": 1,
    "plugin_id": PLUGIN_ID,
    "proposal_instruction": "Inspect the supplied model ABI and permitted training data. Use verified development experience to identify a specific modelling weakness. Propose executable deterministic fit/predict code that addresses that weakness, test it with the supplied training-only tool, and select one valid source using measured feedback. Distinguish model acceptance from unseen-task performance. Preserve compatibility and do not infer algorithmic novelty from a new source hash.",
    "revision_instruction": "Review the supplied development history and the current proposal and revision instructions. Identify recurring weaknesses in how model changes were proposed, tested or selected. Propose one revised pair of instructions that could improve subsequent model-code proposals, including how later revisions should use feedback. Give a falsifiable rationale. Keep both instructions transferable across proteins. Do not add tools, change budgets, modify the evaluator or claim access to final labels. A revision remains untested until matched offspring experiments evaluate it.",
}


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def validate_policy(value):
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "plugin_id",
        "proposal_instruction",
        "revision_instruction",
    }:
        raise ValueError("Improver policy has unexpected fields")
    if type(value["schema"]) is not int or value["schema"] != 1 or value["plugin_id"] != PLUGIN_ID:
        raise ValueError("Unknown improver policy plugin")
    for name in ["proposal_instruction", "revision_instruction"]:
        if not isinstance(value[name], str) or not 40 <= len(value[name].encode()) <= 8000:
            raise ValueError("Improver instruction must contain 40..8000 UTF-8 bytes")
    return copy.deepcopy(value)


def validate_history(entries):
    if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
        raise ValueError("Supply 1..12 bound development episodes")
    sources = 0
    for row in entries:
        required = {"case", "cycle", "status", "goal_sha256", "protocol_sha256"}
        optional = {
            "model_id",
            "source_sha256",
            "source_code",
            "development_acceptance_metrics",
            "acceptance_passed",
            "error_type",
        }
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - optional:
            raise ValueError("Unexpected development history fields")
        if type(row["cycle"]) is not int or row["cycle"] < 0 or not isinstance(row["case"], str) or not row["case"]:
            raise ValueError("Invalid history episode identity")
        if row["status"] not in {"completed", "candidate_rejected", "failed"}:
            raise ValueError("Unfinished development episode")
        for key in ["goal_sha256", "protocol_sha256", "model_id", "source_sha256"]:
            if key in row and (
                not isinstance(row[key], str)
                or len(row[key]) != 64
                or any(c not in "0123456789abcdef" for c in row[key])
            ):
                raise ValueError("Malformed development evidence hash")
        if "development_acceptance_metrics" in row:
            metrics = row["development_acceptance_metrics"]
            if (
                not {"model_id", "source_sha256", "acceptance_passed"} <= set(row)
                or type(row["acceptance_passed"]) is not bool
                or not isinstance(metrics, dict)
                or set(metrics) != METRICS
            ):
                raise ValueError("Incomplete acceptance feedback")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite development feedback")
        if "source_code" in row:
            sources += 1
            source = row["source_code"]
            if (
                not isinstance(source, str)
                or len(source.encode()) > 60000
                or hashlib.sha256(source.encode()).hexdigest() != row.get("source_sha256")
            ):
                raise ValueError("Source projection does not match verified model")
    if sources > 2:
        raise ValueError("Only the last two source implementations may be exposed")
    identity(entries)
    return copy.deepcopy(entries)


def revision_packet(parent, history):
    parent, history = validate_policy(parent), validate_history(history)
    return {
        "schema": 1,
        "parent_id": identity(parent),
        "parent_policy": parent,
        "history_sha256": identity(history),
        "development_history": history,
        "editable_fields": ["proposal_instruction", "revision_instruction"],
        "authority": "Candidate prompt data only. Fixed host evaluation, tools, budgets and final-label isolation override generated instructions.",
        "status": "revision_context_not_performance_evidence",
    }


def validate_packet(packet):
    expected = revision_packet(packet["parent_policy"], packet["development_history"])
    if packet != expected:
        raise ValueError("Improver revision context binding changed")
    return copy.deepcopy(packet)


def candidate(policy, rationale, packet):
    packet = validate_packet(packet)
    policy = validate_policy(policy)
    if identity(policy) == packet["parent_id"]:
        raise ValueError("Unchanged policy is not a new candidate")
    if not isinstance(rationale, str) or not 20 <= len(rationale.encode()) <= 6000:
        raise ValueError("A bounded, substantive revision rationale is required")
    return {
        "schema": 1,
        "policy": policy,
        "policy_id": identity(policy),
        "parent_id": packet["parent_id"],
        "history_sha256": packet["history_sha256"],
        "rationale": rationale,
        "status": "untested_candidate",
        "promoted": False,
    }


def validate_candidate(value, packet):
    expected = candidate(value["policy"], value["rationale"], packet)
    if value != expected:
        raise ValueError("Candidate policy binding or untested status changed")
    return copy.deepcopy(value)
