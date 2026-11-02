"""Narrow, opt-in recovery for an empirically intermittent upstream rejection.

No request body, model binding, scientific result or acceptance gate is changed.
The failed physical request remains charged and separately visible in the audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import digest

POLICY = "dmx_maas_once_v1"
REJECTION_SHA256 = "d16974103aecda7f698bc1b3e4fdcbadb26214952c3beb47b2ac5784d891a1ff"


def matches_rejection(record):
    return (
        record.get("status") == 400
        and record.get("body_prefix_sha256") == REJECTION_SHA256
        and record.get("truncated") is False
        and record.get("inspected_bytes") == 229
    )


def audit_recovery(path, records, requests, responses):
    lockfile = Path(path).parent / "profile.lock.json"
    unbound_records = any(
        "recovery_policy" in row or row.get("event") in {"retry_scheduled", "response_shape"} for row in records
    )
    unbound = {
        "request_recovery": "unbound",
        "recovery_valid": False,
        "recovery_problems": ["unbound_recovery_records"],
        "logical_calls": 0,
        "recovered_calls": [],
        "failed_attempts": 0,
    }
    if not lockfile.exists():
        return unbound if unbound_records else None
    lock = json.loads(lockfile.read_text())
    profile = lock.get("profile", {})
    policy = profile.get("request_recovery", "none")
    if policy == "none":
        return unbound if unbound_records else None
    from .empty_response_recovery import POLICY as EMPTY_POLICY
    from .empty_response_recovery import audit_recovery as audit_empty

    if policy == EMPTY_POLICY:
        return audit_empty(path, records, requests, responses)
    result = {
        "request_recovery": policy,
        "recovery_valid": False,
        "recovery_problems": [],
        "logical_calls": 0,
        "recovered_calls": [],
        "failed_attempts": sum(r.get("status") != 200 or r.get("completed") is not True for r in responses),
    }
    problems = result["recovery_problems"]
    if policy != POLICY or digest({k: v for k, v in lock.items() if k != "sha256"}) != lock.get("sha256"):
        problems.append("unbound_recovery_policy")
        return result
    model = profile.get("model", {})
    if model.get("provider") != "dmx" or urlsplit(model.get("base_url", "")).hostname != "www.dmxapi.cn":
        problems.append("unadmitted_recovery_endpoint")
        return result
    if len(requests) > profile["budget"]["model_calls"]:
        problems.append("physical_request_budget_exceeded")
    groups = {}
    replies = {r["call"]: r for r in responses}
    rejections = [r for r in records if r["event"] == "upstream_rejection"]
    scheduled = [r for r in records if r["event"] == "retry_scheduled"]
    for req in requests:
        if req.get("recovery_policy") != POLICY or type(req.get("logical_call")) is not int:
            problems.append("request_missing_bound_recovery_metadata")
            continue
        groups.setdefault(req["logical_call"], []).append(req)
    result["logical_calls"] = len(groups)
    expected_schedules = []
    for logical, attempts in groups.items():
        group_problem_count = len(problems)
        first = attempts[0]
        if first["call"] != logical or first.get("attempt") != 1 or first.get("retry_of") is not None:
            problems.append("invalid_first_attempt")
            continue
        first_response = replies.get(first["call"], {})
        if len(attempts) == 1:
            if first_response.get("status") != 200 or first_response.get("completed") is not True:
                problems.append("unrecovered_failed_request")
            continue
        if len(attempts) != 2:
            problems.append("more_than_one_retry")
            continue
        second = attempts[1]
        response = replies.get(second["call"], {})
        rejected = [r for r in rejections if r["call"] == first["call"]]
        schedules = [r for r in scheduled if r["failed_call"] == first["call"]]
        expected_schedules.append(first["call"])
        if second.get("attempt") != 2 or second.get("retry_of") != first["call"]:
            problems.append("invalid_retry_ancestry")
        if not first.get("request_sha256") or first.get("request_sha256") != second.get("request_sha256"):
            problems.append("retry_payload_changed")
        if len(rejected) != 1 or not matches_rejection(rejected[0]):
            problems.append("unrecognized_rejection_retried")
        if (
            len(schedules) != 1
            or schedules[0].get("logical_call") != logical
            or schedules[0].get("delay_seconds") != 0.5
        ):
            problems.append("retry_schedule_missing_or_changed")
        elif first_response not in records or not (
            records.index(first_response) < records.index(schedules[0]) < records.index(second)
        ):
            problems.append("retry_preceded_failed_response")
        if (
            first_response.get("status") != 400
            or first_response.get("completed") is not False
            or first_response.get("error_type") is not None
        ):
            problems.append("failed_attempt_relabelled")
        if response.get("status") != 200 or response.get("completed") is not True:
            problems.append("retry_failed")
        if len(problems) == group_problem_count:
            result["recovered_calls"].append(
                {"logical_call": logical, "failed_attempt": first["call"], "successful_attempt": second["call"]}
            )
    if sorted(expected_schedules) != sorted(r["failed_call"] for r in scheduled):
        problems.append("unmatched_retry_schedule")
    result["recovery_valid"] = bool(requests) and not problems
    return result
