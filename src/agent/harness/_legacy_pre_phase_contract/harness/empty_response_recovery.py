"""Opt-in, once-only replay of a complete but empty length-limited response.

This restores an opportunity to produce output; it is not a model improvement.
The prompt and request parameters remain byte-identical across attempts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contracts import digest

POLICY = "empty_length_once_v1"


def eligible(shape):
    return (
        shape.get("recognized") is True
        and shape.get("choice_count") == 1
        and shape.get("finish_reason") == "length"
        and shape.get("content_empty") is True
        and shape.get("tool_payload_present") is False
        and shape.get("refusal_present") is False
    )


def response_shape(raw, streaming):
    """Keep fixed shape signals and hashes, never content, reasoning or tool args."""
    result = {
        "recognized": False,
        "choice_count": 0,
        "finish_reason": "unknown",
        "content_empty": True,
        "tool_payload_present": False,
        "refusal_present": False,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "response_bytes": len(raw),
    }
    choices = {}
    try:
        if streaming:
            objects = []
            done = False
            for line in raw.splitlines():
                if not line.startswith(b"data:"):
                    continue
                value = line[5:].strip()
                if value == b"[DONE]":
                    done = True
                elif value:
                    objects.append(json.loads(value))
            if not done:
                return result
        else:
            objects = [json.loads(raw)]
        for obj in objects:
            if not isinstance(obj, dict) or obj.get("error") or not isinstance(obj.get("choices"), list):
                return result
            for position, choice in enumerate(obj["choices"]):
                if not isinstance(choice, dict):
                    return result
                index = choice.get("index", 0 if streaming else position)
                if type(index) is not int or index < 0:
                    return result
                state = choices.setdefault(index, {"reasons": [], "content": False, "tools": False, "refusal": False})
                reason = choice.get("finish_reason")
                if reason is not None:
                    state["reasons"].append(reason)
                message = choice.get("delta" if streaming else "message")
                if not isinstance(message, dict):
                    return result
                content = message.get("content")
                state["content"] |= content is not None and (not isinstance(content, str) or bool(content.strip()))
                state["tools"] |= bool(message.get("tool_calls") or message.get("function_call"))
                state["refusal"] |= bool(message.get("refusal"))
        result["choice_count"] = len(choices)
        if len(choices) != 1 or 0 not in choices:
            return result
        state = choices[0]
        if len(state["reasons"]) != 1:
            return result
        reason = state["reasons"][0]
        result.update(
            recognized=True,
            finish_reason=reason
            if reason in {"stop", "length", "tool_calls", "function_call", "content_filter"}
            else "unknown",
            content_empty=not state["content"],
            tool_payload_present=state["tools"],
            refusal_present=state["refusal"],
        )
    except (ValueError, TypeError, UnicodeError):
        pass
    return result


def audit_recovery(path, records, requests, responses):
    lock = json.loads((Path(path).parent / "profile.lock.json").read_text())
    profile = lock.get("profile", {})
    result = {
        "request_recovery": POLICY,
        "recovery_valid": False,
        "recovery_problems": [],
        "logical_calls": 0,
        "recovered_calls": [],
        "failed_attempts": 0,
    }
    problems = result["recovery_problems"]
    if profile.get("request_recovery") != POLICY or digest(
        {k: v for k, v in lock.items() if k != "sha256"}
    ) != lock.get("sha256"):
        problems.append("unbound_recovery_policy")
        return result
    if profile.get("model", {}).get("protocol") != "chat_completions":
        problems.append("unsupported_recovery_protocol")
    if len(requests) > profile["budget"]["model_calls"]:
        problems.append("physical_request_budget_exceeded")
    groups = {}
    replies = {r["call"]: r for r in responses}
    shapes = [r for r in records if r["event"] == "response_shape"]
    schedules = [r for r in records if r["event"] == "retry_scheduled"]
    expected_schedules = []
    for request in requests:
        if request.get("recovery_policy") != POLICY or type(request.get("logical_call")) is not int:
            problems.append("request_missing_bound_recovery_metadata")
            continue
        groups.setdefault(request["logical_call"], []).append(request)
    result["logical_calls"] = len(groups)
    evidence = {}
    for response in responses:
        matches = [s for s in shapes if s.get("call") == response["call"]]
        if response.get("status") != 200 or response.get("completed") is not True:
            result["failed_attempts"] += 1
            problems.append("failed_physical_request")
            continue
        if len(matches) != 1:
            problems.append("missing_or_duplicate_response_shape")
            continue
        shape = matches[0]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(shape.get("response_sha256", "")))
            or shape["response_sha256"] != response.get("response_sha256")
            or digest({k: v for k, v in shape.items() if k not in {"event", "call"}})
            != response.get("response_shape_sha256")
            or type(shape.get("response_bytes")) is not int
            or shape["response_bytes"] <= 0
            or records.index(shape) >= records.index(response)
        ):
            problems.append("response_shape_binding_changed")
            continue
        evidence[response["call"]] = shape
        result["failed_attempts"] += int(eligible(shape))
    if len(shapes) != len(evidence):
        problems.append("unmatched_response_shape")
    for logical, attempts in groups.items():
        prior_problems = len(problems)
        first = attempts[0]
        if first["call"] != logical or first.get("attempt") != 1 or first.get("retry_of") is not None:
            problems.append("invalid_first_attempt")
            continue
        first_shape = evidence.get(first["call"], {})
        if len(attempts) == 1:
            if eligible(first_shape):
                problems.append("unrecovered_empty_response")
            continue
        if len(attempts) != 2:
            problems.append("more_than_one_retry")
            continue
        second = attempts[1]
        expected_schedules.append(first["call"])
        scheduled = [s for s in schedules if s.get("failed_call") == first["call"]]
        if not eligible(first_shape):
            problems.append("nonempty_or_nonlength_response_retried")
        if second.get("attempt") != 2 or second.get("retry_of") != first["call"]:
            problems.append("invalid_retry_ancestry")
        if not first.get("request_sha256") or first.get("request_sha256") != second.get("request_sha256"):
            problems.append("retry_payload_changed")
        if any(
            first.get(k) != second.get(k)
            for k in [
                "model",
                "temperature",
                "max_tokens",
                "tools_sent",
                "tool_schema_sha256",
                "thinking_sent",
                "stream",
                "total_deadline_s",
            ]
        ):
            problems.append("retry_policy_changed")
        if (
            len(scheduled) != 1
            or scheduled[0].get("logical_call") != logical
            or scheduled[0].get("delay_seconds") != 0.5
        ):
            problems.append("retry_schedule_missing_or_changed")
        elif first["call"] not in replies or not (
            records.index(replies[first["call"]]) < records.index(scheduled[0]) < records.index(second)
        ):
            problems.append("retry_preceded_failed_response")
        second_shape = evidence.get(second["call"])
        if (
            second_shape is None
            or eligible(second_shape)
            or (
                second_shape.get("content_empty") is True
                and not second_shape.get("tool_payload_present")
                and not second_shape.get("refusal_present")
            )
        ):
            problems.append("retry_failed")
        if len(problems) == prior_problems:
            result["recovered_calls"].append(
                {"logical_call": logical, "failed_attempt": first["call"], "successful_attempt": second["call"]}
            )
    if sorted(expected_schedules) != sorted(s.get("failed_call") for s in schedules):
        problems.append("unmatched_retry_schedule")
    result["recovery_valid"] = bool(requests) and not problems
    return result
