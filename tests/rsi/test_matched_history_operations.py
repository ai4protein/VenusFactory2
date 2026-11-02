"""Operational cost, missing usage and exact history-exposure boundaries."""

import hashlib
import json
from copy import deepcopy

import pytest

from agent.rsi import matched_history_operations as ops


def records(*, usage=None):
    return [
        {"event": "request", "call": 1, "model": "free-test"},
        {
            "event": "response",
            "call": 1,
            "status": 200,
            "completed": True,
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "wall_seconds": 2.5,
        },
    ]


def test_usage_is_counted_once_across_episodes_with_restarted_ids():
    first = ops.request_counts(records(), model="free-test")
    total = ops.aggregate_requests([first, first])
    assert total["requests"] == total["responses"] == 2
    assert total["tokens"]["total_tokens"]["total"] == 28
    assert total["response_wall_seconds"] == 5


def test_failure_and_missing_usage_remain_in_denominator():
    rows = records()
    rows.extend(
        [
            {"event": "request", "call": 2, "model": "free-test"},
            {"event": "response", "call": 2, "status": 403, "completed": False, "usage": None, "wall_seconds": 1},
            {"event": "request", "call": 3, "model": "free-test"},
        ]
    )
    answer = ops.request_counts(rows, model="free-test")
    assert answer["requests"] == 3 and answer["unanswered_requests"] == 1
    assert answer["http_statuses"] == {"200": 1, "403": 1}
    assert answer["tokens"]["total_tokens"] == {
        "total": None,
        "known_sum": 14,
        "known_requests": 1,
        "unknown_requests": 2,
    }
    assert answer["response_wall_seconds"] is None
    assert answer["known_response_wall_seconds"] == 3.5


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_request",
        "duplicate_response",
        "unpaired_response",
        "unexpected_model",
        "bool_tokens",
        "negative_tokens",
        "negative_duration",
    ],
)
def test_invalid_request_evidence_rejected(mutation):
    rows = records()
    if mutation == "duplicate_request":
        rows.insert(1, dict(rows[0]))
    elif mutation == "duplicate_response":
        rows.append(dict(rows[1]))
    elif mutation == "unpaired_response":
        rows.reverse()
    elif mutation == "unexpected_model":
        rows[0]["model"] = "other"
    elif mutation == "bool_tokens":
        rows[1]["usage"]["total_tokens"] = True
    elif mutation == "negative_tokens":
        rows[1]["usage"]["total_tokens"] = -1
    else:
        rows[1]["wall_seconds"] = -1
    with pytest.raises(ValueError):
        ops.request_counts(rows, model="free-test")


def test_tool_errors_and_incomplete_calls_are_counted():
    calls = [
        {"kind": "tool_call_start", "tool_call_id": "a", "tool_name": "mcp__venusfactory__evaluate_model_code"},
        {"kind": "tool_result", "tool_call_id": "a", "is_error": True},
        {"kind": "tool_call_start", "tool_call_id": "b", "tool_name": "mcp__venusfactory__submit_model_code"},
    ]
    value, pairs = ops.tool_pairs(calls)
    assert value == {
        "started": 2,
        "returned": 1,
        "unreturned": 1,
        "error_results": 1,
        "by_tool": {"evaluate_model_code": 1, "submit_model_code": 1},
    }
    assert len(pairs) == 1
    with pytest.raises(ValueError):
        ops.tool_pairs(calls + [calls[1]])
    with pytest.raises(ValueError):
        ops.tool_pairs(calls + [calls[0]])


def history_fixture():
    code = "def predict(x): return 0\n"
    packet = {
        "policy": "recent.v1",
        "attempts": [{"source_code": code, "source_sha256": hashlib.sha256(code.encode()).hexdigest(), "episode": 0}],
    }
    cycle = {"index": 2, "phase": "develop", "training_history_sha256": ops.digest(packet)}
    prompt = ops.HISTORY_MARKER + "\n" + json.dumps(packet, ensure_ascii=False)
    return packet, cycle, prompt


def test_history_counts_programs_not_policy_assignments():
    packet, cycle, prompt = history_fixture()
    value = ops.check_history(packet, expected=packet, prompt=prompt, cycle=cycle, policy="recent.v1")
    assert value["sources"] == 1 and value["has_programs"]
    packet["attempts"] = []
    cycle["training_history_sha256"] = ops.digest(packet)
    prompt = ops.HISTORY_MARKER + "\n" + json.dumps(packet, ensure_ascii=False)
    value = ops.check_history(packet, expected=packet, prompt=prompt, cycle=cycle, policy="recent.v1")
    assert not value["has_programs"] and value["bytes"] > 0


@pytest.mark.parametrize(
    "mutation",
    [
        "projection",
        "prompt",
        "hash",
        "future_source",
        "source_code",
        "duplicate_source",
        "too_large",
        "wrong_phase",
        "wrong_policy",
    ],
)
def test_history_projection_or_prompt_tampering_rejected(mutation):
    packet, cycle, prompt = history_fixture()
    expected = deepcopy(packet)
    policy = "recent.v1"
    if mutation == "projection":
        packet["extra"] = True
    elif mutation == "prompt":
        prompt = "ordinary prompt"
    elif mutation == "hash":
        cycle["training_history_sha256"] = "wrong"
    elif mutation == "wrong_phase":
        cycle["phase"] = "reuse"
    elif mutation == "wrong_policy":
        policy = "different"
    else:
        if mutation == "future_source":
            packet["attempts"][0]["episode"] = 2
        elif mutation == "source_code":
            packet["attempts"][0]["source_code"] += "changed"
        elif mutation == "duplicate_source":
            packet["attempts"].append(dict(packet["attempts"][0]))
        else:
            packet["padding"] = "x" * 50000
        expected = deepcopy(packet)
        cycle["training_history_sha256"] = ops.digest(packet)
        prompt = ops.HISTORY_MARKER + "\n" + json.dumps(packet, ensure_ascii=False)
    with pytest.raises(ValueError):
        ops.check_history(packet, expected=expected, prompt=prompt, cycle=cycle, policy=policy)


def test_no_history_and_reuse_must_not_receive_history():
    answer = ops.check_history(None, expected=None, prompt="ordinary", cycle={"phase": "reuse"}, policy="none")
    assert answer["bytes"] == answer["sources"] == 0
    with pytest.raises(ValueError):
        ops.check_history(None, expected=None, prompt=ops.HISTORY_MARKER, cycle={"phase": "reuse"}, policy="none")


def operational_fixture():
    design, units = {"units": [], "tasks_per_block": 2}, {}
    for i in range(2):
        for arm in ops.ARMS:
            uid = f"b{i}_{arm}"
            design["units"].append({"id": uid, "arm": arm, "project": "source", "source_block": str(i)})
            units[uid] = {
                "allocated": 2,
                "registered": 0 if arm == "no_history" else 1,
                "requests": ops.request_counts(records(), model="free-test"),
                "tool_calls": 1,
                "training_evaluation_calls_with_results": 1,
                "target_training_results": 2,
                "checkpoint_wall_seconds": 5.0,
                "host_execution_wall_seconds": 3.0,
                "campaign_wall_seconds": 6.0,
                "history": {
                    "exposed_development_cycles": 0,
                    "development_cycles": 1,
                    "source_occurrences": 0,
                    "packet_utf8_bytes": 0,
                },
            }
    return design, units


def test_operational_summary_retains_zero_acquisition_and_exposure():
    design, units = operational_fixture()
    report = ops.summarize_operations(design, units)
    arm = report["arms"]["no_history"]
    assert arm["registered"] == 0 and arm["allocated"] == 4
    assert arm["requests_per_registered_model"] is None
    assert arm["requests_per_assigned_task"] == 0.5
    assert report["arms"]["selective"]["history"]["unexposed_units"] == 2


def test_missing_tokens_do_not_select_a_convenient_subset():
    design, units = operational_fixture()
    units["b0_selective"]["requests"]["tokens"]["total_tokens"] = {
        "total": None,
        "known_sum": 0,
        "known_requests": 0,
        "unknown_requests": 1,
    }
    report = ops.summarize_operations(design, units)
    value = report["paired_cost_contrasts"]["selective_minus_recent"]["total_tokens_per_assigned_task"]
    assert value["estimate"] is None and value["coverage_complete"] is False
    assert len(value["missing_blocks"]) == 1
    assert report["arms"]["selective"]["requests"]["tokens"]["total_tokens"]["total"] is None


def test_operational_coverage_and_denominator_errors_fail():
    design, units = operational_fixture()
    units["b0_recent"]["allocated"] = 1
    with pytest.raises(ValueError):
        ops.summarize_operations(design, units)
    units.pop("b0_recent")
    with pytest.raises(ValueError):
        ops.summarize_operations(design, units)
