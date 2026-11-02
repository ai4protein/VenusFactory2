"""Recovered inspections cannot bypass required calls or scientific acceptance."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_generated_lifecycle import spec
from run_protein_lifecycle import audit_events


def call(name, identity, *, error=False):
    return [
        {"kind": "tool_call_start", "tool_call_id": identity, "tool_name": "mcp__venusfactory__" + name},
        {"kind": "tool_result", "tool_call_id": identity, "is_error": error},
    ]


def audit(tmp_path, trace, *, status="completed", retryable=True):
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(x) for x in trace) + "\n")
    declaration = spec("reuse")
    return audit_events(
        tmp_path,
        "reuse",
        {"status": status},
        declaration["required_tools"],
        declaration["recoverable_tools"],
        declaration["retryable_read_tools"] if retryable else set(),
    )


def complete_chain():
    return call("discover_generated_models", "discover") + call("predict_with_generated_model", "predict")


def test_corrected_read_only_call_is_recorded_without_replaying_prediction(tmp_path):
    trace = (
        call("inspect_model_code_task", "bad", error=True) + call("inspect_model_code_task", "retry") + complete_chain()
    )
    result = audit(tmp_path, trace)
    assert result["passed"]
    assert result["read_tool_retries"] == [
        {"tool": "inspect_model_code_task", "failed_call_id": "bad", "successful_retry_call_id": "retry"}
    ]
    assert result["unrecovered_tool_failures"] == []
    assert not audit(tmp_path, trace, retryable=False)["passed"]


@pytest.mark.parametrize(
    "scenario", ["missing", "earlier", "concurrent", "error", "different", "unpaired", "duplicate", "reverse"]
)
def test_a_retry_must_be_later_paired_success_for_the_same_tool(tmp_path, scenario):
    failed = call("inspect_model_code_task", "bad", error=True)
    retry = call("inspect_model_code_task", "retry")
    traces = {
        "missing": failed,
        "earlier": retry + failed,
        "concurrent": [failed[0], retry[0], failed[1], retry[1]],
        "error": failed + call("inspect_model_code_task", "retry", error=True),
        "different": failed + call("discover_generated_models", "other"),
        "unpaired": failed + retry[:1],
        "duplicate": failed + call("inspect_model_code_task", "bad"),
        "reverse": failed + list(reversed(retry)),
    }
    assert not audit(tmp_path, traces[scenario] + complete_chain())["passed"]


def test_mutating_tool_failure_is_not_reclassified_as_a_read_retry(tmp_path):
    trace = call("predict_with_generated_model", "bad", error=True) + complete_chain()
    result = audit(tmp_path, trace)
    assert not result["passed"]
    assert result["unrecovered_tool_failures"] == ["predict_with_generated_model"]


@pytest.mark.parametrize("status", ["runtime_error", "timeout", "incomplete"])
def test_runtime_and_critic_failure_remain_fatal(tmp_path, status):
    trace = (
        call("inspect_model_code_task", "bad", error=True) + call("inspect_model_code_task", "retry") + complete_chain()
    )
    assert not audit(tmp_path, trace, status=status)["passed"]


def test_recovery_does_not_replace_missing_required_prediction(tmp_path):
    trace = (
        call("inspect_model_code_task", "bad", error=True)
        + call("inspect_model_code_task", "retry")
        + call("discover_generated_models", "discover")
    )
    assert not audit(tmp_path, trace)["passed"]


def test_all_read_errors_need_recovery_not_only_the_first(tmp_path):
    trace = (
        call("inspect_model_code_task", "bad1", error=True)
        + call("inspect_model_code_task", "retry")
        + call("inspect_model_code_task", "bad2", error=True)
        + complete_chain()
    )
    result = audit(tmp_path, trace)
    assert not result["passed"]
    assert len(result["read_tool_retries"]) == 1


def test_generated_model_spec_declares_only_read_only_inspections():
    for phase in ("develop", "reuse"):
        declaration = spec(phase)
        assert declaration["retryable_read_tools"] == {"inspect_model_code_task", "discover_generated_models"}
        assert declaration["retryable_read_tools"] <= declaration["tools"]
        assert "submit_model_code" not in declaration["retryable_read_tools"]
