import asyncio
import json
import time
from functools import partial
from types import SimpleNamespace

import pytest

from agent.graph.execution import verify
from agent.rsi.engine.experience import LIMITATION
from agent.rsi.engine.langgraph_tools import build_tools
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.tool_contracts import validate_generated_result


@pytest.fixture
def bound(tmp_path):
    work, caps = tmp_path / "work", tmp_path / "caps"
    work.mkdir()
    caps.mkdir()
    (caps / "task.json").write_text(json.dumps({"phase": "develop", "inputs": {}, "scope": {}}))
    return work, caps


def checker(bound, name):
    work, caps = bound
    return partial(validate_generated_result, name, workspace=work, capability=caps)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,payload",
    [
        ("read_model_experience", {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}),
        ("discover_generated_models", {"models": []}),
    ],
)
async def test_valid_empty_operation_does_not_consume_semantic_model_budget(bound, name, payload, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Valid empty operation was misclassified")

    monkeypatch.setattr(verify, "_run_mls_post_step_verify", forbidden)
    monkeypatch.setattr(verify, "_cb_post_step_check", forbidden)
    ctx = SimpleNamespace(
        tool=SimpleNamespace(metadata={"result_validator": checker(bound, name)}), chains={"llm": object()}
    )
    assert (await verify.run_mls_post_check(ctx, json.dumps({"success": True, **payload}))).ok
    assert (await verify.run_cb_post_check(ctx, payload)).ok


@pytest.mark.asyncio
async def test_success_flag_cannot_bypass_invalid_contract(bound):
    ctx = SimpleNamespace(
        tool=SimpleNamespace(metadata={"result_validator": checker(bound, "discover_generated_models")}), chains={}
    )
    for raw in [
        "broken json",
        {"success": False, "models": []},
        {"success": True, "models": [], "error": "failed"},
        {"models": [{"model_id": "forged"}]},
    ]:
        for check in [verify.run_mls_post_check, verify.run_cb_post_check]:
            result = await check(ctx, raw)
            assert not result.ok and result.failure_type == "result_contract"


def test_generic_empty_result_is_still_not_trusted():
    assert not verify._tool_self_reports_success({"success": True, "entries": []})
    assert verify._trusted_result_check(SimpleNamespace(tool=None), {"success": True}) is None


def test_negative_training_feedback_is_not_publication(bound):
    work, _ = bound
    source = "def fit(): pass"
    import hashlib

    sha = hashlib.sha256(source.encode()).hexdigest()
    feedback = {"rmse": 10.0, "mse_skill_vs_training_mean": -3.0}
    (work / "code_attempt_1.json").write_text(json.dumps({"source_code": source, "training_feedback": feedback}))
    result = {
        "source_sha256": sha,
        "training_only_feedback": feedback,
        "attempts_remaining": 1,
        "status": "tested_not_published",
    }
    check = checker(bound, "evaluate_model_code")
    check(result)
    with pytest.raises(ValueError):
        check({**result, "status": "registered"})
    with pytest.raises(ValueError):
        check({**result, "training_only_feedback": {"rmse": 0}})
    assert not (work / "candidate.json").exists()


def test_experience_packet_cannot_be_substituted(bound):
    _, caps = bound
    value = {"schema": 1, "condition": "summary", "entries": [], "limitation": LIMITATION}
    (caps / "experience.json").write_text(json.dumps(value))
    (caps / "task.json").write_text(
        json.dumps({"phase": "develop", "scope": {}, "inputs": {}, "experience_sha256": digest(value)})
    )
    check = checker(bound, "read_model_experience")
    check(value)
    with pytest.raises(ValueError):
        check({**value, "condition": "none"})
    (caps / "experience.json").write_text(json.dumps({**value, "condition": "source"}))
    with pytest.raises(ValueError):
        check(value)


def test_missing_candidate_never_accepted_by_operation_check(bound):
    with pytest.raises(ValueError, match="missing"):
        checker(bound, "submit_model_code")({"artifact": "candidate.json", "status": "candidate", "registered": False})


class MCP:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def list_tools(self):
        return [
            SimpleNamespace(name="discover_generated_models", description="Discover", inputSchema={"properties": {}})
        ]

    async def call_tool(self, name, kwargs, **ignored):
        self.calls.append(name)
        await asyncio.sleep(0)
        return SimpleNamespace(content=[SimpleNamespace(text=self.content)], is_error=False)


@pytest.mark.asyncio
async def test_malformed_output_has_one_error_event_not_success_then_error(bound):
    work, caps = bound
    events = []
    result = {"tool_calls": 0, "tool_errors": 0}
    tools = await build_tools(
        MCP("not JSON"),
        allowed_tools={"discover_generated_models"},
        workspace=work,
        capability=caps,
        deadline=time.monotonic() + 10,
        max_tool_calls=1,
        result=result,
        event=events.append,
        namespaced=True,
    )
    with pytest.raises(ValueError):
        await tools[0].ainvoke({})
    assert len(events) == 2
    assert events[0]["tool_call_id"] == events[1]["tool_call_id"]
    assert events[1]["is_error"] is True
    assert result == {"tool_calls": 1, "tool_errors": 1}


@pytest.mark.asyncio
async def test_both_modes_share_atomic_tool_budget_and_valid_empty_result(bound):
    work, caps = bound
    for namespaced in [False, True]:
        client = MCP('{"models": []}')
        result = {"tool_calls": 0, "tool_errors": 0}
        events = []
        tools = await build_tools(
            client,
            allowed_tools={"discover_generated_models"},
            workspace=work,
            capability=caps,
            deadline=time.monotonic() + 10,
            max_tool_calls=1,
            result=result,
            event=events.append,
            namespaced=namespaced,
        )
        outputs = await asyncio.gather(tools[0].ainvoke({}), tools[0].ainvoke({}), return_exceptions=True)
        assert sum(isinstance(x, RuntimeError) for x in outputs) == 1
        assert len(client.calls) == 1 and len(events) == 2
        assert json.loads(next(x for x in outputs if isinstance(x, str))) == {"success": True, "models": []}
        assert events[0]["tool_name"] == "mcp__venusfactory__discover_generated_models"


def test_large_classifier_contract_checks_counts(bound):
    work, caps = bound
    scope = {"class_labels": ["zero", "one"]}
    rows = [{"id": "a", "sequence": "AAAA", "target": 0}, {"id": "b", "sequence": "AAAC", "target": 1}]
    (work / "training.json").write_text(json.dumps(rows))
    (caps / "task.json").write_text(
        json.dumps(
            {
                "phase": "develop",
                "inputs": {"training.json": digest(rows)},
                "scope": scope,
                "plugin_id": "generated_classifier.large.v1",
            }
        )
    )
    value = {
        "phase": "develop",
        "scope": scope,
        "abi": {"fit": "fit", "predict": "predict"},
        "constraints": ["bounded"],
        "training": {"rows": 2, "examples": rows, "target_min": 0, "target_max": 1},
        "training_class_counts": [1, 1],
    }
    checker(bound, "inspect_model_code_task")(value)
    with pytest.raises(ValueError):
        checker(bound, "inspect_model_code_task")({**value, "training_class_counts": [2, 0]})


@pytest.mark.parametrize("probabilities,prediction", [([0.7, 0.7], 0), ([0.2, 0.8], 0)])
def test_large_classifier_reuse_rejects_invalid_probability_or_class(bound, probabilities, prediction):
    work, caps = bound
    scope = {"class_labels": ["zero", "one"]}
    rows = [{"id": "a", "sequence": "AAAA"}]
    model = {"scope": scope}
    packet = {"model_id": "test", "model": model, "model_sha256": digest(model)}
    (caps / "registered").mkdir()
    (caps / "registered/capability.json").write_text(json.dumps(packet))
    (caps / "task.json").write_text(
        json.dumps(
            {
                "phase": "reuse",
                "scope": scope,
                "inputs": {"inputs.json": digest(rows)},
                "capability_sha256": digest(packet),
                "plugin_id": "generated_classifier.large.v1",
            }
        )
    )
    (work / "inputs.json").write_text(json.dumps(rows))
    predictions = [{**rows[0], "probabilities": probabilities, "prediction": prediction}]
    (work / "predictions.json").write_text(json.dumps(predictions))
    value = {"artifact": "predictions.json", "model_id": "test", "rows": 1, "prediction_sha256": digest(predictions)}
    with pytest.raises(ValueError):
        checker(bound, "predict_with_generated_model")(value)
