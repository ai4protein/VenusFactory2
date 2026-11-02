"""Typed tool admission and actual isolated execution with simulated model records."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import ValidationError
from test_paper_step_goal import execute_simulation
from test_paper_step_tools import lifecycle as lifecycle

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import paper_step_goal as goals
from agent.rsi.generated_step_library import GeneratedStepLibrary
from agent.rsi.paper_step_sessions import phase_task, read
from agent.rsi.research_tool_contract import validate_contract, validate_result
from agent.rsi.research_tool_goal import create_research_tool_goal, research_tool_driver
from agent.rsi.scientist_tool_context import verify_context
from agent.science_goal import GoalSpec, run_goal

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy():
    return read(ROOT / "configs/rsi/scientist_tool_policy_v1.json")


@pytest.fixture
def contract(lifecycle):
    return {
        "schema_version": "research_tool.v1",
        "tool_id": "bounded_sum",
        "category": "evaluation",
        "purpose": "Check an illustrative aggregate",
        "method_family": "summation",
        "inputs": {"type": "array", "items": {"type": "number"}},
        "outputs": {
            "type": "object",
            "properties": {"sum": {"type": "number"}},
            "required": ["sum"],
            "additionalProperties": False,
        },
        "scope": deepcopy(lifecycle.definition["scope"]),
        "limitations": ["Engineering fixture; no scientific or autonomous-tool claim"],
    }


def profile():
    return Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "simulated", "https://example.org/v1"),
        Budget(tool_calls=8),
    )


@pytest.mark.parametrize("corruption", ["scope", "input", "output", "external_ref", "missing_role"])
def test_invalid_contracts_or_policies_cannot_allocate(tmp_path, lifecycle, contract, policy, corruption):
    if corruption == "scope":
        contract["scope"]["excluded_steps"] = []
    elif corruption in {"input", "output"}:
        contract["inputs" if corruption == "input" else "outputs"] = {"type": "string"}
    elif corruption == "external_ref":
        contract["outputs"]["$ref"] = "https://example.org/private.json"
    else:
        policy["roles"].pop("SC")
    with pytest.raises((ValueError, ValidationError)):
        create_research_tool_goal(
            tmp_path / "goal", lifecycle.definition, contract, policy, profile(), "Measure", tmp_path / "library"
        )
    assert not (tmp_path / "goal").exists()


def test_new_goal_freezes_policy_contract_and_empty_library_decision(tmp_path, lifecycle, contract, policy):
    store = create_research_tool_goal(
        tmp_path / "goal", lifecycle.definition, contract, policy, profile(), "Measure", tmp_path / "library"
    )
    driver = research_tool_driver(store, None, {})
    context = driver.protocol["scientist_tool_context"]
    assert context["policy"] == policy and context["contract"] == contract
    assert context["discovery"]["action"] == "create"
    assert context["discovery"]["matches"] == []
    assert not store.read()["cycles"]
    assert store.read()["usage"]["model_calls"] == 0


def test_simulated_library_entries_are_not_scientific_donors(tmp_path, lifecycle, contract, policy, monkeypatch):
    monkeypatch.setattr(
        GeneratedStepLibrary,
        "find",
        lambda *_: [{"entry_id": "a" * 64, "origin_record_kind": "simulated_model_records"}],
    )
    store = create_research_tool_goal(
        tmp_path / "goal", lifecycle.definition, contract, policy, profile(), "Measure", tmp_path / "library"
    )
    assert read(store.root / "protocol.json")["scientist_tool_context"]["discovery"]["action"] == "create"


def test_compatible_library_is_reused_not_regenerated(tmp_path, lifecycle, contract, policy, monkeypatch):
    from agent.rsi import research_tool_goal as module

    donor = "a" * 64
    monkeypatch.setattr(
        GeneratedStepLibrary, "find", lambda *_: [{"entry_id": donor, "origin_record_kind": "recorded_model_session"}]
    )
    calls = []

    def transfer(root, definition, profile, spec, library, selected, *, tool_context):
        calls.append((selected, tool_context, spec))
        return "transferred"

    monkeypatch.setattr(module, "create_transfer_goal", transfer)
    assert (
        create_research_tool_goal(
            tmp_path / "goal", lifecycle.definition, contract, policy, profile(), "Measure", tmp_path / "library"
        )
        == "transferred"
    )
    assert calls[0][0] == donor and calls[0][1]["discovery"]["action"] == "reuse"
    assert "verified_source_imported" in calls[0][2].criteria


def test_actual_output_must_obey_declared_schema(lifecycle, contract):
    validate_contract(contract, lifecycle.definition)
    with pytest.raises(ValidationError):
        validate_result(contract, [1, 2], {"sum": 3, "unadmitted": True})
    with pytest.raises(ValidationError):
        validate_result(contract, [1, 2], {"sum": "3"})


@pytest.mark.asyncio
async def test_context_survives_real_validation_and_new_session_reuse(
    tmp_path, lifecycle, contract, policy, monkeypatch
):
    # Fake model provenance, real isolated source execution and host subprocess.
    p = replace(profile(), mode="science_agent", harness="kimi-code")
    context = {
        "contract": contract,
        "policy": policy,
        "discovery": {"library": str(tmp_path / "library"), "matches": [], "action": "create", "selected_entry": None},
    }
    spec = GoalSpec(
        "Implement and reuse a typed computation",
        goals.CRITERIA,
        max_cycles=4,
        model_calls=32,
        tool_calls=32,
        wall_seconds=7200,
    )
    store = goals.create_step_goal(
        tmp_path / "goal", lifecycle.definition, p, spec, condition="persistent_agent_ablation", tool_context=context
    )
    driver = goals.PaperStepGoal(store, "unused-test-credential", {})
    calls = []
    monkeypatch.setattr(goals, "execute_phase", execute_simulation(calls))
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "completed", state
    assert len(calls) == 2
    for folder in calls:
        verify_context(folder, context, lifecycle.definition)
        task = phase_task(folder)
        assert read(task.capability / "scientist_policy.json") == policy
    altered = deepcopy(context)
    altered["policy"]["common"] += " altered"
    with pytest.raises(ValueError, match="context"):
        verify_context(calls[0], altered, lifecycle.definition)
    library = GeneratedStepLibrary(tmp_path / "library")
    identity = library.publish(store.root)
    assert library.resolve(identity)["origin_record_kind"] == "simulated_model_records"


def test_hidden_acceptance_schema_cannot_be_bypassed_by_numerical_tolerance(tmp_path, lifecycle, contract):
    from agent.rsi.research_tool_lifecycle import ResearchToolLifecycle

    definition = deepcopy(lifecycle.definition)
    definition["tolerance"] = {"atol": 0.001, "rtol": 0.0}
    contract["outputs"]["properties"]["sum"]["type"] = "integer"
    typed = ResearchToolLifecycle(tmp_path / "typed", definition, contract)
    identity = typed.propose(
        'def reproduce(inputs):\n    return {"sum": sum(inputs) + 0.000001}\n', declared_origin="controller_fixture"
    )
    verdict = typed.assess(identity)
    assert not verdict["passed"]
    with pytest.raises(ValueError):
        typed.register(identity)
