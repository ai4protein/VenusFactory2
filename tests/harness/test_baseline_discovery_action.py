"""A prepared sequence study first inspects controls without choosing an algorithm."""
import copy

import pytest

from agent.harness.autonomous_research import development_next_action
from agent.harness.research_focus import repair_action, selected_tools
from agent.harness.research_workspace import ResearchWorkspace


def prepared(task="regression", fields=None):
    return {"evaluation_file": "data/evaluation.json", "task": task,
            "model_interface": {"input_fields": fields or {"sequence": "string", "substrate_smiles": "string"}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["regression", "multilabel"])
async def test_initial_action_resolves_readonly_reference_catalog_without_materializing_or_fitting(tmp_path, task):
    dataset = prepared(task)
    original = copy.deepcopy(dataset)
    action = development_next_action({"generations": []}, {}, dataset)
    assert action["stage"] == "evaluate_baseline"
    assert action["tool"] == "get_research_baseline"
    assert action["arguments"] == {"task": task}
    assert action["evaluation_file"] == dataset["evaluation_file"]
    assert dataset == original
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tools = service.tools()
    context = {"plan": {"deliverable": "model"}, "model_development": {
        "prepared_dataset": dataset, "next_action": action}}
    assert action["tool"] in selected_tools(context, tools)
    result = await service.get_research_baseline(**action["arguments"])
    assert result
    assert not list(service.workspace.rglob("*.py"))
    assert not service.development.records()
    assert not list((service.development.root / "training").glob("*/receipt.json"))
    assert "sequence-only" in action["instruction"]


@pytest.mark.parametrize("dataset", [prepared("classification"), prepared(fields={"temperature": "string"}),
    {"evaluation_file": "data/evaluation.json"}])
def test_other_inputs_keep_complete_implementation_path(dataset):
    action = development_next_action({"generations": []}, {}, dataset)
    assert action["tool"] == "write_session_file"
    assert "arguments" not in action


def test_uploaded_source_and_existing_measurements_take_precedence():
    dataset = prepared()
    source = {"files": [{"path": "inputs/method.py", "sha256": "a" * 64}]}
    action = development_next_action({"generations": []}, {}, dataset, materials=source)
    assert action["tool"] == "read_session_file"
    assert action["supplied_sources"] == source["files"]
    assert "arguments" not in action
    baseline = {"generations": [{"id": "measured", "variant": "baseline"}]}
    assert development_next_action(baseline, {}, dataset)["stage"] == "evaluate_revision"
    candidate = {"generations": [{"id": "child", "variant": "candidate", "parent": "measured"}]}
    assert development_next_action(candidate, {}, dataset)["stage"] == "submit_for_review"


def test_actual_source_repair_still_precedes_discovery():
    action = development_next_action({"generations": []}, {}, prepared())
    repair = {"path": "src/method.py", "sha256": "b" * 64, "code": "broken source",
              "redacted": False, "source_check": {"valid": False}}
    updated = repair_action(action, repair)
    assert updated["stage"] == "repair_model"
    assert updated["tool"] == "write_session_file"
    assert "arguments" not in updated
    assert updated["expected_sha256"] == repair["sha256"]
    assert updated["evaluation_file"] == action["evaluation_file"]
