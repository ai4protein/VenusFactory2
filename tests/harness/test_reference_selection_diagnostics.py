"""Replay the r7 malformed reference selection without choosing a model for the agent."""
import hashlib

import pytest

from agent.harness.research_algorithm_checks import algorithm_diagnostics
from agent.harness.research_reference_baselines import reference_baseline
from agent.harness.research_workspace import ResearchWorkspace


@pytest.mark.asyncio
@pytest.mark.parametrize("task,selected", [
    ("regression", "aac_ridge"), ("multilabel", "training_prevalence"),
])
async def test_missing_reference_explains_catalog_then_explicit_save_without_side_effects(tmp_path, task, selected):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tool = next(t for t in service.tools() if t.name == "get_research_baseline")
    assert "only task=" in tool.description.split(".", 1)[0]
    expected = reference_baseline(task=task)
    # This is the actual failing r7 call shape. A filename must not select an algorithm.
    with pytest.raises(ValueError) as caught:
        await tool.ainvoke({"task": task, "output_file": "src/baseline.py"})
    message = str(caught.value)
    assert len(message) < 1300
    assert expected["scope"] in message
    assert all(item["id"] in message for item in expected["references"])
    assert f"get_research_baseline(task={task!r}) without reference or output_file" in message
    assert "reference='<selected ID>'" in message and "No source was written" in message
    assert not list(service.workspace.iterdir())
    assert not service.receipts and not service.development.records() and not service.proposals
    assert not list((service.development.root / "training").glob("*/receipt.json"))

    catalog = await tool.ainvoke({"task": task})
    assert catalog == expected and "source" not in catalog
    assert not list(service.workspace.iterdir())
    saved = await tool.ainvoke({"task": task, "reference": selected, "output_file": "src/baseline.py"})
    source = (service.workspace / "src/baseline.py").read_bytes()
    assert saved["reference_id"] == selected and saved["kind"] == "platform_reference"
    assert hashlib.sha256(source).hexdigest() == saved["source_sha256"]
    assert algorithm_diagnostics(source.decode())["valid"]
    assert not service.development.records() and not service.proposals
    assert not list((service.development.root / "training").glob("*/receipt.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("task,selected", [
    ("regression", "aac_ridge"), ("multilabel", "training_prevalence"),
])
@pytest.mark.parametrize("destination", ["../escape.py", "/tmp/escape.py", ".hidden/escape.py", "input.csv"])
async def test_selection_guidance_does_not_relax_scoped_python_destination(tmp_path, task, selected, destination):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    with pytest.raises((ValueError, OSError)):
        await service.get_research_baseline(reference=selected, task=task, output_file=destination)
    assert not list(service.workspace.rglob("*.py"))
    assert not service.receipts and not service.development.records() and not service.proposals


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["regression", "multilabel"])
async def test_unknown_reference_still_rejects_without_selecting_a_default(tmp_path, task):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    with pytest.raises(ValueError, match="Choose a listed"):
        await service.get_research_baseline(reference="not-a-reference", task=task, output_file="src/baseline.py")
    assert not list(service.workspace.iterdir())
    assert not service.receipts and not service.development.records() and not service.proposals
