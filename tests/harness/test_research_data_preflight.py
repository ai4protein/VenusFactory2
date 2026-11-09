"""Training preflight uses host validation, without fitting or disclosing labels."""
import copy
import hashlib
import json

import pytest

from agent.harness.research_development import inspect_evaluation
from agent.harness.research_evaluation import validate_data
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError


def dataset():
    def rows(start, count):
        return [{"id": f"row-{i}", "group": f"group-{i}",
                 "inputs": {"x": i}, "target": i * 1.23456789}
                for i in range(start, start+count)]
    return {"task": "regression", "provenance": "Synthetic software test",
            "grouping": "Independent synthetic groups",
            "splits": {"train": rows(0, 8), "validation": rows(8, 3), "test": rows(11, 3)}}


@pytest.mark.asyncio
async def test_preflight_is_read_only_and_reports_counts_not_data(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace", tmp_path/"control", {"steps":[]})
    path=service.workspace/"evaluation.json"
    raw=json.dumps(dataset()).encode();path.write_bytes(raw)
    result=await service.validate_research_evaluation("evaluation.json")
    assert result["passed"] and result["validation_scope"]=="partition_schema_and_disjointness"
    assert result["source_sha256"]==hashlib.sha256(raw).hexdigest()
    assert result["splits"]["train"]=={"rows":8,"groups":8,"unique_inputs":8}
    assert result["splits"]["validation"]["rows"]==3
    assert path.read_bytes()==raw and service.development.records()==[]
    encoded=json.dumps(result)
    assert "1.23456789" not in encoded and "row-0" not in encoded and "group-0" not in encoded
    assert service.receipts[-1]["tool"]=="validate_research_evaluation"


@pytest.mark.parametrize("corruption", ["group_overlap","input_overlap","duplicate_id","invalid_target",
                                      "validation_size","test_size","root_type"])
def test_preflight_and_measured_validation_reject_the_same_invalid_partitions(tmp_path, corruption):
    value=dataset()
    if corruption=="group_overlap":
        value["splits"]["test"][0]["group"]=value["splits"]["train"][0]["group"]
    elif corruption=="input_overlap":
        value["splits"]["validation"][0]["inputs"]=copy.deepcopy(value["splits"]["train"][0]["inputs"])
    elif corruption=="duplicate_id":
        value["splits"]["train"][1]["id"]=value["splits"]["train"][0]["id"]
    elif corruption=="invalid_target":
        value["splits"]["test"][0]["target"]=float("nan")
    elif corruption in {"validation_size","test_size"}:
        split=corruption.split("_")[0]
        value["splits"][split]=value["splits"][split]*171
    else:
        value=[]
    (tmp_path/"evaluation.json").write_text(json.dumps(value))
    with pytest.raises(ValueError) as before:
        validate_data(value, "supervised")
    with pytest.raises(ValueError) as after:
        inspect_evaluation(tmp_path, "evaluation.json")
    assert str(before.value)==str(after.value)
    if corruption in {"validation_size","test_size"}:
        assert corruption.split("_")[0] in str(after.value) and "512" in str(after.value)


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_preflight_cannot_read_outside_the_session(tmp_path, path):
    with pytest.raises((ValueError, OSError)):
        inspect_evaluation(tmp_path, path)


def test_discovery_is_progressive_and_honors_tool_availability(tmp_path):
    service=ResearchWorkspace(tmp_path/"workspace", tmp_path/"control", {"steps":[]})
    mapping={tool.name:tool for tool in service.tools()}
    enabled=set(mapping)
    index=ToolDisclosure(mapping,schema=lambda tool:tool.get_input_schema().model_json_schema(),
                         allowed=lambda tool:tool.name in enabled)
    page=index.metadata("search_tools", {"query":"validate_research_evaluation"})
    card=next(t for t in page["tools"] if t["name"]=="validate_research_evaluation")
    assert set(card)=={"name","description"}
    description=index.metadata("describe_tool", {"name":"validate_research_evaluation"})
    assert "evaluation_file" in json.dumps(description)
    enabled.remove("validate_research_evaluation")
    with pytest.raises(DisclosureError):
        index.resolve("call_tool", {"name":"validate_research_evaluation",
                                   "arguments":{"evaluation_file":"evaluation.json"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "validation_size", "regression_target", "unlabeled_target", "unlabeled_id",
    "unlabeled_inputs", "unlabeled_duplicate", "unlabeled_type", "split_type", "root_type",
])
async def test_preflight_and_admission_share_partition_rules_before_execution(tmp_path, monkeypatch, fault):
    from agent.harness.capability_review import model_evaluation
    from agent.harness import research_workspace

    value = dataset()
    if fault == "validation_size":
        value["splits"]["validation"] = [
            {"id": f"v-{i}", "group": f"vg-{i}", "inputs": {"x": 100+i}, "target": i}
            for i in range(513)]
    elif fault == "regression_target":
        value["splits"]["test"][0]["target"] = "not a numeric measurement"
    elif fault == "unlabeled_type":
        value["unlabeled"] = None
    elif fault == "split_type":
        value["splits"] = []
    elif fault == "root_type":
        value = []
    else:
        row = {"id": "unlabeled", "inputs": {"x": 999}}
        if fault == "unlabeled_target":
            row["target"] = 999
        elif fault == "unlabeled_id":
            row["id"] = value["splits"]["test"][0]["id"]
        elif fault == "unlabeled_inputs":
            row["inputs"] = copy.deepcopy(value["splits"]["validation"][0]["inputs"])
        value["unlabeled"] = [row, copy.deepcopy(row)] if fault == "unlabeled_duplicate" else [row]
    (tmp_path/"evaluation.json").write_text(json.dumps(value))
    async def forbidden(*args, **kwargs):
        raise AssertionError("Invalid evaluation data must not start model execution")
    monkeypatch.setattr(research_workspace, "invoke_bundle", forbidden)
    manifest = {"review_spec": {"evaluation_file": "evaluation.json", "input_schema": {
        "type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}}}
    with pytest.raises(ValueError) as preflight:
        inspect_evaluation(tmp_path, "evaluation.json")
    with pytest.raises(ValueError) as admission:
        await model_evaluation(tmp_path, manifest, tmp_path/"review", forbidden)
    assert str(preflight.value) == str(admission.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["meta_learning", "semi_supervised"])
async def test_admission_checks_the_selected_generation_strategy(tmp_path, monkeypatch, strategy):
    from agent.harness.capability_review import model_evaluation
    from agent.harness import research_workspace

    (tmp_path/"evaluation.json").write_text(json.dumps(dataset()))
    async def forbidden(*args, **kwargs):
        raise AssertionError("Missing strategy prerequisites must not start a worker")
    monkeypatch.setattr(research_workspace, "invoke_bundle", forbidden)
    manifest = {"review_spec": {"evaluation_file": "evaluation.json"}, "development": {
        "selected_experiment_id": "candidate",
        "generations": [{"id": "baseline", "strategy": "supervised"},
                        {"id": "candidate", "strategy": strategy}]}}
    with pytest.raises(ValueError) as preflight:
        inspect_evaluation(tmp_path, "evaluation.json", strategy)
    with pytest.raises(ValueError) as admission:
        await model_evaluation(tmp_path, manifest, tmp_path/"review", forbidden)
    assert str(preflight.value) == str(admission.value)


@pytest.mark.asyncio
async def test_retained_supervised_baseline_does_not_inherit_descendant_strategy(tmp_path, monkeypatch):
    from agent.harness.capability_review import model_evaluation
    from agent.harness import research_workspace

    (tmp_path/"evaluation.json").write_text(json.dumps(dataset()))
    class ReachedInference(Exception):
        pass
    async def inference(*args, **kwargs):
        raise ReachedInference
    monkeypatch.setattr(research_workspace, "invoke_bundle", inference)
    manifest = {"review_spec": {"evaluation_file": "evaluation.json", "input_schema": {"type": "object"}},
        "development": {"selected_experiment_id": "baseline", "generations": [
            {"id": "baseline", "strategy": "supervised"},
            {"id": "candidate", "strategy": "meta_learning"}]}}
    with pytest.raises(ReachedInference):
        await model_evaluation(tmp_path, manifest, tmp_path/"review", inference)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_valid_unlabeled_pool_is_allowed_for_a_labeled_only_control(tmp_path, task):
    value = dataset()
    value["task"] = task
    if task == "classification":
        for rows in value["splits"].values():
            for row in rows:
                row["target"] = "soluble" if row["inputs"]["x"] % 2 else "insoluble"
    value["unlabeled"] = [{"id": "unlabeled", "inputs": {"x": 999}}]
    (tmp_path/"evaluation.json").write_text(json.dumps(value))
    before = inspect_evaluation(tmp_path, "evaluation.json", "supervised")
    after = inspect_evaluation(tmp_path, "evaluation.json", "semi_supervised")
    assert before["evaluation_sha256"] == after["evaluation_sha256"]
    assert before["unlabeled_rows"] == after["unlabeled_rows"] == 1
