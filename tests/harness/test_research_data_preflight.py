"""Training preflight uses host validation, without fitting or disclosing labels."""
import copy
import hashlib
import json

import pytest

from agent.harness.research_development import inspect_evaluation, validate_data
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
