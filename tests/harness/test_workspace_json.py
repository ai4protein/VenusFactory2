"""Behavioral checks for progressive, exact reads of scientific JSON artifacts."""

import hashlib
import json
import os

import pytest
from pydantic import ValidationError

from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.session_boundary import mcp_input_schema, prepare_tool_arguments
from agent.harness.workspace_json import JSON_INPUT_LIMIT, JsonSelection
from agent.harness.workspace_preview import file_preview


def write_json(root, name, value):
    raw = json.dumps(value, ensure_ascii=False).encode()
    (root / name).write_bytes(raw)
    return raw


def result_fixture():
    return {
        "inputs": [{"id": f"protein-{i}", "temperature": 20 + i} for i in range(500)],
        "predictions": [{"id": f"protein-{i}", "prediction": i / 997} for i in range(500)],
        "scope": {"unit": "ratio_to_reference", "bounds": {"temperature": [20, 70]}},
    }


def test_large_result_indexes_inputs_outputs_and_scope_without_partial_rows(tmp_path):
    value = result_fixture()
    raw = write_json(tmp_path, "result.json", value)
    result = file_preview(tmp_path, "result.json")
    assert result["preview_kind"] == "json_structure" and result["truncated"]
    assert "text" not in result
    entries = {tuple(entry["path"]): entry for entry in result["structure"]}
    assert entries[("inputs",)]["length"] == 500
    assert entries[("predictions",)]["length"] == 500
    assert entries[("scope", "unit")]["value"] == "ratio_to_reference"
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(json.dumps(result).encode()) < 2000
    assert (tmp_path / "result.json").read_bytes() == raw


@pytest.mark.parametrize("archive", [False, True])
def test_record_lookup_returns_exact_predictions_and_reports_missing_ids(tmp_path, archive):
    value = result_fixture()
    if archive:
        value = {"arguments": {"inputs": value["inputs"]}, "result": value}
    raw = write_json(tmp_path, "records.json", value)
    path = ["result", "predictions"] if archive else ["predictions"]
    result = file_preview(
        tmp_path,
        "records.json",
        select={
            "path": path,
            "match_field": "id",
            "match_values": ["protein-63", "protein-127", "protein-191", "absent"],
        },
    )
    assert result["value"] == [{"id": f"protein-{i}", "prediction": i / 997} for i in [63, 127, 191]]
    assert result["missing_values"] == ["absent"]
    assert result["indices"] == [63, 127, 191] and result["matched_items"] == 3
    assert result["next_offset"] is None and not result["truncated"]
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    scope = file_preview(tmp_path, "records.json", select={"path": (["result"] if archive else []) + ["scope"]})
    assert scope["value"]["bounds"] == {"temperature": [20, 70]}
    assert (tmp_path / "records.json").read_bytes() == raw


def test_exact_string_matching_does_not_coerce_or_guess(tmp_path):
    values = [{"id": 1}, {"id": True}, {"id": "1"}, {"id": "11"}, {"id": "1"}, {"other": "1"}]
    write_json(tmp_path, "records.json", values)
    result = file_preview(tmp_path, "records.json", select={"match_field": "id", "match_values": ["1"]})
    assert result["value"] == [values[2], values[4]] and result["indices"] == [2, 4]
    assert result["matched_items"] == 2


def test_paged_matching_keeps_source_order_and_covers_every_record(tmp_path):
    values = [{"id": "a" if i % 2 else "b", "value": i} for i in range(25)]
    write_json(tmp_path, "records.json", values)
    offset = 0
    collected = []
    indices = []
    while offset is not None:
        result = file_preview(
            tmp_path, "records.json", select={"match_field": "id", "match_values": ["a"], "offset": offset, "limit": 3}
        )
        collected.extend(result["value"])
        indices.extend(result["indices"])
        offset = result["next_offset"]
        assert result["matched_items"] == 12
    assert collected == values[1::2] and indices == list(range(1, 25, 2))


def test_value_budget_never_cuts_a_record_or_utf8_string(tmp_path):
    values = [{"id": str(i), "sequence": "甲" * 20} for i in range(10)]
    write_json(tmp_path, "records.json", values)
    offset = 0
    collected = []
    while offset is not None:
        result = file_preview(tmp_path, "records.json", max_bytes=256, select={"offset": offset})
        assert len(json.dumps(result["value"], ensure_ascii=False, separators=(",", ":")).encode()) <= 256
        assert result["value"] == values[offset : offset + result["returned_items"]]
        collected.extend(result["value"])
        offset = result["next_offset"]
    assert collected == values


def test_oversized_selected_record_is_not_presented_as_complete(tmp_path):
    write_json(tmp_path, "huge.json", [{"id": "x", "sequence": "A" * 40000}])
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        file_preview(tmp_path, "huge.json", select={})
    small = file_preview(tmp_path, "huge.json", select={"path": [0, "id"]})
    assert small["value"] == "x" and not small["truncated"]


def test_empty_arrays_no_matches_and_array_indexes(tmp_path):
    write_json(tmp_path, "records.json", {"rows": [{"id": "a", "value": 7}], "empty": []})
    none = file_preview(
        tmp_path, "records.json", select={"path": ["rows"], "match_field": "id", "match_values": ["missing"]}
    )
    assert none["value"] == [] and none["missing_values"] == ["missing"] and none["matched_items"] == 0
    assert file_preview(tmp_path, "records.json", select={"path": ["empty"]})["value"] == []
    assert file_preview(tmp_path, "records.json", select={"path": ["rows", 0, "value"]})["value"] == 7
    with pytest.raises(ValueError, match="offset"):
        file_preview(tmp_path, "records.json", select={"path": ["rows"], "offset": 2})


@pytest.mark.parametrize(
    "selection",
    [
        {"path": [True]},
        {"path": [-1]},
        {"path": [1.2]},
        {"path": ("rows",)},
        {"match_field": "id"},
        {"match_values": ["x"]},
        {"match_field": "id", "match_values": []},
        {"match_field": "id", "match_values": [1]},
        {"offset": True},
        {"limit": 101},
        {"path": ["a"] * 17},
        {"unexpected": "field"},
    ],
)
def test_selection_rejects_ambiguous_or_unbounded_parameters(selection):
    with pytest.raises(ValidationError):
        JsonSelection.model_validate(selection)


@pytest.mark.parametrize(
    "raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e400}', '{"x":', "[" * 1200 + "]" * 1200]
)
def test_ambiguous_malformed_or_unbounded_json_is_not_a_successful_selection(tmp_path, raw):
    (tmp_path / "data.json").write_text(raw)
    with pytest.raises(ValueError, match="valid UTF-8"):
        file_preview(tmp_path, "data.json", select={})


def test_large_json_limit_does_not_block_explicit_raw_pagination(tmp_path):
    (tmp_path / "data.json").write_text('"' + ("A" * JSON_INPUT_LIMIT) + '"')
    with pytest.raises(ValueError, match="8 MiB"):
        file_preview(tmp_path, "data.json")
    raw = file_preview(tmp_path, "data.json", view="text", max_bytes=256)
    assert len(raw["text"]) == 256 and raw["next_offset_bytes"] == 256


def test_small_json_retains_existing_text_contract(tmp_path):
    raw = write_json(tmp_path, "data.json", {"path": "/workspace/literal", "value": 7})
    result = file_preview(tmp_path, "data.json")
    assert result["text"].encode() == raw and result["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kwargs", [{"view": "text"}, {"offset_bytes": 1}])
def test_selection_cannot_silently_be_ignored(tmp_path, kwargs):
    write_json(tmp_path, "data.json", {"value": 7})
    with pytest.raises(ValueError, match="JSON select requires"):
        file_preview(tmp_path, "data.json", select={}, **kwargs)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "parent", "absolute", "hidden"])
def test_structured_reads_preserve_the_session_boundary(tmp_path, kind):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "private.json"
    write_json(tmp_path, "private.json", {"private": "not in this session"})
    if kind == "symlink":
        (work / "data.json").symlink_to(outside)
        name = "data.json"
    elif kind == "hardlink":
        os.link(outside, work / "data.json")
        name = "data.json"
    elif kind == "parent":
        name = "../private.json"
    elif kind == "absolute":
        name = str(outside)
    else:
        (work / ".private.json").write_bytes(outside.read_bytes())
        name = ".private.json"
    with pytest.raises((ValueError, OSError)):
        file_preview(work, name, select={})


@pytest.mark.asyncio
async def test_shared_mcp_schema_and_virtual_path_expose_the_same_selection(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tool = next(t for t in service.workspace_tools() if t.name == "read_session_file")
    schema = mcp_input_schema(tool)
    assert "select" in schema["properties"] and "JsonSelection" in schema["$defs"]
    write_json(service.workspace, "data.json", {"predictions": [{"id": "x", "prediction": 0.7}]})
    query = {
        "path": "/workspace/data.json",
        "select": {"path": ["predictions"], "match_field": "id", "match_values": ["x"]},
    }
    args = prepare_tool_arguments(tool.name, tool, query, service.workspace)
    result = await tool.ainvoke(args)
    assert result["value"] == [{"id": "x", "prediction": 0.7}]
    assert query["path"] == "/workspace/data.json"
    assert args["select"] == query["select"]


@pytest.mark.asyncio
async def test_non_json_materials_do_not_ignore_selection(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    for name in ["paper.pdf", "data.csv"]:
        with pytest.raises(ValueError, match="JSON select requires"):
            await service.read_research_file(name, select=JsonSelection())


def test_nesting_limit_ignores_brackets_and_escaped_quotes_inside_strings(tmp_path):
    value = {"description": "[" * 1500 + '"' + "]" * 1500, "prediction": 0.7}
    write_json(tmp_path, "data.json", value)
    assert file_preview(tmp_path, "data.json", select={})["value"] == value
    (tmp_path / "nested.json").write_text("[" * 64 + "7" + "]" * 64)
    assert file_preview(tmp_path, "nested.json", select={})["matched_items"] == 1
    (tmp_path / "nested.json").write_text("[" * 65 + "7" + "]" * 65)
    with pytest.raises(ValueError, match="64 nested"):
        file_preview(tmp_path, "nested.json", select={})


@pytest.mark.asyncio
async def test_structured_read_does_not_block_the_event_loop(tmp_path, monkeypatch):
    import asyncio
    import threading

    import agent.harness.workspace_preview as previews

    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    started, release = threading.Event(), threading.Event()

    def slow_read(*args, **kwargs):
        started.set()
        assert release.wait(3), "The event loop could not release the file reader"
        return {"value": 0.7}

    monkeypatch.setattr(previews, "file_preview", slow_read)
    task = asyncio.create_task(service.read_research_file("result.json", select=JsonSelection()))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        release.set()
        assert await task == {"value": 0.7}
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_brief_disclosure_is_enough_to_select_exact_records_and_recover_bad_fields(tmp_path):
    from agent.harness.tool_disclosure import ToolDisclosure
    from mcp_tools.common.tool_arguments import ToolArgumentError

    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    tool = next(t for t in service.workspace_tools() if t.name == "read_session_file")
    catalog = ToolDisclosure({tool.name: tool}, schema=mcp_input_schema, allowed=lambda tool: True, direct={tool.name})
    public = next(t for t in catalog.surface() if t["name"] == tool.name)
    assert len(public["description"]) <= 180
    assert "JSON fields and records by ID" in public["description"]
    assert "match_values" in public["inputSchema"]["properties"]["select"]["description"]
    assert len(json.dumps(public)) < 4000
    value = result_fixture()
    raw = write_json(service.workspace, "results.json", value)
    bad = {"path": "/workspace/results.json", "select": {"path": ["predictions"], "view": "text"}}
    with pytest.raises(ToolArgumentError) as error:
        prepare_tool_arguments(tool.name, tool, bad, service.workspace)
    assert "Declared object fields for select: limit, match_field, match_values, offset, path." in str(error.value)
    assert "view" not in str(error.value)
    corrected = {
        "path": bad["path"],
        "select": {
            "path": ["predictions"],
            "match_field": "id",
            "match_values": ["protein-17", "protein-420"],
        },
    }
    result = await tool.ainvoke(prepare_tool_arguments(tool.name, tool, corrected, service.workspace))
    assert result["value"] == [value["predictions"][17], value["predictions"][420]]
    assert not result["truncated"] and result["missing_values"] == []
    assert (service.workspace / "results.json").read_bytes() == raw
