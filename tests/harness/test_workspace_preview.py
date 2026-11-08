import hashlib
import json
import os
from pathlib import Path

import pytest

from agent.harness.workspace_preview import file_preview, text_window
from agent.harness.research_workspace import ResearchWorkspace


@pytest.mark.asyncio
async def test_explicit_text_view_keeps_byte_pagination(tmp_path):
    workspace = tmp_path/"work"
    workspace.mkdir()
    raw = ("sequence,target\n" + "ACDEFGHIKLMNPQRSTVWY,0.5\n"*100000).encode()
    (workspace/"training.csv").write_bytes(raw)
    service = ResearchWorkspace(workspace, tmp_path/"control", {"steps":[]})
    first = await service.read_research_file("training.csv", view="text")
    assert first["text"].startswith("sequence,target\n")
    assert len(first["text"].encode()) <= 20000
    assert first["truncated"] and first["bytes"] == len(raw)
    assert "sha256" not in first
    assert first["slice_sha256"] == hashlib.sha256(first["text"].encode()).hexdigest()
    second = await service.read_research_file("training.csv", offset_bytes=first["next_offset_bytes"])
    assert (first["text"]+second["text"]).encode() == raw[:second["next_offset_bytes"]]


def test_utf8_windows_round_trip_without_cutting_characters(tmp_path):
    raw = ("甲乙丙\n"*100).encode()
    (tmp_path/"notes.txt").write_bytes(raw)
    offset = 0
    texts = []
    while offset is not None:
        result = text_window(tmp_path, "notes.txt", offset_bytes=offset, max_bytes=256)
        texts.append(result["text"])
        offset = result["next_offset_bytes"]
    assert "".join(texts).encode() == raw
    whole = text_window(tmp_path, "notes.txt")
    assert whole["sha256"] == hashlib.sha256(raw).hexdigest()
    assert whole["next_offset_bytes"] is None and not whole["truncated"]


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "parent", "absolute", "hidden"])
def test_preview_keeps_session_file_isolation(tmp_path, kind):
    workspace = tmp_path/"work"
    workspace.mkdir()
    outside = tmp_path/"private.txt"
    outside.write_text("outside session")
    if kind == "symlink":
        (workspace/"data.txt").symlink_to(outside)
        name = "data.txt"
    elif kind == "hardlink":
        os.link(outside, workspace/"data.txt")
        name = "data.txt"
    elif kind == "parent":
        name = "../private.txt"
    elif kind == "absolute":
        name = str(outside)
    else:
        (workspace/".env").write_text("private")
        name = ".env"
    with pytest.raises((ValueError, OSError)):
        text_window(workspace, name)


@pytest.mark.parametrize("kwargs", [
    {"offset_bytes":-1}, {"offset_bytes":True}, {"offset_bytes":999},
    {"max_bytes":255}, {"max_bytes":200001}, {"max_bytes":True}])
def test_preview_rejects_invalid_windows(tmp_path, kwargs):
    (tmp_path/"data.txt").write_text("text")
    with pytest.raises(ValueError):
        text_window(tmp_path,"data.txt",**kwargs)


def test_binary_preview_does_not_return_binary_payload(tmp_path):
    (tmp_path/"weights.bin").write_bytes(b"state\x00weights")
    with pytest.raises(ValueError, match="Binary file"):
        text_window(tmp_path,"weights.bin")



@pytest.mark.asyncio
async def test_large_training_table_defaults_to_columns_and_three_rows(tmp_path):
    raw = ("sequence,target\n" + "ACDEFGHIKLMNPQRSTVWY,0.5\n"*100000).encode()
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]})
    (service.workspace/"training.csv").write_bytes(raw)
    result = await service.read_research_file("training.csv")
    assert result["preview_kind"] == "table"
    assert result["columns"] == ["sequence", "target"]
    assert result["sample_rows"] == [["ACDEFGHIKLMNPQRSTVWY", "0.5"]]*3
    assert result["rows_shown"] == 3
    assert result["bytes"] == len(raw) and result["truncated"]
    assert len(json.dumps(result)) < 1000
    assert "next_offset_bytes" not in result and "text" not in result
    assert result["source_prefix_sha256"] == hashlib.sha256(raw[:20000]).hexdigest()
    # The regular shared tool, not just the direct Python method, exposes the view.
    tool = next(tool for tool in service.workspace_tools() if tool.name == "read_session_file")
    assert tool.args_schema.model_json_schema()["properties"]["view"]["enum"] == ["auto", "text"]


@pytest.mark.parametrize("name,raw,expected", [
    ("data.tsv", "\ufeffsequence\ttarget\r\nACD\t1\r\n", [["ACD", "1"]]),
    ("data.tab", "sequence\ttarget\nACD\t2", [["ACD", "2"]]),
    ("data.csv", "sequence;target\nACD;3\n", [["ACD", "3"]]),
    ("data.CSV", 'sequence,target,notes\nACD,4,"first line\nsecond, quoted line"\n',
     [["ACD", "4", "first line\nsecond, quoted line"]]),
])
def test_table_dialects_and_complete_records(tmp_path, name, raw, expected):
    (tmp_path/name).write_bytes(raw.encode())
    result = file_preview(tmp_path, name)
    assert result["columns"][:2] == ["sequence", "target"]
    assert result["sample_rows"] == expected
    assert not result["truncated"]
    assert result["source_prefix_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


def test_preview_bounds_wide_tables_and_marks_clipped_values(tmp_path):
    columns = ["column"+str(i) for i in range(100)]
    columns[0] = "h"*500
    raw = ",".join(columns)+"\n"+(",".join(["A"*500]*100)+"\n")*5
    (tmp_path/"wide.csv").write_text(raw)
    result = file_preview(tmp_path, "wide.csv", max_bytes=200000)
    assert result["column_count"] == 100
    assert len(result["columns"]) == 32 and len(result["columns"][0]) == 128
    assert result["omitted_columns"] == 68
    assert result["source_prefix_bytes"] == 65536
    assert result["clipped_column_indexes"] == [0]
    assert result["clipped_cell_indexes"] == [[0, i] for i in range(32)]
    assert all(len(value) == 80 for value in result["sample_rows"][0])
    assert result["sample_row_field_counts"] == [100]
    assert len(json.dumps(result)) < 6000


@pytest.mark.parametrize("ending", ["A"*400+",2\n", '"'+("A\n"*200)+'",2\n'])
def test_incomplete_prefix_never_returns_a_partial_record(tmp_path, ending):
    (tmp_path/"data.csv").write_text("sequence,target\nACD,1\n"+ending)
    result = file_preview(tmp_path, "data.csv", max_bytes=256)
    assert result["sample_rows"] == [["ACD", "1"]]
    assert result["truncated"]
    assert "incomplete_line_omitted" in result or "parse_notice" in result


def test_complete_malformed_csv_reports_parse_issue(tmp_path):
    (tmp_path/"data.csv").write_text('sequence,target\nACD,1\n"unfinished,2')
    result = file_preview(tmp_path, "data.csv")
    assert result["sample_rows"] == [["ACD", "1"]]
    assert "Malformed" in result["parse_notice"]
    assert result["truncated"]


@pytest.mark.parametrize("raw", ["", "sequence,target\n", "sequence,target\nACD,1\n\nDEF,2"])
def test_empty_or_short_table_does_not_invent_rows(tmp_path, raw):
    (tmp_path/"data.csv").write_text(raw)
    result = file_preview(tmp_path, "data.csv")
    assert result["rows_shown"] == len([s for s in raw.splitlines()[1:] if s])
    assert not result["truncated"]


@pytest.mark.parametrize("max_bytes", [256, 20000])
def test_protected_table_notice_remains_a_notice(tmp_path, max_bytes):
    raw = json.dumps({"data_access":"protected_research_table", "source_file":"data.csv",
                      "sha256":"a"*64, "columns":["sequence", "target"],
                      "next_step":"Use prepare_research_dataset. Only training rows will be exposed."})
    (tmp_path/"data.csv").write_text(raw)
    result = file_preview(tmp_path, "data.csv", max_bytes=max_bytes)
    assert "preview_kind" not in result
    assert result["text"] == raw[:max_bytes]
    assert "protected_research_table" in result["text"]


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "parent", "absolute", "hidden"])
def test_table_preview_preserves_session_boundary(tmp_path, kind):
    workspace = tmp_path/"work"
    workspace.mkdir()
    outside = tmp_path/"other.csv"
    outside.write_text("secret,label\nprivate,1\n")
    if kind == "symlink":
        (workspace/"data.csv").symlink_to(outside)
        name = "data.csv"
    elif kind == "hardlink":
        os.link(outside, workspace/"data.csv")
        name = "data.csv"
    elif kind == "parent":
        name = "../other.csv"
    elif kind == "absolute":
        name = str(outside)
    else:
        (workspace/".hidden.csv").write_text(outside.read_text())
        name = ".hidden.csv"
    with pytest.raises((ValueError, OSError)):
        file_preview(workspace, name)


@pytest.mark.parametrize("kwargs", [
    {"max_bytes":True}, {"max_bytes":200001}, {"offset_bytes":True}, {"view":"everything"}
])
def test_table_preview_does_not_bypass_argument_validation(tmp_path, kwargs):
    (tmp_path/"data.csv").write_text("sequence,target\nACD,1\n")
    with pytest.raises(ValueError):
        file_preview(tmp_path, "data.csv", **kwargs)


def test_non_table_text_remains_exact(tmp_path):
    raw = "def run(inputs):\n    return {'score': 0.5}\n"
    (tmp_path/"model.py").write_text(raw)
    result = file_preview(tmp_path, "model.py")
    assert result["text"] == raw and result["sha256"]
    assert "preview_kind" not in result



@pytest.mark.asyncio
async def test_preview_reads_only_training_partition_from_protected_data(tmp_path):
    from agent.harness.research_data_store import ResearchDataStore

    work = tmp_path/"work"
    work.mkdir()
    raw = ("id,group,x,target\n" +
           "".join(f"sample-{i},group-{i},{i},{10000+i}\n" for i in range(60))).encode()
    store = ResearchDataStore(tmp_path/"private", work)
    store.ingest("data.csv", raw, hashlib.sha256(raw).hexdigest())
    service = ResearchWorkspace(work, tmp_path/"control", {"steps":[]}, data_store=store)

    notice = await service.read_research_file("data.csv")
    assert json.loads(notice["text"])["data_access"] == "protected_research_table"
    assert "sample-0" not in notice["text"]

    prepared = await service.prepare_research_dataset(
        "data.csv", "target", ["x"], "group", "Synthetic isolation fixture",
        "Independent fixture groups", id_column="id")
    evaluation = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    train_ids = {row["id"] for row in evaluation["splits"]["train"]}
    heldout_ids = {row["id"] for split in ("validation", "test")
                   for row in evaluation["splits"][split]}
    sample = await service.read_research_file("data.csv")
    rows = [dict(zip(sample["columns"], row)) for row in sample["sample_rows"]]
    assert len(rows) == 3
    assert {row["id"] for row in rows} <= train_ids
    assert not {row["id"] for row in rows} & heldout_ids
    assert store.original_bytes("data.csv") == raw
