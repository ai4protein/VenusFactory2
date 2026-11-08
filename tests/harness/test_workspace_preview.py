import hashlib
import os
from pathlib import Path

import pytest

from agent.harness.workspace_preview import text_window
from agent.harness.research_workspace import ResearchWorkspace


@pytest.mark.asyncio
async def test_large_training_table_can_be_inspected_without_loading_it_into_context(tmp_path):
    workspace = tmp_path/"work"
    workspace.mkdir()
    raw = ("sequence,target\n" + "ACDEFGHIKLMNPQRSTVWY,0.5\n"*100000).encode()
    (workspace/"training.csv").write_bytes(raw)
    service = ResearchWorkspace(workspace, tmp_path/"control", {"steps":[]})
    first = await service.read_research_file("training.csv")
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
