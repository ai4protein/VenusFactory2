"""The public MCP path contract must work with each harness' virtual root."""
import os

import pytest

from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.session_boundary import prepare_tool_arguments, relative_workspace_arguments


@pytest.mark.asyncio
async def test_virtual_file_roundtrip_uses_anchored_session_access(tmp_path):
    workspace = tmp_path/"work"
    service = ResearchWorkspace(workspace, tmp_path/"control", {"steps":[]})
    tools = {tool.name:tool for tool in service.workspace_tools()}
    async def invoke(name, **arguments):
        tool = tools[name]
        return await tool.ainvoke(prepare_tool_arguments(
            name, tool, arguments, workspace))
    content = '{"path":"/workspace/example","value":7}'
    written = await invoke("write_session_file", path="/workspace/results/a.json", content=content)
    assert written["path"] == "results/a.json"
    read = await invoke("read_session_file", path="/workspace/results/a.json")
    assert read["text"] == content
    assert (workspace/"results/a.json").read_text() == content


@pytest.mark.parametrize("path", [
    "/etc/passwd", "/workspace-other/data.txt", "/workspace/../secret.txt",
    "/workspace/.env", "/workspace/results/../../secret", "/workspace/",
    "/workspace//etc/passwd", "C:/data.txt", r"/workspace/..\secret",
])
def test_virtual_paths_do_not_expand_authority(path):
    with pytest.raises(ValueError):
        relative_workspace_arguments({"path":path})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["symlink","hardlink"])
async def test_virtual_path_still_rejects_linked_files(tmp_path, kind):
    workspace = tmp_path/"work"
    service = ResearchWorkspace(workspace, tmp_path/"control", {"steps":[]})
    external = tmp_path/"private"
    external.write_text("private data")
    if kind == "symlink":
        (workspace/"linked").symlink_to(external)
    else:
        os.link(external,workspace/"linked")
    args=relative_workspace_arguments({"path":"/workspace/linked"})
    with pytest.raises((ValueError,OSError)):
        await service.read_research_file(**args)


def test_only_declared_file_arguments_are_normalized():
    inputs={"source":"/workspace/model.py","files":["/workspace/weights.bin"],
            "evaluation_file":"/workspace/eval.json","proposal_file":"",
            "content":'open("/workspace/raw.csv")',
            "test_inputs":[{"path":"/workspace/literal-input"}],
            "description":"/workspace is a virtual directory"}
    result=relative_workspace_arguments(inputs)
    assert result["source"]=="model.py" and result["files"]==["weights.bin"]
    assert result["evaluation_file"]=="eval.json" and result["proposal_file"]==""
    for key in ("content","test_inputs","description"):
        assert result[key]==inputs[key]
    assert inputs["source"]=="/workspace/model.py"


def test_project_relative_receipts_roundtrip_as_virtual_paths():
    import json
    from pathlib import Path

    import agent.harness.session_boundary as boundary
    project=Path(boundary.__file__).resolve().parents[3]
    workspace=project/"runtime/local/files/backend.web_v2/sessions/owned/2026/11/06"
    relative=workspace.relative_to(project).as_posix()
    preview=json.dumps({"path":"literal", "value":7}, indent=2)
    raw=json.dumps({"file_info":{"file_path":relative+"/results/a.json"}, "preview":preview})
    result=json.loads(boundary.public_value(raw,workspace))
    assert result["file_info"]["file_path"]=="/workspace/results/a.json"
    assert result["preview"]==preview
    assert boundary.public_value(relative+"-other/file",workspace)==relative+"-other/file"
    assert boundary.public_value(str(workspace/"results/a.json"),workspace)=="/workspace/results/a.json"

@pytest.mark.asyncio
async def test_write_rejects_read_metadata_with_actionable_fields(tmp_path):
    workspace = tmp_path/"work"
    service = ResearchWorkspace(workspace, tmp_path/"control", {"steps":[]})
    tool = next(t for t in service.workspace_tools() if t.name == "write_session_file")
    source = "# private analysis source\n" + "x = 1\n" * 4000
    bad = {"path":"src/analysis.py", "content":source, "bytes":len(source),
           "offset_bytes":0, "truncated":False}
    with pytest.raises(ValueError) as error:
        prepare_tool_arguments(tool.name, tool, bad, workspace)
    message = str(error.value)
    assert "Unexpected fields: bytes, offset_bytes, truncated" in message
    assert "Allowed fields: content, path" in message
    assert "assigned by the server" not in message
    assert source not in message and not (workspace/"src/analysis.py").exists()
    args = prepare_tool_arguments(tool.name, tool, {"path":bad["path"], "content":source},
                                  workspace)
    await tool.ainvoke(args)
    assert (workspace/"src/analysis.py").read_text() == source


def test_argument_diagnostics_do_not_echo_values_or_invalid_key_names(tmp_path):
    service = ResearchWorkspace(tmp_path/"work", tmp_path/"control", {"steps":[]})
    tool = next(t for t in service.workspace_tools() if t.name == "write_session_file")
    args = {"path":"test.txt", "content":"private", "workspace_root":"/private/account",
            "malicious\\nkey <payload>":"do not expose"}
    with pytest.raises(ValueError) as error:
        prepare_tool_arguments(tool.name, tool, args, tmp_path/"work")
    message = str(error.value)
    assert "workspace_root" in message and "<invalid field name>" in message
    assert "assigned by the server" in message
    assert "/private/account" not in message and "payload" not in message
