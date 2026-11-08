import json
from types import SimpleNamespace

import pytest

from agent.harness.research_timeline import ResearchTimeline


def timeline(tmp_path):
    folder = tmp_path / "episode"
    folder.mkdir()
    value = ResearchTimeline(folder, 1, tmp_path / "workspace")
    value.phase("MLS")
    return value, folder


def test_interleaved_calls_keep_their_own_progress(tmp_path):
    value, folder = timeline(tmp_path)
    value.tool({"tool": "train_protein_model", "status": "running", "call_id": "a",
                "job_id": "job-a", "progress": {"epoch": 2}})
    value.tool({"tool": "train_protein_model", "status": "running", "call_id": "b",
                "job_id": "job-b", "progress": {"epoch": 9}})
    value.tool({"tool": "train_protein_model", "status": "completed", "call_id": "a"})
    rows = json.loads((folder / "timeline.json").read_text())
    assert len(rows) == 2
    assert rows[0]["status"] == "completed"
    assert rows[0]["job_id"] == "job-a" and rows[0]["progress"] == {"epoch": 2}
    assert rows[1]["status"] == "running" and rows[1]["job_id"] == "job-b"


def test_separate_calls_do_not_inherit_errors_or_progress(tmp_path):
    value, folder = timeline(tmp_path)
    value.tool({"tool": "train_research_model", "status": "failed", "call_id": "a",
                "error_type": "ValueError", "progress": {"rows": 40}})
    value.tool({"tool": "train_research_model", "status": "completed", "call_id": "b"})
    rows = json.loads((folder / "timeline.json").read_text())
    assert len(rows) == 2
    assert rows[0]["error_type"] == "ValueError"
    assert rows[1]["status"] == "completed"
    assert "error_type" not in rows[1] and "progress" not in rows[1]


def test_reused_legacy_id_starts_a_new_operation(tmp_path):
    value, folder = timeline(tmp_path)
    for tool in ("read_session_file", "write_session_file", "write_session_file"):
        value.tool({"tool": tool, "status": "running", "call_id": "1"})
        value.tool({"tool": tool, "status": "completed", "call_id": "1"})
    rows = json.loads((folder / "timeline.json").read_text())
    assert [row["tool"] for row in rows] == [
        "read_session_file", "write_session_file", "write_session_file"]
    assert all(row["status"] == "completed" for row in rows)


def test_legacy_call_completion_retains_only_current_progress(tmp_path):
    value, folder = timeline(tmp_path)
    for n in (1, 2):
        value.tool({"tool": "train_protein_model", "status": "running", "progress": {"epoch": n}})
        value.tool({"tool": "train_protein_model", "status": "completed"})
    rows = json.loads((folder / "timeline.json").read_text())
    assert len(rows) == 2
    assert [row["progress"]["epoch"] for row in rows] == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code", "deepseek-harness", "codex", "claude-code"])
async def test_native_retries_use_distinct_progress_ids(tmp_path, monkeypatch, harness):
    import httpx
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile, Budget, ModelBinding
    from agent.harness.research_workspace import ResearchWorkspace

    work = tmp_path / "workspace"
    work.mkdir()
    (work / "notes.txt").write_text("Public fixture notes.")
    progress, episode = timeline(tmp_path)
    service = ResearchWorkspace(work, episode / "tools", {"steps": []})
    events = []

    def sink(event):
        events.append(event)
        if event.get("kind") == "research_tool_status":
            progress.tool(event)

    async def adapter(ctx):
        connection = json.loads((ctx.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            async def call(name, arguments):
                response = await client.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
                return response.json()["result"]
            assert not (await call("read_session_file", {"path": "notes.txt"}))["isError"]
            await call("describe_tool", {"name": "read_session_file"})
            failed = await call("call_tool", {"name": "read_session_file", "arguments": {"path": "missing.txt"}})
            assert failed["isError"]
        return {"status": "completed", "text": "Read public fixture notes."}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=3))
    for attempt in range(2):
        control = episode / ("attempt-" + str(attempt))
        control.mkdir()
        await worker.run({"profile": profile.public(), "workspace": str(work), "control": str(control),
                          "session_id": "fixture", "runtimes": {}, "base_url": "http://127.0.0.1:1",
                          "token": "fixture", "text": "Read the provided notes."},
                         tools=service.tools(), event_sink=sink)
    statuses = [e for e in events if e.get("kind") == "research_tool_status"]
    assert [e["status"] for e in statuses] == ["running", "completed", "running", "failed"] * 2
    assert {e["tool"] for e in statuses} == {"read_session_file"}
    ids = [e["call_id"] for e in statuses]
    assert all(ids[i] == ids[i + 1] for i in range(0, 8, 2))
    assert len(set(ids)) == 4
    rows = json.loads((episode / "timeline.json").read_text())
    assert [row["status"] for row in rows] == ["completed", "failed"] * 2
