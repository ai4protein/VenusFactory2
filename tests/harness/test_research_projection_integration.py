"""Native HTTP wiring keeps original context and protects projected references."""
from __future__ import annotations

import copy
import hashlib
import json
from functools import wraps
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import web_native_worker
from agent.harness.web_research_role_context import project_role_context
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_workspace import ResearchWorkspace


def research_context():
    plan = {"summary": "Compare observed enzyme-affinity models", "deliverable": "model", "needs_input": [],
            "steps": [{"id": f"step{i}", "title": f"Milestone {i}",
                       "acceptance": f"Keep criterion {i}: original physical target, biological partition and measured comparison",
                       "approach": "Current method with preserved assay units. " * 50,
                       "verification": "model", "status": "pending", "evidence": []} for i in range(3)]}
    return {"objective": "Develop a predictor using the supplied physical measurements.",
            "criteria": ["research_plan_executed", "research_outputs_reviewed", "capability_admission_resolved"],
            "PI": "Revise the measured baseline while retaining the endpoint and biological split.",
            "plan": plan,
            "progress": {"plan": copy.deepcopy(plan), "execution_feedback": {"pending": [{"error": "Missing module-level predict"}]},
                         "sc_revision": {"issues": ["No measured revision"], "measurement_state": {"baseline_measured": True}},
                         "user_updates": ["Do not alter the target"], "future_required_feedback": {"pending": True}},
            "resources": {"budget": {"model_calls": 64, "tool_calls": 32, "wall_seconds": 600}},
            "model_development": {
                "prepared_dataset": {"evaluation_file": "checked/evaluation.json", "task": "regression",
                                     "model_interface": {"input_fields": {"sequence": "str"}, "target_unit": "nM"}},
                "repair_source": {"code": "def fit(training, seed): return {}\n", "sha256": "c" * 64,
                                  "redacted": False, "source_check": {"valid": False}, "error": "Missing predict"}},
            "previous_episodes": [{"summary": f"Historical episode {i}: " + "Actual measured feedback 独立. " * 100,
                                   "status": "needs_revision"} for i in range(12)]}


@pytest.mark.parametrize("objective_bytes", [0, 300000])
@pytest.mark.asyncio
async def test_native_projection_reads_immutable_references_without_changing_request_or_budget(tmp_path, monkeypatch, objective_bytes):
    for key, value in {"VENUS_RUNTIME_ROOT": tmp_path / "runtime", "VENUS_JOB_QUEUE_ROOT": tmp_path / "queue",
                       "VENUS_USAGE_DB": tmp_path / "usage.sqlite", "VENUS_ACCESS_DB": tmp_path / "access.sqlite",
                       "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "WEBUI_V2_MODE": "local", "VENUS_SCIENTIST_ONLINE": "0",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        monkeypatch.setenv(key, str(value))
    for name in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN",
                 "VENUS_CAPABILITY_POLICY_SNAPSHOT", "VENUS_MODEL_RUNTIMES_CONFIG", "VENUS_SCIENTIST_PBS_RECORD"):
        monkeypatch.delenv(name, raising=False)
    source = research_context()
    if objective_bytes:
        source["objective"] += "x" * objective_bytes + "FINAL USER REQUIREMENT"
    role_input = project_role_context("MLS", source)
    workspace, control = tmp_path / "workspace", tmp_path / "control"
    service = ResearchWorkspace(workspace, control / "tools", source["plan"])
    race = {"path": None, "triggered": False}
    original_read = service.read_research_file

    @wraps(original_read)
    async def read_with_concurrent_replacement(path, offset_bytes=0, max_bytes=20000, view="auto", select=None):
        if race["path"] == path:
            # The native handler has already checked the reference hash. Replace
            # bytes immediately before the real descriptor-backed reader opens it.
            (workspace / path).write_bytes(b"race-replacement-must-not-reach-model")
            race.update(path=None, triggered=True)
        return await original_read(path, offset_bytes, max_bytes, view, select=select)

    monkeypatch.setattr(service, "read_research_file", read_with_concurrent_replacement)
    profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(model_calls=7, tool_calls=32, wall_seconds=90))
    request = {"session_id": "projection-fixture", "workspace": str(workspace), "control": str(control),
               "profile": profile.public(), "budget": copy.deepcopy(profile.public()["budget"]),
               "research_allowances": {"discovery": 3, "checkpoint": 1}, "messages": [],
               "runtimes": {"fixture": "transport-runtime-marker"}, "base_url": "http://127.0.0.1:1",
               "token": "synthetic-relay-must-not-be-written", "text": json.dumps(source, ensure_ascii=False)}
    original = copy.deepcopy(request)
    observed, calls = [], []

    async def adapter(context):
        projected = json.loads(context.task.prompt.rsplit("## Current user request\n", 1)[1])
        assert request == original
        if objective_bytes:
            assert projected["objective"].endswith("FINAL USER REQUIREMENT")
            assert len(projected["objective"].encode()) > 262144
        assert projected["role_context"]["policy"] == "scientist_role_context.v2"
        assert projected["plan"] == role_input["plan"]  # No current-step guess: retain every approach.
        assert "PI" not in projected and "previous_episodes" not in projected
        for key in ("objective", "criteria", "model_development", "resources"):
            assert projected[key] == source[key]
        for key, value in role_input["progress"].items():
            if key != "plan":
                assert projected["progress"][key] == value
        assert context.profile.budget.model_calls == profile.budget.model_calls
        assert context.profile.budget.wall_seconds == profile.budget.wall_seconds
        assert context.profile.budget.output_tokens == profile.budget.output_tokens
        # Existing transport accounting adds declared discovery/checkpoint limits;
        # projection must not replenish or replace the original execution budget.
        assert context.profile.budget.tool_calls == profile.budget.tool_calls + 3 + 1
        references = projected["context_references"]["documents"]
        assert references
        parts = [part for document in references for part in document["parts"]]
        assert all(part["bytes"] <= 16000 for part in parts)
        connection = json.loads((context.task.capability / "connection.json").read_text())
        read_content = {}
        async with httpx.AsyncClient(trust_env=False, timeout=30) as http:
            async def call(name, arguments):
                response = await http.post(connection["url"],
                    headers={"Authorization": "Bearer " + connection["token"]},
                    json={"jsonrpc": "2.0", "id": len(calls) + 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
                response.raise_for_status()
                body = response.json()
                assert "error" not in body, body
                calls.append(name)
                return body["result"]

            for part in parts:
                result = await call("read_session_file", {"path": part["path"], "view": "text"})
                assert result["isError"] is False, result
                value = json.loads(result["content"][0]["text"])
                assert value["next_offset_bytes"] is None and value["truncated"] is False
                assert value["file_sha256"] == value["sha256"] == part["sha256"]
                raw = value["text"].encode()
                assert len(raw) == part["bytes"] and hashlib.sha256(raw).hexdigest() == part["sha256"]
                read_content[part["path"]] = value["text"]
                assert request["token"] not in value["text"]
                assert "transport-runtime-marker" not in value["text"]
            recovered = {}
            for document in references:
                raw = "".join(read_content[part["path"]] for part in document["parts"]).encode()
                assert hashlib.sha256(raw).hexdigest() == document["sha256"]
                recovered[document["name"]] = json.loads(raw)
            assert recovered["plan"] == role_input["plan"]
            assert "previous-episodes" not in recovered  # Historical role prose is not an execution handoff.
            target = parts[0]
            path = target["path"]
            aliases = [path, "/workspace/" + path, "./" + path,
                       path.replace("/", "//", 1), path.replace("/", "/./", 1)]
            original_bytes = (workspace / path).read_bytes()
            window = await call("read_session_file", {"path": path, "offset_bytes": 1, "max_bytes": 256, "view": "text"})
            assert window["isError"] is False
            window_value = json.loads(window["content"][0]["text"])
            assert window_value["truncated"] is True
            assert window_value["file_sha256"] == target["sha256"]
            assert window_value["slice_sha256"] != target["sha256"]
            for alias in aliases:
                denied = await call("write_session_file", {"path": alias, "content": "replacement"})
                assert denied["isError"] is True and "read-only" in denied["content"][0]["text"], alias
                assert (workspace / path).read_bytes() == original_bytes
                denied = await call("edit_session_file", {"path": alias, "old_text": read_content[path],
                    "new_text": "replacement", "expected_sha256": target["sha256"]})
                assert denied["isError"] is True and "read-only" in denied["content"][0]["text"], alias
                assert (workspace / path).read_bytes() == original_bytes
            race["path"] = path
            raced = await call("read_session_file", {"path": path, "view": "text"})
            assert race["triggered"] is True
            assert raced["isError"] is True and "reference changed" in raced["content"][0]["text"]
            assert "race-replacement-must-not-reach-model" not in json.dumps(raced)
            assert (workspace / path).read_bytes() == b"race-replacement-must-not-reach-model"
            (workspace / path).write_bytes(original_bytes)
            # Simulate a separate sandbox process modifying its writable mounted
            # workspace. Real RPC reads must verify stored bytes, not cached text.
            (workspace / path).write_bytes(b"tampered-context-marker")
            for alias in aliases:
                denied = await call("read_session_file", {"path": alias, "view": "text"})
                assert denied["isError"] is True and "reference changed" in denied["content"][0]["text"], alias
                assert "tampered-context-marker" not in json.dumps(denied)
            normal = await call("write_session_file", {"path": "notes.md", "content": "Ordinary output is still writable."})
            assert normal["isError"] is False
            ordinary_read = await call("read_session_file", {"path": "notes.md"})
            assert ordinary_read["isError"] is False
            assert json.loads(ordinary_read["content"][0]["text"])["text"] == "Ordinary output is still writable."
        assert len(calls) <= profile.budget.tool_calls
        return {"status": "completed", "text": "Read complete scientific references and preserved the original contracts."}

    monkeypatch.setattr(web_native_worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    await web_native_worker.run(request, tools=service.tools(), event_sink=observed.append)
    assert request == original and json.loads(request["text"]) == source
    receipt = json.loads((control / "native/context-projection.json").read_text())
    assert receipt["original_sha256"] == hashlib.sha256(original["text"].encode()).hexdigest()
    assert receipt["reference_files"]
    assert request["token"] not in json.dumps(receipt)
    completed = next(row for row in observed if row["kind"] == "completed")
    assert completed["tool_calls"] == len(calls)
    assert completed["discovery_calls"] == completed["checkpoint_calls"] == 0
    assert completed["messages"][0]["content"] == original["text"]
    assert (workspace / "notes.md").read_text() == "Ordinary output is still writable."
