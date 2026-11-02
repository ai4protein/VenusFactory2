import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile, Task, gateway, host


@pytest.mark.asyncio
@pytest.mark.parametrize("modify_input", [False, True])
async def test_cancel_preserves_host_record_and_propagates(tmp_path, monkeypatch, modify_input):
    started, cleaned = asyncio.Event(), asyncio.Event()
    work, cap = tmp_path / "work", tmp_path / "cap"
    work.mkdir()
    cap.mkdir()
    (work / "input.json").write_text("{}")
    (cap / "task.json").write_text("{}")
    task = Task(
        work, cap, tmp_path / "execution", "fixture", "fixture", frozenset({"inspect_model_code_task"}), phase="develop"
    )
    profile = Profile(
        "science_expert", "langgraph", ModelBinding("test", "fixture", "https://example.org/v1"), Budget()
    )

    class OfflineGateway:
        def __init__(self, **kwargs):
            self.path = kwargs["output"]

        async def __aenter__(self):
            self.path.write_text("")
            return self

        async def __aexit__(self, *args):
            cleaned.set()

    class Adapter:
        async def run(self, context):
            (context.task.output / "events.jsonl").write_text("")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                if modify_input:
                    (work / "input.json").write_text('{"modified":true}')

    monkeypatch.setattr(gateway, "EvidenceGateway", OfflineGateway)
    registry = SimpleNamespace(resolve=lambda *args: (Adapter(), {"fixture": True}))
    pending = asyncio.create_task(host.execute(profile, task, {}, registry=registry, credential="synthetic-no-network"))
    await asyncio.wait_for(started.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    record = json.loads((task.output / "run.json").read_text())
    lock = json.loads((task.output / "profile.lock.json").read_text())
    assert cleaned.is_set()
    assert record["status"] == ("binding_modified" if modify_input else "interrupted")
    assert record["error_type"] == "CancelledError" and record["cancellation_finalized"] is True
    assert record["profile_sha256"] == lock["sha256"]
    assert record["llm_calls"] == record["tool_calls"] == 0
    assert record["phase"] == "develop" and record["host_wall_seconds"] >= 0
