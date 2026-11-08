"""Verified preparation metadata persists; measured observations stay out of context."""
import json
from pathlib import Path

import pytest

from agent.harness.research_data_memory import prepared_dataset_context
from agent.harness.research_workspace import ResearchWorkspace


async def prepared_at(tmp_path):
    workspace = tmp_path / "workspace"
    episodes = tmp_path / "episodes"
    service = ResearchWorkspace(workspace, episodes / "cycle_000" / "tools", {"steps": []},
                                development_root=tmp_path / "development")
    csv = "id,group,feature,measurement\n" + "".join(
        f"sample{i},group{i},feature-{i},{i / 17}\n" for i in range(40))
    await service.write_research_file("inputs/data.csv", csv)
    result = await service.prepare_research_dataset(
        "inputs/data.csv", "measurement", ["feature"], "group",
        "Synthetic software fixture", "Independent groups", id_column="id")
    return service, episodes, result


@pytest.mark.asyncio
async def test_prepared_data_context_contains_verified_columns_counts_and_paths(tmp_path):
    service, episodes, result = await prepared_at(tmp_path)
    saved = prepared_dataset_context(service.workspace, episodes)
    assert saved["status"] == "verified"
    assert saved["evaluation_file"] == result["evaluation_file"]
    assert saved["columns"] == {"target_column": "measurement", "input_columns": ["feature"],
                                 "group_column": "group", "id_column": "id"}
    assert sum(split["rows"] for split in saved["splits"].values()) == 40
    assert "feature-0" not in json.dumps(saved)
    assert "sample0" not in json.dumps(saved)
    assert "targets" not in saved and "training" not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["source", "evaluation", "symlink", "receipt", "hash"])
async def test_changed_or_untrusted_preparations_are_not_reused(tmp_path, damage):
    service, episodes, result = await prepared_at(tmp_path)
    if damage == "source":
        (service.workspace / "inputs/data.csv").write_text("changed")
    elif damage == "evaluation":
        (service.workspace / result["evaluation_file"]).write_text("{}")
    elif damage == "symlink":
        path = service.workspace / "inputs/data.csv"
        path.unlink()
        outside = tmp_path / "other-session.csv"
        outside.write_text("private")
        path.symlink_to(outside)
    else:
        receipt = episodes / "cycle_000/tools/receipts.json"
        rows = json.loads(receipt.read_text())
        for row in rows:
            if row["tool"] == "prepare_research_dataset":
                if damage == "receipt":
                    row["tool"] = "write_research_file"
                else:
                    row["result"]["table_sha256"] = "wrong"
        receipt.write_text(json.dumps(rows))
    assert prepared_dataset_context(service.workspace, episodes) is None


def test_no_host_receipt_means_no_verified_dataset(tmp_path):
    assert prepared_dataset_context(tmp_path, tmp_path / "episodes") is None


@pytest.mark.asyncio
async def test_next_scientist_iteration_reuses_host_verified_preparation(tmp_path, monkeypatch):
    from agent.harness import Profile
    from agent.harness import autonomous_research as module

    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"

        def __init__(self, **kwargs):
            Path(kwargs["output"]).touch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(module, "EvidenceGateway", Relay)
    monkeypatch.setattr(module, "audit_requests", lambda path: {"passed": True, "model_calls": 4})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120,
                   "output_tokens": 4096, "thinking": "disabled"}})
    store = module.create(tmp_path / "goal", profile, "Develop a predictor from the measured data",
                          {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240})
    # A real upload is protected by the host before any Agent execution.
    from agent.harness.research_data_store import store_for_goal
    import hashlib
    csv = "id,group,feature,measurement\n" + "".join(
        f"sample{i},group{i},{i},{i / 17}\n" for i in range(40))
    raw = csv.encode()
    item = {"path": "inputs/data.csv", "name": "data.csv",
            "sha256": hashlib.sha256(raw).hexdigest()}
    store_for_goal(store.root).ingest(item["path"], raw, item["sha256"])
    store.mutate(lambda state: state["progress"].update(inputs=[item]))
    contexts = []
    requests = []
    research_plan = {"summary": "Prepare observations and fit a predictor",
                     "deliverable": "model", "needs_input": [], "steps": [
        {"id": "data", "title": "Prepare measured data", "acceptance": "Verified group splits", "verification": "data"},
        {"id": "model", "title": "Fit predictor", "acceptance": "Measured trained model", "verification": "model"}]}

    async def consult(role, context):
        if role == "PI":
            contexts.append(context["model_development"].get("prepared_dataset"))
            return "Develop a reusable predictor."
        if role == "CB":
            return json.dumps(research_plan)
        return json.dumps({"decision": "needs_revision", "summary": "Implement the model next", "issues": []})

    async def executor(request, service, events):
        requests.append(json.loads(request["text"]))
        if len(requests) == 1:
            prepared = await service.prepare_research_dataset(
                "inputs/data.csv", "measurement", ["feature"], "group",
                "Synthetic software fixture", "Independent groups", id_column="id")
            await service.submit_research_checkpoint(
                "Prepared measured observations", ["data"], [prepared["evaluation_file"]], [])
        events.extend([{"kind": "tool_call_start"},
                       {"kind": "completed", "text": "Prepared data; training remains incomplete.",
                        "event_audit": {"passed": True}}])

    runner = module.AutonomousResearch(store, "fixture", {}, consult=consult, executor=executor, tools=[])
    result = await runner.run()
    assert len(result["cycles"]) == 2
    assert contexts[0] is None and contexts[1]["status"] == "verified"
    assert requests[1]["model_development"]["prepared_dataset"]["columns"]["target_column"] == "measurement"
    assert requests[1]["model_development"]["selected_tools"] == ["get_research_baseline", "train_research_model"]
    assert result["status"] != "completed"
