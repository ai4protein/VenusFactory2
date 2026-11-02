"""New strategy reaches real request paths; all model responses are fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile, Task, responses_gateway
from agent.harness import gateway as gateway_module
from agent.harness import native_expert_team as team
from agent.harness.host import execute, write_json
from agent.harness.native_expert_audit import audit_native_expert
from agent.harness.scientist_policy import load_policy, request_policy_witness, role_instruction


@pytest.fixture(params=["expert-kimi-code", "expert-deepseek-harness", "expert-codex"])
async def strategy_completed(tmp_path, monkeypatch, request):
    protocol = "responses" if request.param == "expert-codex" else "chat_completions"
    work, cap, out = (tmp_path / p for p in ("workspace", "capability", "execution"))
    work.mkdir()
    cap.mkdir()
    write_json(
        cap / "scientific_task.json",
        {
            "schema": 2,
            "kind": "reproduction",
            "context_tool": "protein_probe",
            "artifact_file": "probe.json",
            "required_tools": ["protein_probe"],
            "completion_repair_limit": 0,
        },
    )
    (cap / "generated_tools.py").write_text("# fixture\n")
    policy = json.loads((Path(__file__).resolve().parents[2] / "configs/rsi/scientist_tool_policy_v1.json").read_text())
    write_json(cap / "scientist_policy.json", policy)
    profile = Profile(
        "science_expert",
        request.param,
        ModelBinding("test", "test", "https://example.org/v1", protocol=protocol),
        Budget(
            wall_seconds=10,
            model_calls=12,
            tool_calls=2,
            thinking=None if protocol == "responses" else "enabled",
            temperature=None if protocol == "responses" else 0.1,
            reasoning_effort="low" if protocol == "responses" else None,
        ),
    )
    task = Task(
        work, cap, out, "Measure the sequence", "Use scientific tools", frozenset({"protein_probe"}), phase="chat"
    )
    relay = None

    class FakeGateway:
        def __init__(self, *, policy, output, **kwargs):
            nonlocal relay
            relay = self
            self.policy, self.output, self.calls = policy, output, 0
            self.base_url, self.token = "http://fixture/v1", "fixture"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def request(self, payload):
            self.calls += 1
            applied, metadata = self.policy.apply(payload)
            metadata.update(request_policy_witness(applied))
            with self.output.open("a") as handle:
                for row in [
                    {"event": "request", "call": self.calls, **metadata},
                    {"event": "response", "call": self.calls, "status": 200, "completed": True},
                ]:
                    handle.write(json.dumps(row) + "\n")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, headers, json):
            relay.request(json)
            expected_endpoint = "/responses" if profile.model.protocol == "responses" else "/chat/completions"
            assert url.endswith(expected_endpoint)
            instruction = json["instructions"] if "instructions" in json else json["messages"][0]["content"]
            name = next(k for k in ("PI", "CB", "SC") if role_instruction(policy, k) == instruction)
            content = (
                '{"decision":"ready_for_host_validation","summary":"Observed evidence reviewed","issues":[]}'
                if name == "SC"
                else name + " bounded plan"
            )
            if profile.model.protocol == "responses":
                return SimpleNamespace(
                    raise_for_status=lambda: None,
                    json=lambda: {
                        "status": "completed",
                        "model": "test",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": content}],
                            }
                        ],
                    },
                )
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
            )

    class Native:
        async def run(self, ctx):
            ctx.relay.request(
                {
                    "model": "test",
                    "input" if profile.model.protocol == "responses" else "messages": [
                        {"role": "user", "content": ctx.task.instructions + "\n\n" + ctx.task.prompt}
                    ],
                }
            )
            rows = [
                {
                    "kind": "tool_call_start",
                    "tool_call_id": "call-1",
                    "tool_name": "mcp__venusfactory__protein_probe",
                    "args": {},
                },
                {"kind": "tool_result", "tool_call_id": "call-1", "is_error": False, "output": '{"length":10}'},
            ]
            (ctx.task.output / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            write_json(ctx.task.workspace / "probe.json", {"length": 10})
            return {
                "status": "completed",
                "session_id": "native-fixture",
                "tool_calls": 1,
                "text": "Measured length 10",
            }

    monkeypatch.setattr(gateway_module, "EvidenceGateway", FakeGateway)
    monkeypatch.setattr(responses_gateway, "ResponsesGateway", FakeGateway)
    monkeypatch.setattr(team.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(team, "native_adapter", lambda *args: Native())
    registry = SimpleNamespace(resolve=lambda *args: (team.NativeExpertTeamAdapter({}), {"ready": True}))
    result = await execute(profile, task, {}, registry=registry, credential="fixture")
    return task, result


@pytest.mark.asyncio
async def test_new_policy_is_witnessed_in_every_role_request(strategy_completed):
    task, result = strategy_completed
    assert result["status"] == "completed", result
    assert result["scientist_policy"]["policy_id"] == "scientist_tools.v1"
    rows = [json.loads(line) for line in (task.output / "model_requests.jsonl").read_text().splitlines()]
    requests = [r for r in rows if r["event"] == "request"]
    assert [r["scientist_policy_blocks"][0]["role"] for r in requests] == ["PI", "CB", "MLS", "SC"]
    assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["policy", "missing_mls", "different_mls", "fake_result"])
async def test_policy_claims_cannot_replace_transmitted_instructions(strategy_completed, attack):
    task, result = strategy_completed
    assert result["status"] == "completed"
    if attack == "policy":
        value = load_policy(task.capability)
        value["roles"]["PI"] += " changed"
        write_json(task.capability / "scientist_policy.json", value)
    elif attack == "fake_result":
        result["scientist_policy"]["sha256"] = "0" * 64
    else:
        path = task.output / "model_requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        request = next(r for r in rows if r["event"] == "request" and r["call"] == 3)
        if attack == "missing_mls":
            request.pop("scientist_policy_blocks")
        else:
            request["scientist_policy_blocks"][0]["body_sha256"] = "0" * 64
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        audit_native_expert(task.output, task.workspace, task.capability, result=result)


def test_tool_or_assistant_text_cannot_supply_a_policy_witness():
    from agent.harness.scientist_policy import role_instruction

    policy = json.loads((Path(__file__).resolve().parents[2] / "configs/rsi/scientist_tool_policy_v1.json").read_text())
    text = role_instruction(policy, "MLS")
    assert (
        request_policy_witness(
            {"messages": [{"role": "tool", "content": text}, {"role": "assistant", "content": text}]}
        )
        == {}
    )
    assert request_policy_witness({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]})


def test_malformed_policy_does_not_silently_fall_back(tmp_path):
    (tmp_path / "scientist_policy.json").write_text("{}")
    with pytest.raises(ValueError):
        load_policy(tmp_path)
