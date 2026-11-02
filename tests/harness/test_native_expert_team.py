"""Native executor portability keeps four roles, independent gates and total caps."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness import native_expert_team as team
from agent.harness.host import RuntimeContext
from agent.rsi.model_gateway import GatewayPolicy


@pytest.fixture
def context(tmp_path):
    work, capability, output = (tmp_path / name for name in ("work", "capability", "output"))
    for p in (work, capability, output):
        p.mkdir()
    (capability / "scientific_task.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "kind": "reproduction",
                "context_tool": "protein_probe",
                "artifact_file": "probe.json",
                "required_tools": ["protein_probe"],
                "completion_repair_limit": 0,
            }
        )
    )
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test", "test", "https://example.org/v1"),
        Budget(wall_seconds=10, model_calls=12, tool_calls=2),
    )
    task = Task(
        work,
        capability,
        output,
        "Measure ACDEFGHIKC",
        "Use observed results",
        frozenset({"protein_probe"}),
        phase="chat",
    )
    return RuntimeContext(profile, task, SimpleNamespace(calls=0), {})


def install(
    monkeypatch,
    *,
    decision="ready_for_host_validation",
    native_status="completed",
    artifact=True,
    pair=True,
    native_calls=2,
    tool="protein_probe",
):
    calls = []

    async def role(ctx, role, evidence, folder):
        calls.append(role)
        assert evidence["artifact_policy"]["owner"] == "host-mediated scientific tools"
        assert "no manual filesystem" in evidence["artifact_policy"]["mutation"]
        ctx.relay.calls += 1
        if role == "SC":
            assert "execution_evidence" in evidence
            return json.dumps({"decision": decision, "summary": "Review complete", "issues": []})
        return "Bounded task direction and plan"

    class Native:
        async def run(self, ctx):
            calls.append("MLS")
            assert ctx.profile.mode == "science_agent"
            assert ctx.task.phase == "probe"
            assert "Never manually write" in ctx.task.instructions
            assert "prior role proposes a different format" in ctx.task.instructions
            assert ctx.profile.model.model == "test"
            ctx.relay.calls += native_calls
            events = [
                {
                    "kind": "tool_call_start",
                    "tool_call_id": "call-1",
                    "tool_name": "mcp__venusfactory__" + tool,
                    "args": {"sequence": "ACDEFGHIKC"},
                }
            ]
            if pair:
                events.append(
                    {
                        "kind": "tool_result",
                        "tool_call_id": "call-1",
                        "is_error": False,
                        "output": '{"length": 10, "cysteines": 2}',
                    }
                )
            (ctx.task.output / "events.jsonl").write_text("\n".join(json.dumps(x) for x in events) + "\n")
            if artifact:
                (ctx.task.workspace / "probe.json").write_text('{"length":10,"cysteines":2}')
            return {"status": native_status, "session_id": "native-session-1", "tool_calls": 1, "text": "Measured"}

    monkeypatch.setattr(team, "role_completion", role)
    monkeypatch.setattr(team, "native_adapter", lambda native, runtimes: Native())
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", list(team.NATIVE_EXPERT_HARNESSES))
async def test_four_roles_use_selected_native_backend_and_shared_gateway(context, monkeypatch, harness):
    context.profile = replace(context.profile, harness=harness)
    calls = install(monkeypatch)
    result = await team.NativeExpertTeamAdapter({}).run(context)
    assert result["status"] == "completed"
    assert calls == result["role_ids"] == ["PI", "CB", "MLS", "SC"]
    assert result["native_executor"] == team.NATIVE_EXPERT_HARNESSES[harness]
    assert result["llm_calls"] == 5 and result["tool_calls"] == 1
    assert result["four_role_path_completed"] and result["finalizer_succeeded"]
    history = json.loads((context.task.output / "role_history.json").read_text())
    assert history[2]["native_session_id"] == "native-session-1"
    assert json.loads((context.task.output / "roles/MLS/native_result.json").read_text())["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"decision": "needs_revision"}, "incomplete"),
        ({"artifact": False}, "incomplete"),
        ({"native_status": "runtime_error"}, "execution_failed"),
        ({"pair": False}, "execution_failed"),
        ({"tool": "unadmitted"}, "execution_failed"),
    ],
)
async def test_no_completion_from_critic_or_artifact_alone(context, monkeypatch, kwargs, status):
    install(monkeypatch, **kwargs)
    result = await team.NativeExpertTeamAdapter({}).run(context)
    assert result["status"] == status
    assert result["status"] != "completed"
    assert (context.task.output / "role_history.json").is_file()


@pytest.mark.asyncio
async def test_exhaustion_does_not_grant_an_extra_critic_call(context, monkeypatch):
    calls = install(monkeypatch, native_calls=10)
    result = await team.NativeExpertTeamAdapter({}).run(context)
    assert calls == ["PI", "CB", "MLS"]
    assert result["status"] == "timeout" and result["llm_calls"] == 12
    assert not result["four_role_path_completed"]


@pytest.mark.asyncio
async def test_old_artifact_cannot_complete_a_new_episode(context, monkeypatch):
    install(monkeypatch)
    (context.task.workspace / "probe.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        await team.NativeExpertTeamAdapter({}).run(context)
    assert context.relay.calls == 0


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"decision":"ready_for_host_validation","summary":"","issues":[]}',
        '{"decision":"approved","summary":"ok","issues":[]}',
        "approved",
    ],
)
def test_critic_requires_a_complete_typed_decision(text):
    with pytest.raises((ValueError, TypeError)):
        team.critic_decision(text)


def test_native_expert_gateway_filters_shell_and_internal_tools():
    mode = team.scientific_tool_exposure("science_expert", "expert-kimi-code")
    assert mode == "science_agent"
    policy = GatewayPolicy(model="test", mode=mode, scientific_tools={"protein_probe"})
    payload, _ = policy.apply(
        {
            "model": "test",
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
                for name in ["Shell", "WriteFile", "mcp__venusfactory__protein_probe"]
            ],
        }
    )
    assert [t["function"]["name"] for t in payload["tools"]] == ["mcp__venusfactory__protein_probe"]
    assert team.scientific_tool_exposure("science_expert", "langgraph") == "science_expert"


def test_explicit_registry_profiles_preserve_model_and_reject_agent_use(context, tmp_path):
    registry = builtin_registry()
    runtime = tmp_path / "runtime"
    runtime.write_text("fixture")
    runtimes = {key: str(runtime) for key in ("kimi_bin", "dsh_bin", "codex_bin", "claude_bin")}
    for harness in team.NATIVE_EXPERT_HARNESSES:
        profile = replace(context.profile, harness=harness)
        if harness == "expert-codex":
            profile = replace(
                profile,
                model=replace(profile.model, protocol="responses"),
                budget=replace(profile.budget, thinking=None, temperature=None, reasoning_effort="low"),
            )
        if harness == "expert-claude-code":
            profile = replace(profile, budget=replace(profile.budget, thinking="disabled"), request_recovery="none")
        assert registry.preflight(profile, runtimes)["ready"]
        assert not registry.preflight(replace(profile, mode="science_agent"), runtimes)["ready"]
        assert not registry.preflight(profile, {})["ready"]
        assert profile.model.model == context.profile.model.model
        assert profile.model.base_url == context.profile.model.base_url
