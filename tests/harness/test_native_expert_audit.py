"""A fake provider exercises the real coordinator and host verification gate."""

import importlib
import json
from json import dumps
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile, Task, responses_gateway
from agent.harness import gateway as gateway_module
from agent.harness import native_expert_team as team
from agent.harness.host import execute, write_json
from agent.harness.native_expert_audit import audit_native_expert


@pytest.fixture
def historical_revision(request):
    return getattr(request, "param", None)


@pytest.fixture
def critic_prefix(request):
    return getattr(request, "param", "")


@pytest.fixture
def critic_outputs(request):
    return getattr(request, "param", None)


@pytest.fixture
def role_budget(request):
    return getattr(request, "param", 12)


@pytest.fixture
def native_artifact(request):
    return getattr(request, "param", True)


@pytest.fixture(params=["chat_completions", "responses"])
async def completed(
    tmp_path, monkeypatch, request, historical_revision, critic_prefix, critic_outputs, role_budget, native_artifact
):
    coordinator = team
    if historical_revision:
        # Execute the actual pinned coordinator and record the sources it used.
        # Historical tests must not relabel current-policy requests as old ones.
        from agent.harness import host

        if historical_revision == "pre_isolation_v4":
            from types import ModuleType
            fixture = Path(__file__).parent / "fixtures/native_expert_team_v4.py"
            coordinator = ModuleType("agent.harness._v4_replay")
            coordinator.__package__ = "agent.harness"
            coordinator.__file__ = str(fixture)
            exec(compile(fixture.read_text(), str(fixture), "exec"), coordinator.__dict__)
        else:
            coordinator = importlib.import_module("agent.harness." + historical_revision + ".harness.native_expert_team")
        source_root = Path(coordinator.__file__).parent
        original_hash = host.file_hash

        def recorded_source(path):
            if historical_revision == "pre_isolation_v4" and path.parent == Path(host.__file__).parent:
                from agent.harness.native_expert_audit import LEGACY_V4_SOURCES
                if path.name in LEGACY_V4_SOURCES:
                    return LEGACY_V4_SOURCES[path.name]
            candidate = source_root / path.name
            if path.parent == Path(host.__file__).parent and candidate.is_file():
                return original_hash(candidate)
            return original_hash(path)

        monkeypatch.setattr(host, "file_hash", recorded_source)

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
    profile = Profile(
        "science_expert",
        "expert-codex" if request.param == "responses" else "expert-kimi-code",
        ModelBinding("test", "test", "https://example.org/v1", protocol=request.param),
        Budget(
            wall_seconds=10,
            model_calls=role_budget,
            tool_calls=2,
            thinking=None if request.param == "responses" else "enabled",
            temperature=None if request.param == "responses" else 0.1,
            reasoning_effort="low" if request.param == "responses" else None,
        ),
    )
    task = Task(
        work, cap, out, "Measure the sequence", "Use scientific tools", frozenset({"protein_probe"}), phase="chat"
    )
    relay = None
    supplied_critic_outputs = list(critic_outputs) if critic_outputs is not None else None

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
            _, metadata = self.policy.apply(payload)
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
            if not historical_revision:
                assert not {"conversation", "previous_response_id", "session_id"}.intersection(json)
                if "messages" in json:
                    assert [item["role"] for item in json["messages"]] == ["system", "user"]
                else:
                    assert [item["role"] for item in json["input"]] == ["user"]
                    assert json["store"] is False
            relay.request(json)
            expected_endpoint = "/responses" if profile.model.protocol == "responses" else "/chat/completions"
            assert url.endswith(expected_endpoint)
            instruction = json["instructions"] if "instructions" in json else json["messages"][0]["content"]
            name = next(k for k, v in coordinator.ROLE_INSTRUCTIONS.items() if v == instruction)
            content = (
                '{"decision":"ready_for_host_validation","summary":"Observed evidence reviewed","issues":[]}'
                if name == "SC"
                else name + " bounded plan"
            )
            if name != "SC" and not historical_revision:
                content = dumps({
                    "direction" if name == "PI" else "steps": ["Bounded proposal"],
                    "constraints": [], "unknowns": [],
                })
            if name == "SC":
                if supplied_critic_outputs is not None:
                    assert supplied_critic_outputs, "No unbudgeted third critic response is available"
                    content = supplied_critic_outputs.pop(0)
                else:
                    content = critic_prefix + content
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
                        {"role": "user", "content": ctx.task.prompt}
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
            if native_artifact:
                write_json(ctx.task.workspace / "probe.json", {"length": 10})
            return {
                "status": "completed",
                "session_id": "native-fixture",
                "tool_calls": 1,
                "text": "Measured length 10",
            }

    monkeypatch.setattr(gateway_module, "EvidenceGateway", FakeGateway)
    monkeypatch.setattr(responses_gateway, "ResponsesGateway", FakeGateway)
    monkeypatch.setattr(coordinator.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(coordinator, "native_adapter", lambda *args: Native())
    registry = SimpleNamespace(resolve=lambda *args: (coordinator.NativeExpertTeamAdapter({}), {"ready": True}))
    result = await execute(profile, task, {}, registry=registry, credential="fixture")
    return task, result


@pytest.mark.asyncio
async def test_host_verifies_roles_gateway_and_critic_artifact(completed):
    task, result = completed
    assert result["status"] == "completed"
    assert result["audit"]["native_expert"]["passed"]
    assert result["audit"]["native_expert"]["model_calls"] == 4
    assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "artifact",
        "critic_artifact",
        "phase_contract",
        "phase_handoff",
        "plan",
        "request",
        "native_session",
        "interval",
        "trace",
        "response",
        "capability",
        "old_artifact",
        "extra_call",
        "native_model",
        "native_tools",
        "native_sampling",
    ],
)
async def test_tampering_cannot_keep_completed_expert_evidence(completed, corruption):
    task, result = completed
    assert result["status"] == "completed"
    out = task.output
    if corruption == "artifact":
        write_json(task.workspace / "probe.json", {"length": 11})
    elif corruption in {"phase_contract", "phase_handoff"}:
        p = out / ("roles/PI/context.json" if corruption == "phase_contract" else "roles/SC/context.json")
        value = json.loads(p.read_text())
        value["episode_review_contract"]["deferred_host_decisions"] = []
        write_json(p, value)
    elif corruption in {"critic_artifact", "plan"}:
        p = out / "roles/SC/context.json"
        value = json.loads(p.read_text())
        if corruption == "critic_artifact":
            value["execution_evidence"]["artifact"]["content"] = {"length": 11}
        else:
            value["CB"] = "A different plan"
        write_json(p, value)
    elif corruption in {"native_session", "interval"}:
        p = out / "role_history.json"
        value = json.loads(p.read_text())
        value[2]["native_session_id" if corruption == "native_session" else "calls_before"] = (
            "another" if corruption == "native_session" else 1
        )
        write_json(p, value)
    elif corruption == "trace":
        (out / "roles/MLS/events.jsonl").write_text("{}\n")
    elif corruption == "response":
        (out / "roles/SC/response.txt").write_text('{"decision":"needs_revision","summary":"Missing work","issues":[]}')
    elif corruption == "capability":
        (task.capability / "generated_tools.py").write_text("# changed\n")
    elif corruption == "old_artifact":
        p = out / "profile.lock.json"
        value = json.loads(p.read_text())
        from agent.harness.contracts import digest, file_hash

        value["task"]["inputs"]["probe.json"] = file_hash(task.workspace / "probe.json")
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
        write_json(p, value)
        result["profile_sha256"] = value["sha256"]
    else:
        p = out / "model_requests.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        if corruption == "native_model":
            rows[4]["model"] = "unbound-model"
        elif corruption == "native_tools":
            rows[4]["tools_sent"] = ["Bash"]
        elif corruption == "native_sampling":
            row = rows[4]
            row["max_output_tokens" if row.get("protocol") == "responses" else "max_tokens"] = 999999
        elif corruption == "request":
            rows[0]["request_sha256"] = "0" * 64
        else:
            rows += [
                {"event": "request", "call": 5},
                {"event": "response", "call": 5, "status": 200, "completed": True},
            ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        audit_native_expert(out, task.workspace, task.capability, result=result)


@pytest.mark.asyncio
async def test_host_rejects_an_artifact_changed_after_review(completed, monkeypatch):
    # Reuse the fixture's fake transport, but make the trusted adapter return
    # evidence for a different artifact after the critic has finished.
    task, _ = completed
    from dataclasses import replace

    original = team.NativeExpertTeamAdapter.run

    async def changed(self, ctx):
        result = await original(self, ctx)
        write_json(ctx.task.workspace / "probe.json", {"length": 99})
        return result

    monkeypatch.setattr(team.NativeExpertTeamAdapter, "run", changed)
    new_work = task.workspace.parent / "second_workspace"
    new_work.mkdir()
    task = replace(task, workspace=new_work, output=task.output.parent / "second_execution")
    profile = Profile.from_dict(json.loads((task.output.parent / "execution/profile.lock.json").read_text())["profile"])
    registry = SimpleNamespace(resolve=lambda *args: (team.NativeExpertTeamAdapter({}), {"ready": True}))
    result = await execute(profile, task, {}, registry=registry, credential="fixture")
    assert result["status"] == "protocol_failed"
    assert not result["audit"]["native_expert"]["passed"]
    assert not result["audit"]["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["_legacy_native_v2"], indirect=True)
@pytest.mark.parametrize("attack", [None, "unknown_source", "capsule", "artifact"])
async def test_pinned_historical_auditor_preserves_strict_checks(completed, monkeypatch, attack):
    from agent.harness import historical_native_audit as historical
    from agent.harness.contracts import digest

    task, result = completed
    lock = read_lock = json.loads((task.output / "profile.lock.json").read_text())
    for name, sha in historical.PINNED_SOURCES.items():
        if name.startswith("harness/"):
            lock["adapter_source"][name.removeprefix("harness/")] = sha
    if attack == "unknown_source":
        lock["adapter_source"]["contracts.py"] = "0" * 64
    lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
    write_json(task.output / "profile.lock.json", lock)
    result["profile_sha256"] = lock["sha256"]
    if attack == "capsule":
        real_hash = historical.file_hash
        monkeypatch.setattr(
            historical, "file_hash", lambda path: "0" * 64 if "_legacy_native_v2" in str(path) else real_hash(path)
        )
    if attack == "artifact":
        write_json(task.workspace / "probe.json", {"length": 999})
    if attack or read_lock["profile"]["model"]["protocol"] == "responses":
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        audit = audit_native_expert(task.output, task.workspace, task.capability, result=result)
        assert audit["passed"] and audit["roles"] == ["PI", "CB", "MLS", "SC"]
        assert audit["model_calls"] == 4


READY = '{"decision":"ready_for_host_validation","summary":"Observed evidence reviewed","issues":[]}'
REVISION = '{"decision":"needs_revision","summary":"Required work missing","issues":["No candidate"]}'
INVALID = "Let me implement a new candidate instead of reviewing the execution."


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID, READY)], indirect=True)
async def test_one_format_repair_preserves_original_and_verifies_both_requests(completed):
    task, result = completed
    assert result["status"] == "completed" and result["llm_calls"] == 5
    assert result["audit"]["native_expert"]["passed"]
    assert (task.output / "roles/SC/response.txt").read_text() == INVALID
    assert (task.output / "roles/SC/repair_001/response.txt").read_text() == READY
    history = json.loads((task.output / "role_history.json").read_text())
    assert [row["role"] for row in history] == ["PI", "CB", "MLS", "SC"]
    assert history[-1]["calls_after"] - history[-1]["calls_before"] == 2
    assert history[-1][team.CRITIC_REPAIR_FIELD] == result[team.CRITIC_REPAIR_FIELD]
    assert result[team.CRITIC_REPAIR_FIELD]["format_valid"] is True
    assert audit_native_expert(task.output, task.workspace, task.capability)["model_calls"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(REVISION, READY)], indirect=True)
async def test_valid_revision_is_never_retried_or_upgraded(completed):
    task, result = completed
    assert result["status"] == "incomplete" and result["llm_calls"] == 4
    assert team.CRITIC_REPAIR_FIELD not in result
    assert not (task.output / "roles/SC/repair_001").exists()
    assert json.loads((task.output / "critic.json").read_text())["decision"] == "needs_revision"


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID, REVISION)], indirect=True)
async def test_repaired_revision_remains_incomplete(completed):
    task, result = completed
    assert result["status"] == "incomplete" and result["llm_calls"] == 5
    assert result[team.CRITIC_REPAIR_FIELD]["format_valid"]
    assert json.loads((task.output / "critic.json").read_text())["decision"] == "needs_revision"


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID, INVALID)], indirect=True)
async def test_second_invalid_response_cannot_start_a_third_request(completed):
    task, result = completed
    assert result["status"] == "runtime_error" and result["llm_calls"] == 5
    assert result[team.CRITIC_REPAIR_FIELD]["format_valid"] is False
    assert not (task.output / "roles/SC/repair_002").exists()
    assert not (task.output / "critic.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID,)], indirect=True)
@pytest.mark.parametrize("role_budget", [4], indirect=True)
async def test_critic_repair_cannot_exceed_frozen_model_budget(completed):
    task, result = completed
    assert result["status"] == "timeout" and result["llm_calls"] == 4
    assert result[team.CRITIC_REPAIR_FIELD]["status"] == "budget_unavailable"
    assert not (task.output / "roles/SC/repair_001").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID, READY)], indirect=True)
@pytest.mark.parametrize("native_artifact", [False], indirect=True)
async def test_repaired_ready_cannot_substitute_for_missing_computation(completed):
    _, result = completed
    assert result["status"] == "incomplete" and result["llm_calls"] == 5
    assert result["output_artifact_written"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_outputs", [(INVALID, READY)], indirect=True)
@pytest.mark.parametrize(
    "attack", ["original", "context", "response", "binding", "interval", "valid_original", "extra_directory"]
)
async def test_critic_repair_evidence_cannot_be_changed(completed, attack):
    from agent.harness.contracts import digest

    task, result = completed
    assert result["status"] == "completed"
    folder = task.output / "roles/SC"
    if attack == "original":
        (folder / "response.txt").write_text("A different invalid response")
    elif attack == "response":
        (folder / "repair_001/response.txt").write_text(REVISION)
    elif attack in {"context", "binding"}:
        path = folder / "repair_001" / ("context.json" if attack == "context" else "request_binding.json")
        value = json.loads(path.read_text())
        if attack == "context":
            value["execution_evidence"]["native_result"]["text"] = "Invented new work"
        else:
            value["payload_sha256"] = "0" * 64
        write_json(path, value)
    elif attack == "extra_directory":
        (folder / "repair_002").mkdir()
    else:
        path = task.output / "role_history.json"
        value = json.loads(path.read_text())
        if attack == "interval":
            value[-1]["calls_after"] += 1
        else:
            (folder / "response.txt").write_text(REVISION)
            value[-1]["response_sha256"] = digest(REVISION)
        write_json(path, value)
    with pytest.raises(ValueError):
        audit_native_expert(task.output, task.workspace, task.capability, result=result)


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["_legacy_pre_critic_repair"], indirect=True)
@pytest.mark.parametrize("attack", [None, "capsule", "unknown_source"])
async def test_pre_repair_native_revision_keeps_its_exact_auditor(completed, monkeypatch, attack):
    from agent.harness import historical_critic_audit as historical
    from agent.harness.contracts import digest

    task, result = completed
    assert result["coordination_policy"] == "four_roles_native_execution.v3"
    assert result["status"] == "completed" and result["llm_calls"] == 4
    if attack == "capsule":
        original = historical.file_hash
        monkeypatch.setattr(
            historical, "file_hash", lambda p: "0" * 64 if "_legacy_pre_critic_repair" in str(p) else original(p)
        )
    if attack == "unknown_source":
        path = task.output / "profile.lock.json"
        lock = json.loads(path.read_text())
        lock["adapter_source"]["native_expert_audit.py"] = "0" * 64
        lock["sha256"] = digest({key: value for key, value in lock.items() if key != "sha256"})
        write_json(path, lock)
        result["profile_sha256"] = lock["sha256"]
    if attack:
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["_legacy_pre_claude"], indirect=True)
@pytest.mark.parametrize("attack", [None, "capsule", "unknown_source", "support", "artifact", "unsupported_profile"])
async def test_pre_claude_records_remain_readable_without_relaxing_verification(completed, monkeypatch, attack):
    from agent.harness import historical_claude_audit as historical
    from agent.harness.contracts import digest

    task, result = completed
    assert result["status"] == "completed" and result["llm_calls"] == 4
    if attack == "capsule":
        original = historical.file_hash
        monkeypatch.setattr(historical, "file_hash", lambda p: "0" * 64 if "_legacy_pre_claude" in str(p) else original(p))
    elif attack in {"unknown_source", "support", "unsupported_profile"}:
        path = task.output / "profile.lock.json"
        lock = json.loads(path.read_text())
        if attack == "unknown_source":
            lock["adapter_source"]["native_expert_audit.py"] = "0" * 64
        elif attack == "support":
            lock["supporting_source"]["rsi/model_gateway.py"] = "0" * 64
        else:
            lock["profile"]["harness"] = "expert-claude-code"
        lock["sha256"] = digest({key: value for key, value in lock.items() if key != "sha256"})
        write_json(path, lock)
        result["profile_sha256"] = lock["sha256"]
    elif attack == "artifact":
        write_json(task.workspace / "probe.json", {"length": 999})
    if attack:
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        audited = audit_native_expert(task.output, task.workspace, task.capability, result=result)
        assert audited["passed"] and audited["roles"] == ["PI", "CB", "MLS", "SC"] and audited["model_calls"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["pre_isolation_v4"], indirect=True)
@pytest.mark.parametrize("attack", [None, "policy", "source", "artifact"])
async def test_explicit_v4_replay_keeps_original_semantics(completed, attack):
    from agent.harness.contracts import digest, file_hash
    from agent.harness.native_expert_audit import LEGACY_V4_SOURCES
    task, result = completed
    assert file_hash(Path(__file__).parent / "fixtures/native_expert_team_v4.py") == LEGACY_V4_SOURCES["native_expert_team.py"]
    assert result["status"] == "completed", result
    assert result["coordination_policy"] == "four_roles_native_execution.v4"
    if attack == "policy":
        result["coordination_policy"] = team.POLICY
    elif attack == "artifact":
        write_json(task.workspace / "probe.json", {"length": 999})
    elif attack == "source":
        path = task.output / "profile.lock.json"
        lock = json.loads(path.read_text())
        lock["adapter_source"]["native_expert_audit.py"] = "0" * 64
        lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
        write_json(path, lock)
        result["profile_sha256"] = lock["sha256"]
    if attack:
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]


@pytest.mark.asyncio
async def test_v5_contexts_and_sessions_are_independent(completed):
    task, result = completed
    assert result["status"] == "completed"
    contexts = {role: json.loads((task.output / "roles" / role / "context.json").read_text())
                for role in ["PI", "CB", "MLS", "SC"]}
    assert "handoffs" not in contexts["PI"]
    assert set(contexts["CB"]["handoffs"]) == {"PI"}
    assert set(contexts["MLS"]["handoffs"]) == {"PI", "CB"}
    critic = contexts["SC"]
    assert "handoffs" not in critic and not any(k in critic for k in ["PI", "CB", "MLS"])
    assert "text" not in critic["execution_evidence"]["native_result"]
    assert critic["episode_review_contract"] == contexts["PI"]["episode_review_contract"]
    bindings = [json.loads((task.output / "roles" / r / "request_binding.json").read_text())
                for r in ["PI", "CB", "SC"]]
    assert result["context_policy"] == "scientist_role_context.v2"
    assert all(value["role_context"]["policy"] == "scientist_role_context.v2" for value in contexts.values())
    assert all(value["context_policy"] == "scientist_role_context.v2" for value in bindings)
    assert len({value["session_id"] for value in bindings}) == 3
    assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]
    assert (task.output / "roles/MLS/native_result.json").read_text().find("Measured length 10") >= 0


@pytest.mark.asyncio
async def test_v5_audit_rejects_reused_role_session(completed):
    task, result = completed
    pi = json.loads((task.output / "roles/PI/request_binding.json").read_text())
    path = task.output / "roles/SC/request_binding.json"
    sc = json.loads(path.read_text())
    sc["session_id"] = pi["session_id"]
    write_json(path, sc)
    with pytest.raises(ValueError, match="Reused role session"):
        audit_native_expert(task.output, task.workspace, task.capability)
