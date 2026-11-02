import json
from functools import partial
from types import SimpleNamespace

import pytest

from agent.rsi.reproduction_tool_contracts import validate_reproduction_result
from agent.rsi.scientific_episode import completion_feedback, episode_contract, successful_tools


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def paired(name, cid="1", value=None, error=False):
    return [
        {"kind": "tool_call_start", "tool_call_id": cid, "tool_name": "mcp__venusfactory__" + name, "args": {}},
        {
            "kind": "tool_result",
            "tool_call_id": cid,
            "is_error": error,
            "output": json.dumps({"success": True} if value is None else value),
        },
    ]


def contract(capability, **updates):
    value = {
        "schema": 2,
        "kind": "reproduction",
        "context_tool": "discover_reproduction_plugins",
        "artifact_file": "case_submission.json",
        "completion_repair_limit": 1,
        "required_tools": ["discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"],
    }
    value.update(updates)
    write(capability, "scientific_task.json", value)
    return value


@pytest.mark.parametrize(
    "updates",
    [
        {"required_tools": []},
        {"required_tools": ["not_admitted"]},
        {"required_tools": ["submit_reproduction"]},
        {"required_tools": ["discover_reproduction_plugins"] * 2},
        {"required_tools": [None]},
        {"required_tools": "discover_reproduction_plugins"},
        {"completion_repair_limit": True},
        {"completion_repair_limit": 2},
    ],
)
def test_completion_obligations_cannot_expand_tools_or_repair_budget(tmp_path, updates):
    contract(tmp_path, **updates)
    with pytest.raises(ValueError):
        episode_contract(
            tmp_path, "chat", {"discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"}
        )


def test_only_successful_paired_calls_discharge_obligations(tmp_path):
    value = contract(tmp_path)
    bound = episode_contract(tmp_path, "chat", set(value["required_tools"]))
    events = paired("discover_reproduction_plugins") + paired("submit_reproduction", "2", error=True)
    events += paired("inspect_reproduction_job", "3", value={"accepted": False})
    assert successful_tools(events) == {"discover_reproduction_plugins"}
    feedback = completion_feedback(bound, events)
    assert "submit_reproduction" in feedback and "inspect_reproduction_job" in feedback
    assert "original budget" in feedback and "Do not repeat a successful submission" in feedback
    with pytest.raises(ValueError, match="Unpaired"):
        successful_tools([events[-1]])
    with pytest.raises(ValueError, match="duplicate"):
        successful_tools(paired("x") + paired("x"))


@pytest.mark.parametrize(
    "tool,action",
    [
        ("discover_reproduction_plugins", "catalog"),
        ("submit_reproduction", "submit"),
        ("inspect_reproduction_job", "status"),
    ],
)
def test_broker_output_must_match_actual_response_and_action(tmp_path, tool, action):
    work, cap = tmp_path / "work", tmp_path / "cap"
    value = {
        "success": True,
        "job_id": "case_01234567890123456789",
        "plugin_id": "phl7.hrex_validation.v1",
        "parameters": {"backend": "cuda"},
        "status": "queued",
    }
    write(work, "case_rpc/one.request.json", {"action": action})
    write(work, "case_rpc/one.response.json", value)
    if tool == "submit_reproduction":
        write(
            work,
            "case_submission.json",
            {
                "schema": 1,
                "job_id": value["job_id"],
                "plugin_id": value["plugin_id"],
                "parameters": value["parameters"],
                "submission_only": True,
            },
        )
        value["submission_receipt"] = "case_submission.json"
    validate_reproduction_result(tool, value, workspace=work, capability=cap)
    with pytest.raises(ValueError):
        validate_reproduction_result(tool, {**value, "status": "completed"}, workspace=work, capability=cap)
    write(work, "case_rpc/one.request.json", {"action": "different"})
    with pytest.raises(ValueError, match="host-broker"):
        validate_reproduction_result(tool, value, workspace=work, capability=cap)


@pytest.mark.asyncio
async def test_registered_lookup_is_verified_as_data_not_fabricated_file(tmp_path):
    from agent.graph.execution.verify import run_cb_post_check, run_mls_post_check

    recipe = {
        "kind": "verified_reproduction_recipe",
        "plugin": "phl7.hrex_validation.v1",
        "parameters": {"backend": "cuda"},
        "novel_model_claim": False,
    }
    write(tmp_path, "registered_recipe.json", recipe)
    validate = partial(
        validate_reproduction_result, "inspect_registered_reproduction", workspace=tmp_path, capability=tmp_path
    )
    ctx = SimpleNamespace(tool=SimpleNamespace(metadata={"result_validator": validate}))
    for verify in (run_mls_post_check, run_cb_post_check):
        assert (await verify(ctx, json.dumps({"success": True, **recipe}))).ok
        assert not (await verify(ctx, json.dumps({"success": True, **recipe, "parameters": {}}))).ok
    path = tmp_path / "registered_recipe.json"
    path.rename(tmp_path / "original.json")
    path.symlink_to(tmp_path / "original.json")
    with pytest.raises(ValueError, match="symlink"):
        validate(recipe)


def test_review_contract_preserves_negative_feedback_but_rejects_false_acceptance(tmp_path):
    from agent.rsi.case_evidence import validate_review

    facts = {"full_paper_reproduced": False, "replicas": 8}
    evidence = {"job_id": "fixture", "plugin": "phl7.hrex_validation.v1", "report_sha256": "pinned", "facts": facts}
    write(tmp_path, "evidence.json", evidence)
    validate = partial(validate_reproduction_result, workspace=tmp_path, capability=tmp_path)
    validate("inspect_case_evidence", evidence)
    with pytest.raises(ValueError):
        validate("inspect_case_evidence", {**evidence, "facts": {}})
    rejected = {
        "accepted": False,
        "status": "claims_rejected",
        "attempts": 1,
        "attempts_remaining": 1,
        "reason": "wrong fields",
    }
    write(tmp_path, "review_attempt_1.json", {"feedback": rejected})
    validate("submit_evidence_review", rejected)
    review = validate_review(facts, facts)
    write(tmp_path, "review.json", review)
    accepted = {"accepted": True, "artifact": "review.json", "claims_sha256": review["claims_sha256"], "attempts": 2}
    write(tmp_path, "review_attempt_2.json", {"feedback": accepted})
    validate("submit_evidence_review", accepted)
    write(tmp_path, "review.json", {**review, "claims": {**facts, "full_paper_reproduced": True}})
    with pytest.raises(ValueError):
        validate("submit_evidence_review", accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repair_succeeds,tool_cap,expected,attempts",
    [
        (True, 12, "completed", 2),
        (False, 12, "incomplete", 2),
        (True, 2, "incomplete", 1),
    ],
)
async def test_missing_inspection_is_replanned_once_in_same_session_without_resubmission(
    tmp_path, monkeypatch, repair_succeeds, tool_cap, expected, attempts
):
    from langchain_core.runnables import RunnableLambda

    from agent import chat_agent
    from agent.graph import compile as graph_compile
    from agent.rsi import scientist_runtime as runtime

    work, cap, out = [tmp_path / x for x in ("work", "cap", "out")]
    for path in (work, cap, out):
        path.mkdir()
    declaration = contract(cap)
    observed, callbacks = [], {}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def build(client, **kwargs):
        callbacks.update(kwargs)
        return [SimpleNamespace(name="discover_reproduction_plugins")]

    def execute(name):
        observed.append(name)
        callbacks["result"]["tool_calls"] += 1
        for item in paired(name, str(len(observed))):
            callbacks["event"](item)

    class Graph:
        calls = 0

        async def astream(self, state, config, **kwargs):
            self.calls += 1
            if self.calls == 1:
                execute("discover_reproduction_plugins")
                execute("submit_reproduction")
                write(work, "case_submission.json", {"job_id": "fixture"})
            else:
                assert state["waiting_for"] == "skip_to_plan"
                assert state["finalizer_succeeded"] is False
                assert "Do not repeat a successful submission" in state["messages"][-1].content
                assert (work / "case_submission.json").exists()
                if repair_succeeds:
                    execute("inspect_reproduction_job")
            for node in ("research_plan_node", "plan_node", "execute_node", "finalize_node"):
                yield {
                    node: {"status": "waiting_for_iteration", "waiting_for": "iteration", "finalizer_succeeded": True}
                }

    graph = Graph()
    monkeypatch.setattr(runtime, "Client", Client)
    monkeypatch.setattr(runtime, "StdioTransport", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda value: "offline"))
    monkeypatch.setattr(runtime, "build_tools", build)
    monkeypatch.setattr(chat_agent, "_build_chains_for_llm", lambda *args: {})
    monkeypatch.setattr(graph_compile, "create_agent_graph", lambda: graph)
    result = await runtime.run_scientist(
        workspace=work,
        capability=cap,
        python_root=tmp_path,
        output=out,
        prompt="Launch and inspect the allocated job",
        model="offline",
        base_url="http://unused",
        api_key="fixture",
        allowed_tools=set(declaration["required_tools"]),
        phase="chat",
        max_tool_calls=tool_cap,
    )
    assert result["status"] == expected and graph.calls == attempts
    assert observed.count("submit_reproduction") == 1
    assert result["tool_calls"] <= tool_cap and result["llm_calls"] == 0
    assert len(result["completion_contract"]["repairs"]) == attempts - 1
    assert result["human_interventions"] == 0
