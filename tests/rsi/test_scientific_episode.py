import json
from types import SimpleNamespace

import pytest

from agent.rsi.engine.scientific_episode import artifact_written, episode_contract


def contract(capability, **updates):
    value = {
        "schema": 1,
        "kind": "reproduction",
        "context_tool": "inspect_case_evidence",
        "artifact_file": "review.json",
    }
    value.update(updates)
    (capability / "scientific_task.json").write_text(json.dumps(value))
    return value


@pytest.mark.parametrize("phase,artifact", [("develop", "candidate.json"), ("reuse", "predictions.json")])
def test_model_lifecycle_retains_existing_contract(tmp_path, phase, artifact):
    received = episode_contract(tmp_path, phase, set())
    assert received.artifact_file == artifact and received.context_tool == "inspect_model_code_task"


@pytest.mark.parametrize(
    "updates",
    [
        {"schema": True},
        {"schema": 2},
        {"kind": "untrusted"},
        {"extra": "field"},
        {"context_tool": "not_admitted"},
        {"context_tool": None},
        {"artifact_file": "../escape.json"},
        {"artifact_file": "/absolute.json"},
        {"artifact_file": "sub/artifact.json"},
        {"artifact_file": "artifact.txt"},
        {"artifact_file": None},
    ],
)
def test_reproduction_contract_rejects_scope_or_path_drift(tmp_path, updates):
    contract(tmp_path, **updates)
    with pytest.raises(ValueError):
        episode_contract(tmp_path, "chat", {"inspect_case_evidence"})


def test_explicit_frozen_contract_is_required(tmp_path):
    with pytest.raises(ValueError, match="host-authored"):
        episode_contract(tmp_path, "chat", {"inspect_case_evidence"})
    other = tmp_path / "outside.json"
    other.write_text("{}")
    (tmp_path / "scientific_task.json").symlink_to(other)
    with pytest.raises(ValueError, match="host-authored"):
        episode_contract(tmp_path, "chat", {"inspect_case_evidence"})


def test_output_presence_does_not_mean_scientific_acceptance(tmp_path):
    contract(tmp_path)
    received = episode_contract(tmp_path, "chat", {"inspect_case_evidence"})
    path = tmp_path / "review.json"
    assert not artifact_written(tmp_path, received)
    for text in ["", "[]", "not JSON"]:
        path.write_text(text)
        assert not artifact_written(tmp_path, received)
    path.write_text('{"unverified": true}')
    assert artifact_written(tmp_path, received)  # Acceptance remains an independent host gate.
    path.unlink()
    path.symlink_to(tmp_path / "scientific_task.json")
    assert not artifact_written(tmp_path, received)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["chat", "develop", "reuse"])
@pytest.mark.parametrize(
    "complete_path,finalizer,expected",
    [(True, True, "completed"), (False, True, "incomplete"), (True, False, "incomplete")],
)
async def test_scientist_generic_episode_keeps_four_role_and_finalizer_gates(
    tmp_path, monkeypatch, complete_path, finalizer, expected, phase
):
    from langchain_core.runnables import RunnableLambda

    from agent import chat_agent
    from agent.graph import compile as graph_compile
    from agent.rsi import scientist_runtime as runtime

    workspace, capability, output = [tmp_path / name for name in ["workspace", "capability", "output"]]
    for path in [workspace, capability, output]:
        path.mkdir()
    contract(capability)
    context_tool = "inspect_case_evidence" if phase == "chat" else "inspect_model_code_task"
    allowed = {context_tool, "submit_evidence_review" if phase == "chat" else "predict_with_generated_model"}
    artifact = "review.json" if phase == "chat" else "candidate.json" if phase == "develop" else "predictions.json"
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def inspect(arguments):
        assert arguments == {}
        calls.append("context")
        return {"facts": {"full_paper_reproduced": False}}

    async def tools(client, **kwargs):
        assert kwargs["validate_results"] is (phase != "chat")
        assert kwargs["allowed_tools"] == allowed
        return [SimpleNamespace(name=context_tool, ainvoke=inspect)]

    def chains(llm, all_tools, pi_tools):
        assert [t.name for t in pi_tools] == [context_tool]
        return {}

    class Graph:
        async def astream(self, state, config, **kwargs):
            chains = config["configurable"]["chains"]
            assert chains["scientific_workflow_required"] is True
            sources, unused = await chains["research_search_runner"]("inspect data")
            assert sources[0]["source"] == "permitted_task_metadata"
            await chains["research_search_runner"]("cached data")
            (workspace / artifact).write_text(
                json.dumps([{"id": "row", "prediction": 0.5}] if phase == "reuse" else {"fixture": True})
            )
            nodes = ["research_plan_node", "plan_node", "execute_node", "finalize_node"]
            if not complete_path:
                nodes.remove("plan_node")
            for node in nodes:
                yield {node: {"finalizer_succeeded": finalizer}}

    monkeypatch.setattr(runtime, "Client", Client)
    monkeypatch.setattr(runtime, "StdioTransport", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda value: "offline fixture"))
    monkeypatch.setattr(runtime, "build_tools", tools)
    monkeypatch.setattr(chat_agent, "_build_chains_for_llm", chains)
    monkeypatch.setattr(graph_compile, "create_agent_graph", Graph)
    result = await runtime.run_scientist(
        workspace=workspace,
        capability=capability,
        output=output,
        python_root=tmp_path,
        prompt="review facts",
        model="offline-fixture",
        base_url="http://unused",
        api_key="fixture",
        allowed_tools=allowed,
        phase=phase,
    )
    assert result["status"] == expected
    assert result["context_tool"] == context_tool and result["llm_calls"] == 0
    assert calls == ["context"]


@pytest.mark.parametrize("phase", ["develop", "reuse"])
@pytest.mark.parametrize("payload", [{}, [], [{"id": "row", "prediction": 0.5}], None])
def test_model_artifact_shape_follows_native_tool_output(tmp_path, phase, payload):
    received = episode_contract(tmp_path, phase, set())
    path = tmp_path / received.artifact_file
    path.write_text(json.dumps(payload))
    expected_type = list if phase == "reuse" else dict
    assert artifact_written(tmp_path, received) is isinstance(payload, expected_type)
    # This is a transport-completion check; row and numerical validation stay
    # in the independent model controller, including rejection of empty output.


@pytest.mark.parametrize("phase", ["develop", "reuse"])
def test_model_artifact_rejects_malformed_json_or_symlink(tmp_path, phase):
    received = episode_contract(tmp_path, phase, set())
    path = tmp_path / received.artifact_file
    path.write_text("[broken")
    assert not artifact_written(tmp_path, received)
    path.unlink()
    target = tmp_path / "outside.json"
    target.write_text("[]" if phase == "reuse" else "{}")
    path.symlink_to(target)
    assert not artifact_written(tmp_path, received)
