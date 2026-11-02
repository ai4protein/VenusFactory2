"""The improver is distinct policy data with source-bound developmental feedback."""

import copy
import json
import sys
from pathlib import Path

import pytest
from test_reap_goal_comparison import candidate as model_candidate
from test_reap_science_goal import setup_goal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_improver_revision import project_development_history, revision_prompt

from agent.harness.host import write_json
from agent.rsi.improver_policy import (
    BASE_POLICY,
    candidate,
    revision_packet,
    validate_candidate,
    validate_history,
    validate_packet,
    validate_policy,
)


def history():
    return [
        {
            "case": "fixture",
            "cycle": 0,
            "status": "failed",
            "goal_sha256": "a" * 64,
            "protocol_sha256": "b" * 64,
            "error_type": "CandidateCodeError",
        }
    ]


def revised():
    return {
        **BASE_POLICY,
        "proposal_instruction": BASE_POLICY["proposal_instruction"]
        + " Before proposing a complex estimator, use development evidence to justify its added variance.",
    }


def test_policy_revision_is_bound_and_remains_untested():
    packet = revision_packet(BASE_POLICY, history())
    result = candidate(revised(), "Test whether variance-aware model proposals improve offspring.", packet)
    assert validate_candidate(result, packet) == result
    assert not result["promoted"] and result["status"] == "untested_candidate"
    changed = copy.deepcopy(result)
    changed["promoted"] = True
    with pytest.raises(ValueError):
        validate_candidate(changed, packet)
    with pytest.raises(ValueError, match="Unchanged"):
        candidate(BASE_POLICY, "A repeated policy is not a successor.", packet)


@pytest.mark.parametrize("extra", ["budget", "evaluator", "tools", "future_labels"])
def test_policy_cannot_change_host_contract_fields(extra):
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_policy({**BASE_POLICY, extra: {}})


def test_packet_rejects_final_scores_and_mutated_parent():
    contaminated = history()
    contaminated[0]["future_spearman"] = 0.9
    with pytest.raises(ValueError, match="Unexpected development"):
        validate_history(contaminated)
    packet = revision_packet(BASE_POLICY, history())
    packet["parent_policy"]["revision_instruction"] += " Mutated."
    with pytest.raises(ValueError, match="binding"):
        validate_packet(packet)


def test_current_parent_revision_instruction_is_actually_used():
    child = {
        **revised(),
        "revision_instruction": "Use failures to compare the proposal assumptions with measured development outcomes, then revise both instructions based on a falsifiable explanation.",
    }
    first = revision_prompt(revision_packet(BASE_POLICY, history()))
    second = revision_prompt(revision_packet(child, history()))
    assert first != second and child["revision_instruction"] in second
    assert BASE_POLICY["revision_instruction"] not in second


def test_history_projects_rejected_model_without_fitted_state_or_future_scores(tmp_path):
    store = setup_goal(tmp_path)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["split"] = {"family": "fixture"}
    write_json(store.root / "protocol.json", protocol)
    cycle, lc, mid = model_candidate(store.root, 0, 1000.0)
    store.mutate(
        lambda s: s.update(
            status="budget_exhausted",
            cycles=[{**cycle, "phase": "develop", "status": "candidate_rejected"}],
            inflight=None,
        )
    )
    write_json(store.root / "future_metrics.json", {"secret_audit_score": 123456})
    rows, refs = project_development_history([store.root])
    packet = revision_packet(BASE_POLICY, rows)
    encoded = json.dumps(packet)
    assert rows[0]["model_id"] == mid and rows[0]["acceptance_passed"] is False
    assert "source_code" in rows[0] and "model_state" not in encoded and "secret_audit_score" not in encoded
    assert len(refs) == 1 and refs[0]["path"] == str(store.root)
    path = lc.root / ("assessment_" + mid + ".json")
    record = json.loads(path.read_text())
    record["passed"] = True
    write_json(path, record)
    with pytest.raises(ValueError, match="binding"):
        project_development_history([store.root])


@pytest.mark.asyncio
async def test_actual_isolated_mcp_revision_preserves_single_attempt(tmp_path):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    import agent.rsi.improver_policy as policy_module
    from agent.rsi.langgraph_tools import mcp_command

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    package = Path(policy_module.__file__).parent
    (bundle / "generated_tools.py").write_bytes((package / "improver_tools.py").read_bytes())
    (bundle / "improver_policy.py").write_bytes((package / "improver_policy.py").read_bytes())
    packet = revision_packet(BASE_POLICY, history())
    write_json(bundle / "revision_context.json", packet)
    transport = StdioTransport(
        command="bwrap", args=mcp_command(workspace, bundle, Path(sys.prefix)), keep_alive=False, cwd=str(workspace)
    )
    async with Client(transport) as client:
        context = await client.call_tool("inspect_improver_context", {})
        assert not getattr(context, "is_error", False)
        reply = await client.call_tool(
            "submit_improver_policy",
            {
                "policy_json": json.dumps(revised()),
                "rationale": "Test whether model complexity justification improves development choices.",
            },
        )
        assert not getattr(reply, "is_error", False)
        path = workspace / "improver_candidate.json"
        saved = path.read_bytes()
        assert validate_candidate(json.loads(saved), packet)["status"] == "untested_candidate"
        try:
            reply = await client.call_tool(
                "submit_improver_policy",
                {
                    "policy_json": json.dumps(revised()),
                    "rationale": "Attempting a second policy must retain the first candidate.",
                },
            )
        except Exception:
            pass
        else:
            assert getattr(reply, "is_error", False)
        assert path.read_bytes() == saved
    assert json.loads((workspace / "improver_attempt.json").read_text())["status"] == "candidate_submitted"


@pytest.mark.asyncio
async def test_invalid_proposal_cannot_be_retried_by_restarting_tool_process(tmp_path):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    import agent.rsi.improver_policy as policy_module
    from agent.rsi.langgraph_tools import mcp_command

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    package = Path(policy_module.__file__).parent
    for src, dst in [("improver_tools.py", "generated_tools.py"), ("improver_policy.py", "improver_policy.py")]:
        (bundle / dst).write_bytes((package / src).read_bytes())
    write_json(bundle / "revision_context.json", revision_packet(BASE_POLICY, history()))
    first = None
    for policy_text in ["{}", json.dumps(revised())]:
        transport = StdioTransport(
            command="bwrap", args=mcp_command(workspace, bundle, Path(sys.prefix)), keep_alive=False, cwd=str(workspace)
        )
        async with Client(transport) as client:
            await client.call_tool("inspect_improver_context", {})
            try:
                reply = await client.call_tool(
                    "submit_improver_policy",
                    {
                        "policy_json": policy_text,
                        "rationale": "An invalid attempt still consumes its single allocated opportunity.",
                    },
                )
            except Exception:
                pass
            else:
                assert getattr(reply, "is_error", False)
        attempt = (workspace / "improver_attempt.json").read_bytes()
        if first is None:
            first = attempt
        assert attempt == first and not (workspace / "improver_candidate.json").exists()


@pytest.mark.asyncio
async def test_frozen_improver_instruction_reaches_model_development(tmp_path, monkeypatch):
    import run_science_goal as base
    from run_reap_improver_goal import PolicyReapGoal
    from run_reap_science_goal import PLUGIN, ReapModelGoal
    from test_generated_features import CODE, ROWS

    from agent.harness.contracts import digest, file_hash
    from agent.rsi.improver_policy import identity
    from agent.rsi.model_plugins import get_model_plugin
    from agent.science_goal import run_goal

    store = setup_goal(tmp_path)
    policy = revised()
    path = store.root / "improver_policy.json"
    write_json(path, policy)
    protocol = json.loads((store.root / "protocol.json").read_text())
    action = (
        ReapModelGoal(store, "fixture").development_action()
        + "\nModel-code proposal instruction from the frozen improver policy:\n"
        + policy["proposal_instruction"]
        + "\nThe fixed model ABI, data access, evaluation rules and remaining budget take precedence."
    )
    protocol["improver"] = {
        "policy_id": identity(policy),
        "file_sha256": file_hash(path),
        "development_action_sha256": digest(action),
    }
    write_json(store.root / "protocol.json", protocol)
    goal = PolicyReapGoal(store, "fixture")
    phases = []

    async def execute(profile, task, runtimes, credential):
        phases.append(task.phase)
        plugin = get_model_plugin(PLUGIN)
        if task.phase == "develop":
            assert policy["proposal_instruction"] in task.prompt
            write_json(
                task.workspace / "candidate.json", plugin.fit(ROWS[:60], goal.scope, {"source_code": CODE, "seed": 19})
            )
        else:
            assert policy["proposal_instruction"] not in task.prompt
            inputs = json.loads((task.workspace / "inputs.json").read_text())
            lc, mid = goal.lifecycle(store.read())
            write_json(task.workspace / "predictions.json", plugin.predict(lc._model(mid), inputs, goal.scope))
        return {
            "status": "completed",
            "session_id": "fixture-" + str(len(phases)),
            "host_wall_seconds": 0.01,
            "audit": {"passed": True, "requests": {"model_calls": 1}, "events": {"tool_calls": 1}},
        }

    monkeypatch.setattr(base, "execute", execute)
    monkeypatch.setattr(base, "audit_events", lambda *args: {"passed": True})
    state = await run_goal(store, goal.cycle, goal.evaluate)
    assert state["status"] == "completed" and phases == ["develop", "reuse"]
    assert not (store.root / "future_metrics.json").exists()
    changed = {**policy, "proposal_instruction": policy["proposal_instruction"] + " Changed after preparation."}
    write_json(path, changed)
    with pytest.raises(ValueError, match="policy changed"):
        PolicyReapGoal(store, "fixture")


@pytest.mark.parametrize("mutation", [None, "missing", "changed", "policy"])
def test_history_survives_runtime_upgrade_only_with_verified_archive(tmp_path, monkeypatch, mutation):
    import hashlib
    from types import SimpleNamespace

    import run_improver_revision as revision

    import agent.rsi.generated_features as feature_module

    store = setup_goal(tmp_path)
    cycle, lc, mid = model_candidate(store.root, 0, 1000.0)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["split"] = {"family": "fixture"}
    package = Path(feature_module.__file__).parent
    names = [
        "generated_features.py",
        "generated_model.py",
        "code_sandbox.py",
        "code_worker.py",
        "protein_references.py",
        "numeric_runtime.py",
    ]
    for name in names:
        relative = "src/agent/rsi/" + name
        target = store.root / "implementation" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / name).read_bytes())
        protocol["implementation"][relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    if mutation == "missing":
        protocol["implementation"].pop("src/agent/rsi/code_worker.py")
    elif mutation == "changed":
        target.write_text("changed historical source")
    elif mutation == "policy":
        saved = json.loads((lc.root / "policy.json").read_text())
        saved["minimum_skill"] = 0.5
        write_json(lc.root / "policy.json", saved)
    write_json(store.root / "protocol.json", protocol)
    store.mutate(
        lambda state: state.update(
            status="budget_exhausted",
            cycles=[{**cycle, "phase": "develop", "status": "candidate_rejected"}],
            inflight=None,
        )
    )
    monkeypatch.setattr(revision, "get_model_plugin", lambda _: SimpleNamespace(source_sha256="a" * 64))
    before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    if mutation:
        with pytest.raises(ValueError, match="Historical"):
            project_development_history([store.root])
    else:
        rows, _ = project_development_history([store.root])
        assert rows[0]["model_id"] == mid and not rows[0]["acceptance_passed"]
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("kind", ["legacy", "references", "current", "unknown"])
def test_archived_fingerprint_schema_is_read_without_execution(tmp_path, kind):
    import hashlib

    from run_improver_revision import historical_plugin_hash

    names = ["generated_features.py", "generated_model.py", "code_sandbox.py", "code_worker.py"]
    if kind in {"references", "current"}:
        names.append("protein_references.py")
        if kind == "current":
            names.append("numeric_runtime.py")
    elif kind == "unknown":
        names.append("unrecognized.py")
    hashes = {}
    for name in names:
        path = tmp_path / "src/agent/rsi" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = "# fixture archive\n"
        if name == "generated_features.py":
            source = f"def implementation_hash():\n    raise AssertionError('Do not execute archived source')\n    return b''.join((root / n).read_bytes() for n in {names!r})\n"
        path.write_text(source)
        hashes[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if kind == "unknown":
        with pytest.raises(ValueError, match="unsupported"):
            historical_plugin_hash(tmp_path, hashes)
    else:
        expected = hashlib.sha256(
            b"".join((tmp_path / "src/agent/rsi" / name).read_bytes() for name in names)
        ).hexdigest()
        assert historical_plugin_hash(tmp_path, hashes) == expected
