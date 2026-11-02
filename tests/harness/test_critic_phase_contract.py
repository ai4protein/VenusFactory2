"""Phase review cannot replace host acceptance or rewrite a critic decision."""

import json

import pytest
from test_native_expert_audit import completed as completed
from test_native_expert_audit import critic_outputs as critic_outputs
from test_native_expert_audit import critic_prefix as critic_prefix
from test_native_expert_audit import historical_revision as historical_revision
from test_native_expert_audit import native_artifact as native_artifact
from test_native_expert_audit import role_budget as role_budget

from agent.harness import historical_phase_audit as historical
from agent.harness import native_expert_team as team
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.harness.native_expert_audit import audit_native_expert
from agent.rsi.engine.scientific_episode import EpisodeContract


@pytest.mark.parametrize("decision", ["ready_for_host_validation", "needs_revision"])
@pytest.mark.parametrize("prefix", ["", "# </think>", "</think>\n"])
@pytest.mark.parametrize("fence", [None, "```json\n", "```\r\n"])
def test_known_wrappers_preserve_decision(decision, prefix, fence):
    value = {"decision": decision, "summary": "Evidence reviewed", "issues": []}
    raw = json.dumps(value)
    if fence:
        raw = fence + raw + "\n```"
    assert team.critic_decision(prefix + raw) == value


GOOD = '{"decision":"ready_for_host_validation","summary":"Observed","issues":[]}'


@pytest.mark.parametrize(
    "raw",
    [
        GOOD + GOOD,
        GOOD + " and another instruction",
        "Here is the result: " + GOOD,
        "<think>unreviewed reasoning</think>" + GOOD,
        "# </think># </think>" + GOOD,
        "```json\n" + GOOD,
        GOOD[:-1],
        GOOD.replace('"issues":[]', '"issues":NaN'),
        GOOD.replace('"issues":[]', '"issues":Infinity'),
        GOOD.replace('"issues":[]', '"issues":[],"decision":"needs_revision"'),
        GOOD.replace('"issues":[]', '"issues":[{}]'),
        GOOD.replace('"Observed"', '" "'),
        " " * 128001 + GOOD,
        None,
    ],
)
def test_parser_never_salvages_ambiguous_or_incomplete_decisions(raw):
    with pytest.raises(ValueError):
        team.critic_decision(raw)


@pytest.mark.parametrize("phase,artifact", [("develop", "candidate.json"), ("reuse", "predictions.json")])
def test_lifecycle_review_defers_host_acceptance(phase, artifact):
    contract = team.episode_review_contract(EpisodeContract("model_lifecycle", "inspect", artifact), phase)
    assert contract["phase"] == phase and contract["artifact"] == artifact
    assert "does not accept a model" in contract["decision_boundary"]
    if phase == "develop":
        assert "submit" in " ".join(contract["current_obligations"])
        assert "fresh-session reuse" in " ".join(contract["deferred_host_decisions"])
    else:
        assert "fresh session" in " ".join(contract["current_obligations"])
        assert "numerical replay" in " ".join(contract["deferred_host_decisions"])


def test_reproduction_obligations_still_require_actual_tool_work():
    contract = team.episode_review_contract(EpisodeContract("reproduction", "inspect", "receipt.json"), "chat")
    assert "required tools successfully" in " ".join(contract["current_obligations"])
    with pytest.raises(ValueError):
        team.episode_review_contract(EpisodeContract("reproduction", "inspect", "receipt.json"), "develop")


@pytest.mark.asyncio
@pytest.mark.parametrize("critic_prefix", ["# </think>"], indirect=True)
async def test_raw_wrapper_is_retained_and_independently_reparsed(completed):
    task, result = completed
    assert result["status"] == "completed"
    raw = (task.output / "roles/SC/response.txt").read_text()
    assert raw.startswith("# </think>")
    assert audit_native_expert(task.output, task.workspace, task.capability)["passed"]
    assert result["audit"]["native_expert"]["scientific_acceptance"] == "requires separate domain validation"
    history = json.loads((task.output / "role_history.json").read_text())
    assert history[-1]["response_sha256"] == digest(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_revision", ["_legacy_pre_phase_contract"], indirect=True)
@pytest.mark.parametrize("attack", [None, "unknown_source", "capsule", "artifact"])
async def test_pre_phase_capsule_preserves_old_requests(completed, monkeypatch, attack):
    task, result = completed
    assert result["status"] == "completed"
    assert result["coordination_policy"] == "four_roles_native_execution.v2"
    lock = json.loads((task.output / "profile.lock.json").read_text())
    if attack == "unknown_source":
        lock["adapter_source"]["scientist_policy.py"] = "0" * 64
        lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
        write_json(task.output / "profile.lock.json", lock)
        result["profile_sha256"] = lock["sha256"]
    elif attack == "capsule":
        monkeypatch.setattr(historical, "file_hash", lambda path: "0" * 64)
    elif attack == "artifact":
        write_json(task.workspace / "probe.json", {"length": 999})
    if attack:
        with pytest.raises(ValueError):
            audit_native_expert(task.output, task.workspace, task.capability, result=result)
    else:
        assert audit_native_expert(task.output, task.workspace, task.capability, result=result)["passed"]


def test_legacy_parser_does_not_retroactively_accept_the_new_wrapper():
    from agent.harness._legacy_pre_phase_contract.harness.native_expert_team import critic_decision

    with pytest.raises(ValueError):
        critic_decision("# </think>" + GOOD)
