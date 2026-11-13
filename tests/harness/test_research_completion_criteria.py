"""Scientific completion and public delivery remain separately verified."""
import json

import pytest

from agent.harness import Profile
from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness.contracts import file_hash


@pytest.fixture
def evaluated_goal(tmp_path):
    def make(deliverable="tool", legacy=False):
        profile = Profile.from_dict({
            "mode": "science_expert", "harness": "expert-deepseek-harness",
            "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
            "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120,
                       "output_tokens": 4096, "thinking": "disabled"},
        })
        store = ar.create(tmp_path / "goal", profile, "Develop a reusable capability",
                          {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240})
        workspace = store.root / "workspace"
        workspace.mkdir(exist_ok=True)
        report = workspace / "report.md"
        report.write_text("Executed research evidence.\n")
        episode = store.root / "episodes/cycle_000"
        episode.mkdir(parents=True)
        (episode / "review.json").write_text(json.dumps({
            "decision": "ready_for_host_validation", "summary": "Research reviewed", "issues": [],
        }))
        state = store.read()
        if legacy:
            state["definition"]["goal"]["criteria"] = [
                key for key in state["definition"]["goal"]["criteria"]
                if key != "capability_admission_resolved"
            ]
        state["progress"].update({
            "plan": {"deliverable": deliverable, "steps": [{
                "id": "compute", "status": "completed",
                "evidence": [{"path": "report.md", "sha256": file_hash(report)}],
            }]},
            "reviewed_episode": 0, "last_review": {"decision": "ready_for_host_validation"},
        })
        return store, state, ar.AutonomousResearch(store, "fixture", {}, tools=[])
    return make


@pytest.mark.parametrize("deliverable", ["tool", "model"])
@pytest.mark.parametrize("legacy", [False, True])
def test_missing_delivery_preserves_verified_research_but_never_completes(evaluated_goal, deliverable, legacy):
    _, state, runner = evaluated_goal(deliverable, legacy)
    result = runner.evaluate(state)
    checks = result["criteria"]
    assert result["research_completed"] is True and result["delivery_pending"] is True
    assert checks["research_plan_executed"]["passed"] is True
    if legacy:
        assert checks["research_outputs_reviewed"]["passed"] is False
    else:
        assert checks["research_outputs_reviewed"]["passed"] is True
        assert checks["capability_admission_resolved"]["passed"] is False
    assert not all(row["passed"] for row in checks.values())


@pytest.mark.parametrize("fault", ["artifact", "review", "review_missing"])
def test_pending_delivery_cannot_mask_invalid_research_evidence(evaluated_goal, fault):
    store, state, runner = evaluated_goal()
    if fault == "artifact":
        (store.root / "workspace/report.md").write_text("Changed after research review.")
    elif fault == "review":
        (store.root / "episodes/cycle_000/review.json").write_text('{"decision":"needs_revision"}')
    else:
        (store.root / "episodes/cycle_000/review.json").unlink()
    result = runner.evaluate(state)
    assert result["delivery_pending"] is True
    assert not any(row["passed"] for row in result["criteria"].values())


def test_missing_model_delivery_does_not_hide_changed_existing_method_artifact(evaluated_goal):
    store, state, runner = evaluated_goal("model")
    artifact = store.root / "workspace/method.py"
    artifact.write_text("def run(): return 1\n")
    state["progress"].update(capabilities=[{"id": "method"}], deliveries=[{
        "candidate_id": "method", "kind": "method",
        "artifacts": [{"path": "method.py", "sha256": file_hash(artifact)}],
    }])
    assert runner.evaluate(state)["criteria"]["research_plan_executed"]["passed"]
    artifact.write_text("def run(): return 999\n")
    result = runner.evaluate(state)
    assert not any(row["passed"] for row in result["criteria"].values())


def test_missing_method_delivery_does_not_hide_invalid_existing_model_sc_binding(evaluated_goal):
    store, state, runner = evaluated_goal("tool")
    receipt = store.root / "episodes/cycle_000/scientific_acceptance.json"
    receipt.write_text('{"passed":true}')
    state["progress"].update(capabilities=[{"id": "model"}], deliveries=[{
        "candidate_id": "model", "kind": "model", "artifacts": [],
        "scientific_acceptance": {
            "policy": sc.POLICY, "candidate_id": "model",
            "receipt": "episodes/cycle_000/scientific_acceptance.json", "sha256": "0" * 64,
        },
    }])
    result = runner.evaluate(state)
    assert result["scientific_acceptance_pending"] is True
    assert not any(row["passed"] for row in result["criteria"].values())


@pytest.mark.parametrize("malformed", [
    {"artifacts": []},
    {"candidate_id": None, "artifacts": []},
    {"candidate_id": "", "artifacts": []},
    {"candidate_id": "model", "artifacts": None},
    {"candidate_id": "model", "artifacts": ["invalid"]},
    {"candidate_id": "model", "artifacts": [{"path": 42, "sha256": "0" * 64}]},
    {"candidate_id": "model", "artifacts": [{"path": "report.md", "sha256": None}]},
])
def test_missing_method_with_malformed_other_delivery_fails_closed(evaluated_goal, malformed):
    _, state, runner = evaluated_goal("tool")
    state["progress"].update(capabilities=[{"id": "model"}], deliveries=[{
        "family_id": "malformed-existing-family", "kind": "model", **malformed,
    }])
    result = runner.evaluate(state)
    assert result["delivery_pending"] is True
    assert not any(row["passed"] for row in result["criteria"].values())
