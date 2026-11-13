"""Host-owned episode epochs distinguish current rejection from unexecuted history."""
import copy
import json

import pytest
import test_analysis_report_review as cases
from test_analysis_history_episode import final_review, partial_review
from test_analysis_history_episode import offline_roles as offline_roles
from test_autonomous_research import profile
from test_research_feedback_admission import ATTEMPT, BUDGET, CAUSE, call, failed

from agent.harness import analysis_interim_review as interim
from agent.harness.autonomous_research import AutonomousResearch, create
from agent.harness.research_cli_inputs import prepare_inputs, stage_inputs
from agent.harness.research_execution_feedback import feedback_state


def test_next_host_episode_archives_request_but_not_actual_fit(tmp_path):
    old = feedback_state(failed() + call(BUDGET), tmp_path, episode_index=0)
    snapshot = copy.deepcopy(old)
    current = feedback_state([], tmp_path, old, episode_index=1)
    actual, = current["pending"]
    assert actual["training_attempt_id"] == ATTEMPT and CAUSE in actual["error"]
    assert len(current["errors"]) == 1
    history, = current["request_history"]
    assert history["episode_index"] == 0 and history["archived_at_episode"] == 1
    assert history["disposition"] == "not_executed" and history["error"] == BUDGET
    assert "superseded" not in current and old == snapshot
    assert feedback_state([], tmp_path, current, episode_index=1) == current
    changed = call("Measured revised algorithm", error=False, options={"seed": 2})
    complete = feedback_state(changed, tmp_path, current, episode_index=1)
    assert complete["errors"] == [] and complete["request_history"] == current["request_history"]


def test_current_rejection_after_other_success_still_blocks(tmp_path):
    events = failed() + call("Other measured fit", source="src/other.py", error=False) + call(BUDGET)
    current = feedback_state(events, tmp_path, episode_index=4)
    assert len(current["errors"]) == 2 and "request_history" not in current
    assert current["pending"][1]["episode_index"] == 4
    assert feedback_state(events, tmp_path, current, episode_index=4) == current


def test_old_goal_without_epoch_is_not_assumed_to_be_historical(tmp_path):
    legacy = feedback_state(failed() + call(BUDGET), tmp_path)
    first = feedback_state([], tmp_path, legacy, episode_index=7)
    assert len(first["errors"]) == 2 and "request_history" not in first
    assert first["pending"][1]["episode_index"] == 7
    next_cycle = feedback_state([], tmp_path, first, episode_index=8)
    assert len(next_cycle["errors"]) == 1
    assert next_cycle["request_history"][0]["episode_index"] == 7


def test_missing_epoch_never_implicitly_archives(tmp_path):
    current = feedback_state(call(BUDGET), tmp_path, episode_index=3)
    assert feedback_state([], tmp_path, current) == current
    assert feedback_state([], tmp_path) == {"pending": [], "errors": []}


@pytest.mark.parametrize("epoch", [-1, True, "2", 1])
def test_invalid_or_backward_epoch_fails_closed(tmp_path, epoch):
    current = feedback_state(call(BUDGET), tmp_path, episode_index=2)
    with pytest.raises(ValueError, match="episode index"):
        feedback_state([], tmp_path, current, episode_index=epoch)


def test_model_payload_cannot_supply_host_epoch(tmp_path):
    events = call(BUDGET, options={"episode_index": -100})
    events[0]["episode_index"] = -100
    events[1]["episode_index"] = -100
    state = feedback_state(events, tmp_path, episode_index=5)
    assert state["pending"][0]["episode_index"] == 5
    assert "request_history" not in state


def test_history_is_bounded_and_replay_does_not_increment_omitted(tmp_path):
    state = None
    for index in range(12):
        state = feedback_state(call(BUDGET + "x" * 3000, options={"seed": index}), tmp_path, state, episode_index=index)
    state = feedback_state([], tmp_path, state, episode_index=12)
    assert not state["errors"] and len(state["request_history"]) == 8
    assert state["request_history_omitted"] == 4
    assert all(len(row["error"]) <= 600 and row["disposition"] == "not_executed" for row in state["request_history"])
    assert feedback_state([], tmp_path, state, episode_index=12) == state


@pytest.mark.asyncio
@pytest.mark.parametrize("deny_again", [False, True])
async def test_full_sc_goal_can_finish_after_old_rejection_but_not_current_rejection(
        tmp_path, offline_roles, deny_again):
    store = create(tmp_path / "goal", profile(), "Analyze the supplied protein sequence and report the measured score.",
                   {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240})
    original = tmp_path / "original.fasta"
    original.write_text(">original\nMKWVTFISLLFLFSSAYS\n")
    with prepare_inputs([original]) as prepared:
        stage_inputs(store, prepared)
    calls, captures, reviews = [], [], []

    async def consult(role, context):
        if role in {"PI", "CB"} and len(calls) == 1:
            feedback = context["progress"]["execution_feedback"]
            assert feedback["episode_index"] == 1 and feedback["errors"] == []
            assert feedback["request_history"][0]["disposition"] == "not_executed"
        if role == "PI":
            return "Use the recorded computation to produce the requested report."
        if role == "CB":
            return json.dumps({"summary": "Analyze the sequence", "deliverable": "analysis", "needs_input": [],
                "steps": [{"id": "analyze", "title": "Analyze the sequence", "verification": "execution",
                           "acceptance": "Report the recorded sequence score."}]})
        reviews.append(copy.deepcopy(context))
        if context["scientific_acceptance"]["analysis_review_policy"] == interim.POLICY:
            return json.dumps(partial_review(context))
        return json.dumps(final_review(context, captures[0]))

    async def executor(_request, service, events):
        calls.append(True)
        if len(calls) == 1:
            task = cases.run_capture(service, {"sequence": "MKWVTFISLLFLFSSAYS"}, {"data": {"score": 1.233}})
            captures.append(task)
            events.extend(call(BUDGET))
            await service.submit_research_checkpoint("Calculation recorded; report pending", [], ["analysis/" + task + ".json"], [])
        else:
            await service.write_research_file("report.md", "Recorded score: 1.233\n")
            await service.submit_research_checkpoint("Report completed", ["analyze"],
                ["report.md", "analysis/" + captures[0] + ".json"], [])
            if deny_again:
                events.extend(call("PermissionError: Tool disabled", options={"seed": 2}))
        events.append({"kind": "completed", "text": "Saved current evidence", "event_audit": {"passed": True}})

    state = await AutonomousResearch(store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert len(calls) == 2 and len(captures) == 1 and len(reviews) == 2
    assert reviews[0]["execution"]["failures"]
    assert bool(reviews[1]["execution"]["failures"]) is deny_again
    assert state["status"] == ("budget_exhausted" if deny_again else "completed")
    assert state["progress"]["research_completed"] is (not deny_again)
    assert state["progress"]["execution_feedback"]["request_history"][0]["disposition"] == "not_executed"
    assert bool(state["progress"]["execution_feedback"]["errors"]) is deny_again
