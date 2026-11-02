"""A history control with the same archive and limits as training selection."""

import copy
import json

import pytest
from test_shared_history_selection import attempt, hashes

from agent.rsi import shared_goal_history as history
from agent.rsi import shared_history_selection as selection


def test_recency_and_selective_share_verified_pool_and_limits():
    observations = [
        attempt(0, {"a": 0.9, "b": -0.2}),
        attempt(1, {"a": -0.3, "b": 0.8}),
        attempt(2, {"a": -0.5, "b": -0.4}),
    ]
    smart = selection.select(observations, ["a", "b"])
    recent = selection.select(observations, ["a", "b"], policy=selection.RECENT_POLICY)
    for name in [
        "data_boundary",
        "max_utf8_bytes",
        "max_sources",
        "verified_attempts_sha256",
        "available_attempts",
        "distinct_sources",
        "target_ids",
    ]:
        assert smart[name] == recent[name]
    assert hashes(smart) == [observations[0]["source_sha256"], observations[1]["source_sha256"]]
    assert hashes(recent) == [observations[2]["source_sha256"], observations[1]["source_sha256"]]
    shared = next(row for row in recent["attempts"] if row["source_sha256"] == observations[1]["source_sha256"])
    assert shared == smart["attempts"][1]
    assert selection.select(observations, ["a", "b"], policy=selection.POLICY) == smart


def test_recency_ignores_training_score_for_order():
    observations = [
        attempt(0, {"a": 0.99}),
        attempt(1, {"a": -0.9}),
        attempt(2, {"a": None}, status="development_failed"),
    ]
    expected = hashes(selection.select(observations, ["a"], policy=selection.RECENT_POLICY))
    changed = copy.deepcopy(observations)
    for index, row in enumerate(changed):
        row["results"][0]["feedback"] = {
            "status": "tested_not_published",
            "training_only_feedback": {"mse_skill_vs_training_mean": index / 3},
        }
    assert hashes(selection.select(changed, ["a"], policy=selection.RECENT_POLICY)) == expected
    assert expected == [observations[2]["source_sha256"], observations[1]["source_sha256"]]


def test_recency_deduplicates_and_uses_last_observed_position():
    one = attempt(0, {"a": 0.8})
    two = attempt(1, {"a": 0.6})
    three = attempt(2, {"a": 0.7})
    again = attempt(3, {"a": 0.8}, code=one["source_code"])
    packet = selection.select([one, two, three, again], ["a"], policy=selection.RECENT_POLICY)
    assert hashes(packet) == [one["source_sha256"], three["source_sha256"]]
    assert packet["attempts"][0]["occurrences"] == 2


def test_recency_skips_whole_oversized_sources_without_splitting_code():
    small = attempt(0, {"a": -0.1})
    huge = attempt(1, {"a": 0.9}, code="#" + "条件" * 10000)
    packet = selection.select([small, huge], ["a"], policy=selection.RECENT_POLICY)
    assert hashes(packet) == [small["source_sha256"]]
    assert packet["attempts"][0]["source_code"] == small["source_code"]
    assert (
        next(d for d in packet["selection_decisions"] if d["source_sha256"] == huge["source_sha256"])["reason"]
        == "context_budget"
    )
    assert selection.history_bytes(packet) <= selection.MAX_HISTORY_BYTES


@pytest.mark.parametrize("policy", [selection.POLICY, selection.RECENT_POLICY])
def test_matched_policies_share_context_instructions_and_empty_option(policy):
    packet = selection.select([], ["a"], policy=policy)
    assert packet["no_source_selected"] and not packet["attempts"]
    actual = history.make_prompt("context", "develop", packet)
    other = selection.select(
        [], ["a"], policy=selection.RECENT_POLICY if policy == selection.POLICY else selection.POLICY
    )
    expected = history.make_prompt("context", "develop", other)
    assert actual.removesuffix(json.dumps(packet, ensure_ascii=False)) == expected.removesuffix(
        json.dumps(other, ensure_ascii=False)
    )
    with pytest.raises(ValueError, match="cannot enter reuse"):
        history.make_prompt("context", "reuse", packet)


def test_recency_projection_uses_all_completed_history(monkeypatch, tmp_path):
    calls = []

    def observe(root, state, selected, *, all_completed=False):
        calls.append(all_completed)
        return [attempt(0, {"a": 0.8})]

    monkeypatch.setattr(history, "_observations", observe)
    packet = history.project(tmp_path, {"cycles": []}, [{"id": "a"}], selection.RECENT_POLICY)
    assert calls == [True] and packet["policy"] == selection.RECENT_POLICY
    assert history.POLICY == "bounded_training_program_history.v2"


def test_unsupported_ranking_cannot_enter_selection():
    with pytest.raises(ValueError, match="Unsupported"):
        selection.select([], ["a"], policy="final_test_score")


@pytest.mark.asyncio
async def test_recent_control_runs_real_training_reuse_and_resume(tmp_path, monkeypatch):
    from test_shared_science_goal import prepared, read, shared, synthetic_executor

    from agent.science_goal import GoalStore, run_goal

    store, driver = prepared(tmp_path, reject_second=True, max_cycles=5, history_policy=selection.RECENT_POLICY)
    calls = synthetic_executor(monkeypatch)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert state["status"] == "budget_exhausted"
    assert [c["phase"] for c in calls] == ["develop", "reuse", "develop", "develop", "develop"]
    packet = read(store.root / "episodes/cycle_004/shared_training_history.json")
    assert packet["available_attempts"] == 3 and packet["distinct_sources"] == 1
    assert packet["policy"] == selection.RECENT_POLICY
    assert packet["target_ids"] == ["synthetic_1"]
    assert len(packet["attempts"]) == 1
    assert all("1234567" not in c["prompt"] for c in calls)
    assert "Prior paired" not in calls[1]["prompt"]
    resumed = shared.SharedScientistGoal(GoalStore(store.root), None)
    assert resumed.training_history(state, [driver.by_id["synthetic_1"]]) == driver.training_history(
        state, [driver.by_id["synthetic_1"]]
    )
