import asyncio
import json

import pytest

from agent.harness import Budget, ModelBinding, Profile, Task
from agent.harness.contracts import file_hash
from agent.science_goal import GoalSpec, GoalStore, goal_cycle_prompt, run_goal


def profile(mode="science_expert"):
    return Profile(
        mode,
        "langgraph",
        ModelBinding("test", "model", "https://example.org/v1"),
        Budget(wall_seconds=2, model_calls=3, tool_calls=2),
    )


def create(tmp_path, **kwargs):
    return GoalStore.create(
        tmp_path / "goal", GoalSpec("Test a persistent scientific goal", ("accepted", "reused"), **kwargs), profile()
    )


def assessment(store, state):
    checks = {}
    for name in state["definition"]["goal"]["criteria"]:
        exists = (store.root / (name + ".json")).exists()
        checks[name] = {
            "passed": exists,
            "evidence": [{"path": name + ".json", "sha256": file_hash(store.root / (name + ".json"))}]
            if exists
            else [],
        }
    return {"criteria": checks}


def outcome(summary="episode complete"):
    return {
        "status": "completed",
        "summary": summary,
        "usage": {"model_calls": 1, "tool_calls": 1, "wall_seconds": 0.0},
    }


@pytest.mark.asyncio
async def test_report_does_not_end_goal_two_cycles_accumulate_verified_progress(tmp_path):
    store = create(tmp_path)
    seen = []

    async def cycle(index, p, checkpoint, prompt):
        seen.append((index, checkpoint["usage"]["model_calls"], prompt))
        (store.root / (["accepted", "reused"][index] + ".json")).write_text('{"passed":true}')
        return {**outcome("The expert team finished this episode"), "progress": {"last_episode": index}}

    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert state["status"] == "completed"
    assert len(seen) == 2 and len(state["cycles"]) == 2
    assert state["usage"]["model_calls"] == 2
    assert "last_episode" in seen[1][2]
    assert GoalStore(store.root).read()["progress"] == {"last_episode": 1}


@pytest.mark.asyncio
async def test_self_reported_success_cannot_complete_unmet_goal(tmp_path):
    store = create(tmp_path, max_cycles=2)

    async def cycle(*args):
        return {**outcome("All science is solved"), "goal_completed": True}

    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert state["status"] == "budget_exhausted"
    assert len(state["cycles"]) == 2 and not state["assessment"]["criteria"]["accepted"]["passed"]


@pytest.mark.asyncio
async def test_paused_goal_reopens_without_repeating_finished_cycle(tmp_path):
    store = create(tmp_path)
    calls = []

    async def cycle(index, *args):
        calls.append(index)
        (store.root / (["accepted", "reused"][index] + ".json")).write_text("{}")
        if index == 0:
            store.pause()
        return outcome()

    first = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert first["status"] == "paused" and calls == [0]
    assert (await run_goal(GoalStore(store.root), cycle, lambda s: assessment(store, s)))["status"] == "paused"
    store.resume()
    final = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert final["status"] == "completed" and calls == [0, 1]


@pytest.mark.asyncio
async def test_pause_cancels_inflight_and_conservatively_keeps_budget(tmp_path):
    store = create(tmp_path)
    started = asyncio.Event()
    ended = asyncio.Event()

    async def cycle(*args):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            ended.set()

    task = asyncio.create_task(run_goal(store, cycle, lambda s: assessment(store, s), pause_poll_seconds=0.01))
    await started.wait()
    store.pause()
    state = await asyncio.wait_for(task, 1)
    assert ended.is_set() and state["status"] == "paused"
    assert state["usage"]["model_calls"] == profile().budget.model_calls
    assert state["cycles"][0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_remaining_budget_caps_next_episode_not_reset_each_turn(tmp_path):
    store = create(tmp_path, model_calls=4, max_cycles=5)
    limits = []

    async def cycle(i, p, *args):
        limits.append(p.budget.model_calls)
        return {
            "status": "completed",
            "usage": {"model_calls": p.budget.model_calls, "tool_calls": 0, "wall_seconds": 0},
        }

    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert limits == [3, 1] and state["status"] == "budget_exhausted"
    assert state["usage"]["model_calls"] == 4


@pytest.mark.asyncio
async def test_forged_or_missing_completion_artifact_interrupts(tmp_path):
    store = create(tmp_path)

    async def forbidden(*args):
        raise AssertionError("must not run")

    def bad(s):
        return {
            "criteria": {
                name: {"passed": True, "evidence": [{"path": "missing", "sha256": "0" * 64}]}
                for name in s["definition"]["goal"]["criteria"]
            }
        }

    state = await run_goal(store, forbidden, bad)
    assert state["status"] == "interrupted" and not state["cycles"]


@pytest.mark.asyncio
async def test_single_owner_and_crash_checkpoint_do_not_duplicate_work(tmp_path):
    store = create(tmp_path)
    started = asyncio.Event()

    async def cycle(*args):
        started.set()
        await asyncio.sleep(30)

    first = asyncio.create_task(run_goal(store, cycle, lambda s: assessment(store, s)))
    await started.wait()
    with pytest.raises(RuntimeError, match="already owns"):
        await run_goal(store, cycle, lambda s: assessment(store, s))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert state["status"] == "interrupted" and len(state["cycles"]) == 1
    assert state["usage"]["model_calls"] == 3


def test_goal_definition_is_explicit_and_agent_turn_does_not_create_one(tmp_path):
    with pytest.raises(ValueError, match="Science"):
        GoalStore.create(tmp_path / "wrong", GoalSpec("Goal", ("result",)), profile("science_agent"))
    for bad in [{"criteria": ()}, {"max_cycles": True}, {"wall_seconds": float("inf")}]:
        with pytest.raises(ValueError):
            GoalSpec(**{**{"objective": "Goal", "criteria": ("result",)}, **bad})
    store = create(tmp_path)
    state = store.read()
    state["definition"]["goal"]["objective"] = "replaced"
    (store.root / "goal.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="changed"):
        store.read()


def test_agent_chat_allows_zero_tools_but_experiment_still_requires_them(tmp_path):
    Task(tmp_path, tmp_path, tmp_path / "out", "Question?", "Be helpful", frozenset(), phase="chat")
    with pytest.raises(ValueError):
        Task(tmp_path, tmp_path, tmp_path / "out", "Task", "Instructions", frozenset(), phase="develop")


@pytest.mark.asyncio
async def test_long_compute_has_separate_cycle_deadline_and_retains_inference_limit(tmp_path):
    from dataclasses import replace

    base = profile()
    base = replace(base, budget=replace(base.budget, wall_seconds=0.03))
    store = GoalStore.create(
        tmp_path / "long", GoalSpec("Wait for bounded computation", ("done",), wall_seconds=2), base
    )

    async def cycle(index, selected, state, context):
        assert selected.budget.wall_seconds == 0.03
        assert state["inflight"]["reserved"]["wall_seconds"] == 1
        await asyncio.sleep(0.08)
        (store.root / "done.json").write_text("{}")
        return outcome()

    state = await run_goal(store, cycle, lambda s: assessment(store, s), cycle_wall_seconds=1)
    assert state["status"] == "completed" and len(state["cycles"]) == 1
    assert state["usage"]["wall_seconds"] >= 0.08
    assert state["usage"]["model_calls"] == 1


@pytest.mark.asyncio
async def test_extended_compute_never_escapes_total_goal_budget(tmp_path):
    store = create(tmp_path, wall_seconds=0.08)
    stopped = asyncio.Event()

    async def cycle(index, selected, state, context):
        assert state["inflight"]["reserved"]["wall_seconds"] <= 0.08
        try:
            await asyncio.sleep(5)
        finally:
            stopped.set()

    state = await run_goal(store, cycle, lambda s: assessment(store, s), cycle_wall_seconds=100)
    assert state["status"] == "budget_exhausted" and stopped.is_set()
    assert state["usage"]["model_calls"] == profile().budget.model_calls
    assert state["cycles"][0]["error_type"] == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, -1, 0, float("inf"), float("nan"), "1"])
async def test_scientific_cycle_deadline_must_be_finite_and_explicit(tmp_path, value):
    store = create(tmp_path)
    with pytest.raises(ValueError, match="scientific cycle"):
        await run_goal(store, None, None, cycle_wall_seconds=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("limits,expected", [
    ({"max_cycles": 1}, "budget_exhausted"),
    ({"max_cycles": 4, "model_calls": 1}, "budget_exhausted"),
    ({"max_cycles": 4, "tool_calls": 1}, "budget_exhausted"),
    ({"max_cycles": 4}, "waiting_for_input"),
])
async def test_user_question_cannot_mask_an_exhausted_goal(tmp_path, limits, expected):
    store = create(tmp_path, **limits)

    async def cycle(*args):
        return {
            **outcome("Partial results retained"),
            "needs_input": True,
            "progress": {"questions": ["Supply additional measurements"], "artifact": "partial.json"},
        }

    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert state["status"] == expected
    assert state["progress"]["questions"] == ["Supply additional measurements"]
    assert len(state["cycles"]) == 1
    assert state["usage"]["model_calls"] == 1

@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", [
    {"source": "model_gateway", "code": "quota", "http_status": 403},
    {"source": "research_progress", "code": "unchanged_training_failure"},
])
async def test_execution_interruption_saves_budget_and_resumes_without_repeating(tmp_path, blocker):
    store = create(tmp_path, max_cycles=4)
    seen = []
    async def cycle(index, *args):
        seen.append(index)
        if index == 0:
            return {**outcome("Execution interrupted"), "status": "failed",
                    "execution_blocker": blocker, "progress": {"execution_blocker": blocker}}
        (store.root / "accepted.json").write_text("{}")
        (store.root / "reused.json").write_text("{}")
        return {**outcome(), "progress": {}}
    first = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert first["status"] == "interrupted" and seen == [0]
    assert first["usage"]["model_calls"] == 1
    saved = (store.root / "cycle_000.json").read_bytes()
    store.resume()
    final = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert final["status"] == "completed" and seen == [0, 1]
    assert final["usage"]["model_calls"] == 2
    assert (store.root / "cycle_000.json").read_bytes() == saved
    assert final["progress"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", [
    {"source": "model_gateway", "code": "quota", "http_status": 403},
    {"source": "research_progress", "code": "unchanged_training_failure"},
])
async def test_execution_interruption_cannot_extend_an_exhausted_budget(tmp_path, blocker):
    store = create(tmp_path, max_cycles=1)
    async def cycle(*args):
        return {**outcome(), "status": "failed",
                "execution_blocker": blocker}
    final = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert final["status"] == "budget_exhausted"
    with pytest.raises(ValueError):
        store.resume()


@pytest.mark.asyncio
async def test_host_rejection_reaches_next_episode_without_validation_rows(tmp_path):
    store = create(tmp_path, max_cycles=2)
    prompts = []

    async def cycle(index, selected, checkpoint, prompt):
        prompts.append(prompt)
        return {
            **outcome("Candidate failed independent development acceptance"),
            "status": "candidate_rejected",
            "rejection_reasons": ["prediction_not_repeatable", "PRIVATE_REASON"],
            "validation": [{"id": "PRIVATE_ROW", "target": 987654321}],
        }

    state = await run_goal(store, cycle, lambda s: assessment(store, s))
    assert state["status"] == "budget_exhausted" and len(prompts) == 2
    assert "prediction_not_repeatable" not in prompts[0]
    assert "prediction_not_repeatable" in prompts[1]
    assert all(value not in prompts[1] for value in ("PRIVATE_ROW", "PRIVATE_REASON", "987654321"))


def test_shared_rejection_context_is_bounded_and_omits_raw_assessment(tmp_path):
    state = create(tmp_path).read()
    state["cycles"] = [{
        "index": 0, "status": "completed", "summary": "Shared development finished",
        "target_outcomes": {
            **{f"task_{i}": {
                "status": "candidate_rejected", "rejection_reasons": ["performance_threshold"],
                "validation": "PRIVATE_VALIDATION", "metrics": {"score": "PRIVATE_SCORE"},
            } for i in range(12)},
            "unknown": {"rejection_reasons": ["PRIVATE_REASON"]},
            "malformed": {"rejection_reasons": "prediction_not_repeatable"},
            "missing": None,
        },
    }]
    prompt = goal_cycle_prompt(state)
    context = json.JSONDecoder().raw_decode(prompt.split("previous episode outcomes: ", 1)[1])[0]
    episode = context["episodes"][0]
    assert len(episode["target_rejections"]) == 8
    assert episode["omitted_target_rejections"] == 4
    assert all(item["rejection_reasons"] == ["performance_threshold"] for item in episode["target_rejections"])
    assert all(value not in prompt for value in ("PRIVATE_VALIDATION", "PRIVATE_SCORE", "PRIVATE_REASON", "prediction_not_repeatable"))


def test_legacy_cycle_context_keeps_its_original_prompt(tmp_path):
    state = create(tmp_path).read()
    cycle = {"index": 0, "status": "completed", "summary": "An archived episode"}
    state["cycles"] = [{**cycle, "target_outcomes": {"task": {"status": "candidate_rejected"}}}]
    prompt = goal_cycle_prompt(state)
    assert json.dumps({"progress": {}, "episodes": [cycle]}, ensure_ascii=False) in prompt
    assert "rejection_reasons" not in prompt
