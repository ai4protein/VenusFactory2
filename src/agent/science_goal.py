"""Persistent expert-team goals above bounded workflow episodes.

A graph final report completes an episode, never the goal. Only a trusted
criterion evaluator with verified artifacts can complete the goal. The store
is outside the agent workspace and has a single runner lease.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from agent.harness.contracts import Profile, digest, file_hash
from agent.rsi.engine.storage import atomic_json, file_lock


@dataclass(frozen=True)
class GoalSpec:
    objective: str
    criteria: tuple[str, ...]
    max_cycles: int = 8
    model_calls: int = 192
    tool_calls: int = 96
    wall_seconds: float = 7200

    def __post_init__(self):
        if not self.objective.strip() or not self.criteria or len(set(self.criteria)) != len(self.criteria):
            raise ValueError("An explicit objective and unique completion criteria are required")
        if any(not isinstance(x, str) or not x.strip() for x in self.criteria):
            raise ValueError("Completion criteria must be non-empty strings")
        for name in ["max_cycles", "model_calls", "tool_calls"]:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Invalid goal budget")
        if (
            type(self.wall_seconds) not in (int, float)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
        ):
            raise ValueError("Invalid goal wall budget")


class GoalStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, root, spec, profile, *, condition="science_goal", admission=None):
        # Non-product conditions require explicit experiment admission. An ordinary
        # Agent conversation never acquires a persistent goal implicitly.
        required_mode = {
            "science_goal": "science_expert",
            "persistent_agent_ablation": "science_agent",
            "scripted_agent_turns": "science_agent",
        }.get(condition)
        if required_mode is None or profile.mode != required_mode:
            raise ValueError("Science goals and explicit research controls require their declared mode")
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=False)
        definition = {"goal": asdict(spec), "profile": profile.public(), "condition": condition}
        if admission is not None:
            if not isinstance(admission, dict):
                raise ValueError("Goal admission must be a host-declared mapping")
            definition["admission"] = copy.deepcopy(admission)
        state = {
            "schema": 1,
            "definition": definition,
            "definition_sha256": digest(definition),
            "status": "active",
            "progress": {},
            "cycles": [],
            "inflight": None,
            "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": 0.0},
            "assessment": None,
            "events": [{"event": "created", "at": time.time()}],
        }
        atomic_json(store.root / "goal.json", state)
        return store

    def read(self):
        state = json.loads((self.root / "goal.json").read_text())
        if digest(state["definition"]) != state["definition_sha256"]:
            raise ValueError("Goal definition changed")
        return state

    def mutate(self, fn):
        with file_lock(self.root / ".state.lock"):
            state = self.read()
            fn(state)
            atomic_json(self.root / "goal.json", state)
            return state

    def pause(self):
        def change(state):
            if state["status"] == "active":
                state["status"] = "paused"
                state["events"].append({"event": "user_paused", "at": time.time()})

        return self.mutate(change)

    def resume(self):
        def change(state):
            if state["status"] not in {"paused", "interrupted", "waiting_for_input"}:
                raise ValueError("Goal is not resumable without a new budget/definition")
            state["status"] = "active"
            state["events"].append({"event": "user_resumed", "at": time.time()})

        return self.mutate(change)


def goal_cycle_prompt(state):
    spec = state["definition"]["goal"]
    condition = state["definition"].get("condition", "science_goal")
    common = (
        "Research objective: "
        + spec["objective"]
        + "\nCompletion criteria: "
        + json.dumps(spec["criteria"])
        + "\nVerified progress and previous episode outcomes: "
        + json.dumps(
            {
                "progress": state["progress"],
                "episodes": [{k: x.get(k) for k in ["index", "status", "summary"]} for x in state["cycles"]],
            },
            ensure_ascii=False,
        )
    )
    if condition == "science_goal":
        guidance = (
            "Persistent Science goal. The expert team must perform the next useful bounded episode. "
            "Use PI for research direction, CB for planning, MLS for implementation and SC for critical review. "
            "This episode's final report is a checkpoint, not proof that the overall goal is complete. "
            "The host independently checks goal criteria and will continue if they remain unmet."
        )
    elif condition == "persistent_agent_ablation":
        guidance = (
            "Experimental persistent single-agent control. Perform the next useful bounded episode. "
            "Choose your own sequence of reasoning and tool use. "
            "This episode's final report is a checkpoint, not proof that the overall goal is complete. "
            "The host independently checks goal criteria and will continue if they remain unmet."
        )
    elif condition == "scripted_agent_turns":
        guidance = (
            "Current conversational user turn. Choose your own sequence of reasoning and tool use "
            "to answer the current request. End this turn with your result. "
            "The research supervisor provides any later request as a separate user turn, "
            "according to the fixed experimental user-turn policy."
        )
    else:
        raise ValueError("Unknown goal experiment condition")
    return common + "\n" + guidance


def validate_assessment(store, state, assessment):
    """Evaluator is host code, not a model-selected tool or a JSON success flag."""
    checks = assessment.get("criteria")
    expected = state["definition"]["goal"]["criteria"]
    if not isinstance(checks, dict) or set(checks) != set(expected):
        raise ValueError("Goal assessment must cover every declared criterion")
    for value in checks.values():
        if not isinstance(value, dict) or type(value.get("passed")) is not bool:
            raise ValueError("Invalid criterion result")
        if value["passed"]:
            evidence = value.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("Passed goal criteria require independently saved evidence")
            for record in evidence:
                path = store.root / record["path"]
                path.resolve().relative_to(store.root)
                if path.is_symlink() or not path.is_file() or file_hash(path) != record["sha256"]:
                    raise ValueError("Goal evidence missing or changed")
    return all(x["passed"] for x in checks.values())


async def run_goal(
    store,
    run_cycle,
    evaluate,
    *,
    pause_poll_seconds=0.25,
    cycle_wall_seconds=None,
    on_checkpoint=None,
    stop_on_uncheckpointed_failure=False,
):
    """Continue until evidence completion, explicit pause/input need or budget.

    run_cycle receives (index, profile with remaining limits, checkpoint,
    goal context). It returns a host-constructed outcome with usage, summary
    and progress. Unknown/interrupted usage keeps the full reservation.
    """
    if type(stop_on_uncheckpointed_failure) is not bool:
        raise ValueError("Invalid checkpoint failure policy")
    if cycle_wall_seconds is not None and (
        type(cycle_wall_seconds) not in (int, float) or not math.isfinite(cycle_wall_seconds) or cycle_wall_seconds <= 0
    ):
        raise ValueError("Invalid scientific cycle wall limit")
    with (store.root / ".runner.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A runner already owns this goal") from exc
        state = store.read()
        if state["status"] != "active":
            return state
        if state["inflight"] is not None:

            def interrupted(s):
                s["status"] = "interrupted"
                s["cycles"].append(
                    {
                        **s["inflight"],
                        "status": "interrupted",
                        "summary": "Previous owner ended without a checkpoint; reservation retained",
                    }
                )
                s["inflight"] = None

            return store.mutate(interrupted)
        spec = GoalSpec(**{**state["definition"]["goal"], "criteria": tuple(state["definition"]["goal"]["criteria"])})
        base = Profile.from_dict(state["definition"]["profile"])
        while True:
            state = store.read()
            if state["status"] != "active":
                return state
            if on_checkpoint is not None and state["cycles"]:
                checkpoint_started = time.monotonic()
                checkpoint_error = None
                try:
                    on_checkpoint(copy.deepcopy(state))
                except Exception as exc:
                    checkpoint_error = type(exc).__name__

                def save_checkpoint(s, checkpoint_started=checkpoint_started, checkpoint_error=checkpoint_error):
                    s["usage"]["wall_seconds"] += time.monotonic() - checkpoint_started
                    if checkpoint_error:
                        if s["status"] == "active":
                            s["status"] = "interrupted"
                        s["events"].append({"event": "checkpoint_callback_error", "error_type": checkpoint_error})

                state = store.mutate(save_checkpoint)
                if checkpoint_error or state["status"] != "active":
                    return state
            assessment_started = time.monotonic()
            try:
                assessment = evaluate(copy.deepcopy(state))
                passed = validate_assessment(store, state, assessment)
            except Exception as exc:

                def evaluator_error(s, error_type=type(exc).__name__):
                    s["status"] = "interrupted"
                    s["events"].append({"event": "assessment_error", "error_type": error_type})

                return store.mutate(evaluator_error)

            def save_assessment(s, assessment_started=assessment_started, assessment=assessment, passed=passed):
                s["usage"]["wall_seconds"] += time.monotonic() - assessment_started
                s["assessment"] = assessment
                if passed and s["status"] == "active" and not s.get("pending_research_updates"):
                    s["status"] = "completed"
                    s["events"].append({"event": "goal_verified", "at": time.time()})

            state = store.mutate(save_assessment)
            if state["status"] != "active":
                return state
            remaining = {k: getattr(spec, k) - state["usage"][k] for k in state["usage"]}
            if len(state["cycles"]) >= spec.max_cycles or any(n <= 0 for n in remaining.values()):
                return store.mutate(lambda s: s.update(status="budget_exhausted"))
            budget = replace(
                base.budget,
                model_calls=min(base.budget.model_calls, remaining["model_calls"]),
                tool_calls=min(base.budget.tool_calls, remaining["tool_calls"]),
                wall_seconds=min(base.budget.wall_seconds, remaining["wall_seconds"]),
            )
            profile = replace(base, budget=budget)
            index = len(state["cycles"])
            reserve = {k: getattr(budget, k) for k in remaining}
            if cycle_wall_seconds is not None:
                # The adapter retains its inference deadline. A scientific cycle
                # can also own a bounded computation; total wall time is charged.
                reserve["wall_seconds"] = min(cycle_wall_seconds, remaining["wall_seconds"])

            def allocate(s, reserve=reserve, index=index, profile=profile):
                if s["status"] != "active":
                    return
                for key in reserve:
                    s["usage"][key] += reserve[key]
                s["inflight"] = {"index": index, "reserved": reserve, "profile": profile.public()}

            state = store.mutate(allocate)
            if state["status"] != "active":
                return state
            started = time.monotonic()
            task = asyncio.create_task(run_cycle(index, profile, copy.deepcopy(state), goal_cycle_prompt(state)))
            outcome = None
            uncheckpointed_failure = False
            recovered_cancellation = False
            failure_trigger = None
            try:
                async with asyncio.timeout(reserve["wall_seconds"]):
                    while not task.done():
                        await asyncio.wait({task}, timeout=pause_poll_seconds)
                        if store.read()["status"] != "active":
                            task.cancel()
                            break
                    outcome = await task
            except asyncio.CancelledError:
                if store.read()["status"] == "active":
                    raise
            except Exception as exc:
                uncheckpointed_failure = True
                failure_trigger = type(exc).__name__
                outcome = {"status": "failed", "summary": "Episode did not complete", "error_type": type(exc).__name__}
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
            if (outcome is None or uncheckpointed_failure) and task.done() and not task.cancelled():
                try:
                    checkpoint = task.result()
                except BaseException:
                    checkpoint = None
                if (
                    isinstance(checkpoint, dict)
                    and checkpoint.get("execution_cancelled") is True
                    and checkpoint.get("status") == "failed"
                    and isinstance(checkpoint.get("usage"), dict)
                ):
                    outcome = checkpoint
                    uncheckpointed_failure = False
                    recovered_cancellation = True
            outcome = outcome or {"status": "interrupted", "summary": "Paused during episode; reservation retained"}
            # Only the trusted runner supplies accounting. Never use prose to complete a goal.
            actual = outcome.get("usage")
            if actual is not None:
                if set(actual) != set(reserve) or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in actual.values()
                ):
                    raise ValueError("Invalid episode accounting")
                for key in ["model_calls", "tool_calls"]:
                    if type(actual[key]) is not int:
                        raise ValueError("Call accounting must use integer counts")
                actual["wall_seconds"] = max(actual["wall_seconds"], time.monotonic() - started)

            def finish(
                s,
                actual=actual,
                reserve=reserve,
                index=index,
                outcome=outcome,
                uncheckpointed_failure=uncheckpointed_failure,
                recovered_cancellation=recovered_cancellation,
                failure_trigger=failure_trigger,
            ):
                if actual is not None:
                    for key in reserve:
                        s["usage"][key] += actual[key] - reserve[key]
                s["cycles"].append({"index": index, **outcome})
                s["inflight"] = None
                if outcome.get("progress") is not None:
                    s["progress"] = outcome["progress"]
                if s["status"] == "active" and outcome.get("needs_input") and not s.get("pending_research_updates"):
                    exhausted = len(s["cycles"]) >= spec.max_cycles or any(
                        s["usage"][key] >= getattr(spec, key) for key in reserve
                    )
                    s["status"] = "budget_exhausted" if exhausted else "waiting_for_input"
                blocker = outcome.get("execution_blocker")
                if s["status"] == "active" and isinstance(blocker, dict) and blocker.get("source") == "model_gateway":
                    exhausted = len(s["cycles"]) >= spec.max_cycles or any(
                        s["usage"][key] >= getattr(spec, key) for key in reserve)
                    s["status"] = "budget_exhausted" if exhausted else "interrupted"
                    s["events"].append({"event": "model_service_unavailable", "index": index,
                                        "code": blocker["code"], "http_status": blocker["http_status"]})
                if stop_on_uncheckpointed_failure and uncheckpointed_failure and s["status"] == "active":
                    # A missing executor checkpoint cannot support later publication.
                    # Preserve its reservation and stop before spending on more cycles.
                    s["status"] = "interrupted"
                    s["events"].append(
                        {
                            "event": "uncheckpointed_cycle_failure",
                            "index": index,
                            "error_type": outcome["error_type"],
                        }
                    )
                if recovered_cancellation:
                    s["events"].append(
                        {
                            "event": "cancelled_cycle_checkpoint_saved",
                            "index": index,
                            "trigger_error_type": failure_trigger,
                        }
                    )
                if actual:
                    count_overrun = any(actual[k] > reserve[k] for k in ("model_calls", "tool_calls"))
                    # Cleanup may exceed a cycle deadline; it is charged to the
                    # unchanged overall wall budget, never granted free time.
                    wall_overrun = actual["wall_seconds"] > reserve["wall_seconds"]
                    total_overrun = any(s["usage"][k] > getattr(spec, k) for k in reserve)
                    if count_overrun or total_overrun or (wall_overrun and not recovered_cancellation):
                        s["status"] = "budget_exhausted"

            state = store.mutate(finish)
            atomic_json(store.root / f"cycle_{index:03d}.json", state["cycles"][-1])
