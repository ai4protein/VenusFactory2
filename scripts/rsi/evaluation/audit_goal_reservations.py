"""Read-only proof of retained budgets when an outer goal deadline cancels a cycle."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import math

from scripts.rsi.cases.audit_reap_generated import read, require

from agent.harness.contracts import digest
from agent.science_goal import goal_cycle_prompt


def retained_timeout_reservation(root, state, cycle):
    """Missing final receipts are valid only for a bound, explicitly failed timeout.

    Return the allocation debit, never an estimate of actual calls. A normal
    episode returns None and remains subject to exact finalized usage checks.
    """
    if "usage" in cycle:
        return None
    index = cycle["index"]
    require(
        cycle
        == {"index": index, "status": "failed", "summary": "Episode did not complete", "error_type": "TimeoutError"},
        "Missing usage is not a recognized outer timeout",
    )
    episode = root / "episodes" / f"cycle_{index:03d}"
    require(not (episode / "outcome.json").exists(), "Timeout unexpectedly has a finalized outcome")
    turn = read(root / "turns" / f"turn_{index:03d}.json")
    checkpoint = turn["checkpoint"]
    prior_progress = {}
    for previous in state["cycles"][:index]:
        if "progress" in previous:
            prior_progress = previous["progress"]
    require(
        turn["index"] == index
        and checkpoint["definition"] == state["definition"]
        and checkpoint["cycles"] == state["cycles"][:index]
        and checkpoint["progress"] == prior_progress,
        "Timeout continuation checkpoint changed",
    )
    require(
        turn["context"] == goal_cycle_prompt(checkpoint) and digest(turn["context"]) == turn["context_sha256"],
        "Timeout continuation prompt changed",
    )
    profile = turn["episode_profile"]
    reserve = checkpoint["inflight"]["reserved"]
    require(set(reserve) == {"model_calls", "tool_calls", "wall_seconds"}, "Timeout reservation fields changed")
    require(
        all(type(reserve[k]) is int and reserve[k] > 0 for k in ["model_calls", "tool_calls"])
        and type(reserve["wall_seconds"]) in {int, float}
        and math.isfinite(reserve["wall_seconds"])
        and reserve["wall_seconds"] > 0,
        "Timeout reservation is invalid",
    )
    require(
        reserve == {k: profile["budget"][k] for k in reserve}
        and checkpoint["inflight"] == {"index": index, "reserved": reserve, "profile": profile},
        "Timeout reservation differs from allocated profile",
    )
    lock = read(episode / "execution/profile.lock.json")
    require(
        lock["profile"] == profile and digest({k: v for k, v in lock.items() if k != "sha256"}) == lock["sha256"],
        "Timeout profile lock changed",
    )
    return reserve
