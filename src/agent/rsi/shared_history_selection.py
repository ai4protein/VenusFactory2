"""Select complementary whole programs from verified same-goal training evidence.

This module receives paired training-tool observations, not controller results.
Selection is a proposal for the next development episode, never acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math

from .protein_model import digest
from .training_feedback_view import compact_feedback

POLICY = "selective_training_program_history.v1"
RECENT_POLICY = "recent_distinct_training_program_history.v1"
MAX_HISTORY_BYTES = 48 * 1024
MAX_SOURCES = 2
MAX_DECISIONS = 64
SKILL_METRICS = ("mse_skill_vs_training_mean", "brier_skill_vs_training_prior")


def require(value, message):
    if not value:
        raise ValueError(message)


def history_bytes(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _skill(feedback):
    if feedback.get("status") != "tested_not_published":
        return None
    metrics = feedback.get("training_only_feedback", {})
    if not isinstance(metrics, dict):
        return None
    available = [(name, metrics[name]) for name in SKILL_METRICS if name in metrics]
    if len(available) != 1:
        return None
    name, value = available[0]
    if type(value) not in (int, float) or not math.isfinite(value) or value > 1:
        return None
    return name, value


def _candidates(observations, task_ids):
    archive = {}
    previous_episode = -1
    for position, attempt in enumerate(observations):
        episode = attempt["episode"]
        require(type(episode) is int and episode >= previous_episode, "Invalid chronological training history")
        previous_episode = episode
        code = attempt["source_code"]
        require(isinstance(code, str), "Missing historical source")
        identity = hashlib.sha256(code.encode()).hexdigest()
        require(identity == attempt["source_sha256"], "Historical source hash changed")
        group = archive.setdefault(identity, {"observations": [], "by_task": {}, "position": position})
        group["observations"].append(attempt)
        group["position"] = position
        seen = set()
        for row in attempt["results"]:
            task_id = row["task_id"]
            require(task_id in task_ids and task_id not in seen, "Invalid historical target identity")
            seen.add(task_id)
            group["by_task"].setdefault(task_id, []).append((episode, row["feedback"]))
    candidates = []
    for identity, group in archive.items():
        last = group["observations"][-1]
        skills, results = {}, []
        for task_id in task_ids:
            evidence = group["by_task"].get(task_id, [])
            if not evidence:
                continue
            values = [_skill(feedback) for _, feedback in evidence]
            # Repeated execution cannot improve selection merely by cherry-picking
            # its best score. Missing/failed feedback invalidates this task's utility.
            if all(value is not None for value in values) and len({v[0] for v in values}) == 1:
                skills[task_id] = {"metric": values[0][0], "value": min(v[1] for v in values)}
            episode, feedback = evidence[-1]
            results.append(
                {
                    "task_id": task_id,
                    "episode": episode,
                    "feedback_sha256": digest(feedback),
                    "feedback": compact_feedback(feedback),
                }
            )
        candidates.append(
            {
                "position": group["position"],
                "source_sha256": identity,
                "episode": last["episode"],
                "source_code": last["source_code"],
                "rationale": last["rationale"],
                "occurrences": len(group["observations"]),
                "observations_sha256": digest(group["observations"]),
                "task_training_skills": skills,
                "results": results,
            }
        )
    return candidates


def select(observations, task_ids, *, policy=POLICY):
    """Greedy gain in positive per-task training skill, with an empty option.

    Each task has equal weight. Utility is zero for failed, unknown or nonpositive
    skill, and the best selected positive skill otherwise. This is a bounded
    prompt-selection heuristic, not an estimate of future model improvement.
    """
    require(policy in (POLICY, RECENT_POLICY), "Unsupported history selection policy")
    task_ids = list(task_ids)
    require(1 <= len(task_ids) <= 32 and len(set(task_ids)) == len(task_ids), "Invalid selective-history targets")
    require(all(isinstance(name, str) and name for name in task_ids), "Invalid selective-history target names")
    candidates = _candidates(observations, set(task_ids))
    # Avoid set iteration affecting serialized task order or replay hashes.
    for item in candidates:
        item["results"].sort(key=lambda row: task_ids.index(row["task_id"]))
        item["task_training_skills"] = {
            t: item["task_training_skills"][t] for t in task_ids if t in item["task_training_skills"]
        }
    selected, gains, excluded = [], {}, {}
    utility = dict.fromkeys(task_ids, 0.0)

    def positive(item, task):
        return max(0.0, item["task_training_skills"].get(task, {}).get("value", 0.0))

    def gain(item):
        return sum(max(0.0, positive(item, task) - utility[task]) for task in task_ids) / len(task_ids)

    def packet(chosen, marginal_gains):
        identities = {item["source_sha256"] for item in chosen}
        decisions = []
        for item in candidates:
            identity = item["source_sha256"]
            reason = (
                "selected"
                if identity in identities
                else excluded.get(identity)
                or (
                    "no_positive_training_evidence"
                    if policy == POLICY and not any(positive(item, t) for t in task_ids)
                    else "source_limit"
                    if len(chosen) >= MAX_SOURCES
                    else "no_incremental_training_coverage"
                )
            )
            decisions.append(
                {
                    "episode": item["episode"],
                    "source_sha256": identity,
                    "occurrences": item["occurrences"],
                    "observations_sha256": item["observations_sha256"],
                    "task_training_skills": item["task_training_skills"],
                    "reason": reason,
                }
            )
        # Complete evidence remains in paired local records, even when prompt
        # decision references exceed the separate fixed metadata capacity.
        references = [
            {
                **{k: v for k, v in d.items() if k != "task_training_skills"},
                "task_training_skills_sha256": digest(d["task_training_skills"]),
                "positive_tasks": sum(v["value"] > 0 for v in d["task_training_skills"].values()),
            }
            for d in decisions
        ]
        visible = [d for d in references if d["source_sha256"] in identities]
        rest = [d for d in references if d["source_sha256"] not in identities]
        visible += rest[-max(0, MAX_DECISIONS - len(visible)) :] if len(visible) < MAX_DECISIONS else []
        return {
            "policy": policy,
            "data_boundary": "paired_training_tools_only",
            "max_utf8_bytes": MAX_HISTORY_BYTES,
            "max_sources": MAX_SOURCES,
            "verified_attempts_sha256": digest(observations),
            "available_attempts": len(observations),
            "distinct_sources": len(candidates),
            "target_ids": task_ids,
            "selection_rule": (
                "Greedy marginal positive training-skill coverage; conservative repeated-source scores; empty source selection allowed"
                if policy == POLICY
                else "Newest distinct source programs from the same cumulative verified pool; no performance ranking"
            ),
            "attempts": [{k: v for k, v in item.items() if k != "position"} for item in chosen],
            "marginal_training_gains": marginal_gains,
            "selection_decisions": visible,
            "all_decisions_sha256": digest(decisions),
            "omitted_decision_records": len(decisions) - len(visible),
            "no_source_selected": not chosen,
        }

    require(history_bytes(packet([], {})) <= MAX_HISTORY_BYTES, "Selective history metadata exceeds prompt budget")
    while len(selected) < MAX_SOURCES:
        selected_ids = {item["source_sha256"] for item in selected}
        remaining = [
            item
            for item in candidates
            if item["source_sha256"] not in selected_ids | set(excluded) and (policy == RECENT_POLICY or gain(item) > 0)
        ]
        if not remaining:
            break
        candidate = min(
            remaining,
            key=lambda item: (
                (-gain(item), item["source_sha256"]) if policy == POLICY else (-item["position"], item["source_sha256"])
            ),
        )
        identity = candidate["source_sha256"]
        proposal = [*selected, candidate]
        proposed_gains = {**gains, identity: gain(candidate)}
        if history_bytes(packet(proposal, proposed_gains)) > MAX_HISTORY_BYTES:
            excluded[identity] = "context_budget"
            continue
        selected, gains = proposal, proposed_gains
        utility = {task: max(utility[task], positive(candidate, task)) for task in task_ids}
    result = packet(selected, gains)
    require(history_bytes(result) <= MAX_HISTORY_BYTES, "Selective history exceeds prompt budget")
    return result
