"""One bounded continuation of an unfinished research execution, with shared budgets."""
from __future__ import annotations

from dataclasses import replace
import json


def execution_usage(events):
    """Sum separate native turns, including a turn interrupted before completion."""
    totals = dict(tool_calls=0, discovery_calls=0, checkpoint_calls=0)
    current = dict.fromkeys(totals, 0)
    observed = False
    fallback = 0
    for event in events:
        kind = event.get("kind")
        if kind == "tool_call_start":
            fallback += 1
        if kind in {"tool_usage", "completed"}:
            for key in totals:
                value = event.get(key)
                if type(value) is int and value >= 0:
                    current[key] = max(current[key], value)
                    observed = observed or key == "tool_calls"
        if kind == "completed":
            if not observed:
                current["tool_calls"] = fallback
            for key in totals:
                totals[key] += current[key]
            current = dict.fromkeys(totals, 0)
            observed = False
            fallback = 0
    if not observed:
        current["tool_calls"] = fallback
    return {key: totals[key] + current[key] for key in totals}


def discovery_only_observation(events):
    """Describe observed read-only turns without treating an attempted write as progress."""
    readonly = {
        "search_tools", "describe_tool", "list_session_files", "read_session_file",
        "list_research_files", "read_research_file", "get_research_strategy_guide",
        "get_research_review_requirements", "get_research_baseline",
    }
    calls = {}
    for event in events:
        if event.get("kind") != "tool_call_start":
            continue
        name = str(event.get("tool_name", "")).rsplit("__", 1)[-1]
        args = event.get("args") or {}
        if name == "call_tool" and isinstance(args, dict):
            name = str(args.get("name", "")).rsplit("__", 1)[-1]
            args = args.get("arguments", {})
        if name not in readonly or (name == "get_research_baseline" and isinstance(args, dict) and args.get("output_file")):
            return None
        calls[event.get("tool_call_id")] = (name, args)
    if not calls:
        return None
    described = set()
    for event in events:
        if event.get("kind") != "tool_result" or event.get("is_error"):
            continue
        name, args = calls.get(event.get("tool_call_id"), ("", {}))
        if name == "describe_tool" and isinstance(args, dict) and isinstance(args.get("name"), str):
            described.add(args["name"][:120])
    return {"discovery_only": True, "described_tools": sorted(described)[:6],
            "observation": "Only tool descriptions or existing files were requested; no code was saved or fitted.",
            "next_step": "Use the already supplied interfaces and model_development.next_action to save or repair a runnable implementation, "
                         "fit it on the verified prepared dataset, and checkpoint the measured result. Do not repeat unchanged tool descriptions or data preparation."}


def execution_followup(request, profile, events, *, checkpoint, remaining_calls, remaining_seconds):
    if checkpoint is not None or any(e.get("kind") == "interaction" for e in events):
        return None
    context = json.loads(request["text"])
    if "execution_followup" in context:
        return None
    completed = [e for e in events if e.get("kind") == "completed"]
    if not completed or not (completed[-1].get("event_audit") or {}).get("passed"):
        return None
    used = execution_usage(events)
    execution_left = profile.budget.tool_calls - used["tool_calls"]
    checkpoints_left = 2 - used["checkpoint_calls"]
    if execution_left < 1 or checkpoints_left < 1 or remaining_calls < 3 or remaining_seconds < 10:
        return None
    retry = replace(profile, budget=replace(profile.budget,
                    model_calls=min(profile.budget.model_calls, remaining_calls),
                    tool_calls=execution_left,
                    wall_seconds=min(profile.budget.wall_seconds, remaining_seconds)))
    context["execution_followup"] = {
        "observation": "The executor returned before saving a research checkpoint.",
        "completed_usage": used,
        "previous_response": str(completed[-1].get("text", ""))[-2000:],
        "next_step": "Use structured MCP tool calls, not a tool name and JSON written as ordinary text. "
                     "Continue from the verified artifacts and measured generations below; do not repeat completed work. "
                     "Use the selected interfaces directly, or describe a needed enabled tool before calling it. "
                     "Execute the next unfinished plan step, then submit_research_checkpoint with actual artifacts. "
                     "If missing input blocks useful work, use ask_user; do not assume an answer."}
    from .unattended_research import POLICY
    if request.get("execution_policy") == POLICY:
        context["execution_followup"]["next_step"] = context["execution_followup"]["next_step"].replace(
            "If missing input blocks useful work, use ask_user; do not assume an answer.",
            "If missing input blocks useful work, save partial evidence and checkpoint needs_input; do not invent an answer.")
    discovery = discovery_only_observation(events)
    if discovery is not None:
        context["execution_followup"].update(discovery)
        if context.get("plan", {}).get("deliverable") != "model":
            context["execution_followup"]["next_step"] = (
                "Use the already supplied interfaces to execute the next unfinished plan step and save its checkpoint. "
                "Do not repeat unchanged tool descriptions or completed data preparation.")
    return {**request, "profile": retry.public(), "budget": retry.public()["budget"],
            "research_allowances": {
                "discovery": max(0, min(48, 2 * profile.budget.tool_calls) - used["discovery_calls"]),
                "checkpoint": checkpoints_left},
            "text": json.dumps(context, ensure_ascii=False)}
