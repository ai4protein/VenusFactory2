"""Freeze a tool contract, strategy and discovery decision into new episodes."""

import json

from agent.harness.scientist_policy import FILENAME, validate_policy

from .paper_step import digest
from .paper_step_sessions import read
from .research_tool_contract import validate_contract
from .storage import atomic_json
from .versions import VersionStore


def validate_context(value, definition):
    if not isinstance(value, dict) or set(value) != {"contract", "policy", "discovery"}:
        raise ValueError("Invalid Scientist tool context")
    validate_contract(value["contract"], definition)
    validate_policy(value["policy"])
    discovery = value["discovery"]
    if not isinstance(discovery, dict) or set(discovery) != {"library", "matches", "action", "selected_entry"}:
        raise ValueError("A frozen capability search is required")
    if discovery["action"] not in {"create", "reuse"}:
        raise ValueError("Unsupported capability action")
    ids = discovery["matches"]
    if (
        not isinstance(ids, list)
        or len(ids) != len(set(ids))
        or any(not isinstance(i, str) or len(i) != 64 or any(c not in "0123456789abcdef" for c in i) for i in ids)
    ):
        raise ValueError("Invalid discovered capability IDs")
    if discovery["action"] == "create" and (ids or discovery["selected_entry"] is not None):
        raise ValueError("Existing compatible capabilities must be reused")
    if discovery["action"] == "reuse" and discovery["selected_entry"] not in ids:
        raise ValueError("Reuse must select a discovered capability")
    return value


def attach_context(folder, value, definition):
    """Called before any runtime request; leave legacy episodes byte-compatible."""
    validate_context(value, definition)
    source = folder / "source"
    task = read(source / "step_task.json")
    task["public_task"]["research_tool_contract"] = value["contract"]
    task["public_task"]["capability_discovery"] = value["discovery"]
    atomic_json(source / "step_task.json", task)
    atomic_json(source / FILENAME, value["policy"])
    manifest = VersionStore(folder / "bundles").snapshot(
        source, metadata={"kind": "scientist_research_tool", "context_sha256": digest(value)}
    )
    declaration = read(folder / "phase.json")
    declaration["capability_version"] = manifest["version_id"]
    declaration["prompt"] = (
        "Develop and accumulate a reusable protein-research capability for the parent objective. "
        "Use the declared interface, scientific scope and library search below. "
        "A completed subtask does not establish improvement of the parent research objective.\n"
        + json.dumps({"contract": value["contract"], "capability_discovery": value["discovery"]}, ensure_ascii=False)
        + "\n\n"
        + declaration["prompt"]
    )
    declaration["instructions"] += "\nThe research-tool contract and capability search are in inspect_paper_step_task."
    atomic_json(folder / "phase.json", declaration)


def verify_context(folder, value, definition):
    from agent.harness.scientist_policy import load_policy

    from .paper_step_sessions import phase_task
    from .paper_step_tool_service import PaperStepTools
    from .research_tool_contract import validate_result

    validate_context(value, definition)
    task = phase_task(folder)
    service = PaperStepTools(task.workspace, task.capability)
    public = service.task()["public_task"]
    if (
        load_policy(task.capability) != value["policy"]
        or public.get("research_tool_contract") != value["contract"]
        or public.get("capability_discovery") != value["discovery"]
    ):
        raise ValueError("Episode changed its admitted research-tool context")
    if service.task()["phase"] == "develop":
        candidate = service.read(task.workspace, "step_candidate.json")
        attempt = service.attempt(candidate["development_attempt"])
        for case, result in zip(public["development_cases"], attempt["results"], strict=True):
            validate_result(value["contract"], case["inputs"], result["output"])
    else:
        record = service.read(task.workspace, "step_reuse.json")
        validate_result(value["contract"], service.reuse_inputs(), record["output"])
