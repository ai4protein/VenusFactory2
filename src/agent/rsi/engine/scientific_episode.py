"""Host-authored contracts for four-role scientific episodes.

A completed episode is not a scientific acceptance decision: the outer host
must still validate the submission receipt or the source-bound review.
"""

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeContract:
    kind: str
    context_tool: str
    artifact_file: str
    required_tools: tuple[str, ...] = ()
    completion_repair_limit: int = 0


def episode_contract(capability, phase, allowed_tools):
    if phase in {"develop", "reuse"}:
        return EpisodeContract(
            "model_lifecycle",
            "inspect_model_code_task",
            "candidate.json" if phase == "develop" else "predictions.json",
        )
    if phase != "chat":
        raise ValueError("Unsupported scientific episode phase")
    path = capability / "scientific_task.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
        raise ValueError("Scientific chat requires a host-authored episode contract")
    value = json.loads(path.read_text())
    fields = {"schema", "kind", "context_tool", "artifact_file"}
    if isinstance(value, dict) and value.get("schema") == 2:
        fields |= {"required_tools", "completion_repair_limit"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Invalid scientific episode contract fields")
    if type(value["schema"]) is not int or value["schema"] not in {1, 2} or value["kind"] != "reproduction":
        raise ValueError("Unsupported scientific episode contract")
    context, artifact = value["context_tool"], value["artifact_file"]
    if (
        not isinstance(context, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", context)
        or context not in allowed_tools
    ):
        raise ValueError("Episode context tool is not admitted")
    if not isinstance(artifact, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}\.json", artifact):
        raise ValueError("Episode artifact must be a plain JSON basename")
    required = value.get("required_tools", [])
    limit = value.get("completion_repair_limit", 0)
    if (
        not isinstance(required, list)
        or any(not isinstance(x, str) for x in required)
        or len(required) != len(set(required))
        or not set(required) <= set(allowed_tools)
        or type(limit) is not int
        or limit not in {0, 1}
        or (value["schema"] == 2 and (not required or context not in required))
    ):
        raise ValueError("Invalid scientific completion obligations")
    return EpisodeContract(value["kind"], context, artifact, tuple(required), limit)


def artifact_written(workspace, contract):
    path = workspace / contract.artifact_file
    if path.is_symlink() or not path.is_file():
        return False
    try:
        # Development/review receipts are objects; native model reuse writes
        # an array of prediction rows. Scientific validation belongs to the
        # outer controller, so this check must not reject the native format.
        expected_type = (
            list if contract.kind == "model_lifecycle" and contract.artifact_file == "predictions.json" else dict
        )
        return isinstance(json.loads(path.read_text()), expected_type)
    except (OSError, ValueError):
        return False


def successful_tools(events):
    """Count only successful paired tool responses, never proposals or failures."""
    pending, completed, seen = {}, set(), set()
    for event in events:
        cid = event.get("tool_call_id")
        if event.get("kind") == "tool_call_start":
            if not isinstance(cid, str) or cid in seen:
                raise ValueError("Invalid or duplicate tool-call identity")
            seen.add(cid)
            pending[cid] = event.get("tool_name", "").removeprefix("mcp__venusfactory__")
        elif event.get("kind") == "tool_result":
            if cid not in pending:
                raise ValueError("Unpaired tool result")
            name = pending.pop(cid)
            if event.get("is_error") is not False:
                continue
            try:
                value = json.loads(event.get("output", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("success") is not False and value.get("accepted") is not False:
                completed.add(name)
    return completed


def completion_feedback(contract, events):
    completed = successful_tools(events)
    missing = sorted(set(contract.required_tools) - completed)
    if not missing:
        return None
    # A deterministic description of unmet obligations, not a generated result.
    return (
        "The scientific episode is not complete. Successfully executed tools: "
        + json.dumps(sorted(completed))
        + ". Required successful tools still missing: "
        + json.dumps(missing)
        + ". Plan and execute only the missing work using actual prior tool results. "
        "Do not repeat a successful submission, change the job identity, or rerun computation. "
        "Preserve all prior results. An output file alone does not satisfy these obligations. "
        "This is one predeclared completion-repair attempt within the original budget, not a new allocation."
    )
