"""Host validation of shared-program MCP receipts and per-target evidence.

These contracts verify performed operations. Scientific acceptance and proof of
native autonomous execution remain independent controller/publication steps.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .development_handoff import description, receipt_fields
from .protein_model import digest
from .shared_model_display import FIELD as DISPLAY_FIELD
from .shared_model_display import POLICIES as DISPLAY_POLICIES
from .shared_model_display import envelope
from .storage import sha256_file

SCHEMA = "shared_model_program.v1"
LEDGER = ".shared_model_development.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(
        path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
        "Missing or symlinked shared evidence",
    )
    value = json.loads(path.read_text())
    digest(value)
    return value


def history_path(root, prefix, value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid ledger history hash")
    return root / (prefix + value + ".json")


def bundle(capability):
    cap = Path(capability)
    task = read(cap / "task.json")
    fields = {"schema", "phase", "batch_id", "targets", "implementation"} | (
        {"development_handoff"} if task.get("phase") == "develop" else set()
    )
    if DISPLAY_FIELD in task:
        require(task[DISPLAY_FIELD] in DISPLAY_POLICIES, "Unsupported shared tool presentation")
        fields.add(DISPLAY_FIELD)
    require(
        set(task) == fields and task["schema"] == SCHEMA and task["phase"] in {"develop", "reuse"},
        "Invalid shared capability",
    )
    require(isinstance(task["targets"], list) and 1 <= len(task["targets"]) <= 32, "Invalid shared target count")
    ids = [x["id"] for x in task["targets"]]
    require(
        len(set(ids)) == len(ids)
        and all(isinstance(x, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,119}", x) for x in ids),
        "Invalid shared target identity",
    )
    implementation = task["implementation"]
    require(set(implementation) == {p.name for p in cap.glob("*.py")}, "Shared implementation coverage changed")
    for name, expected in implementation.items():
        path = cap / name
        require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected, "Shared implementation changed"
        )
    require(
        task["batch_id"] == digest({"phase": task["phase"], "targets": task["targets"]}), "Shared batch binding changed"
    )
    children = {}
    for target in task["targets"]:
        require(set(target) == {"id", "task_sha256"}, "Invalid shared target fields")
        child = read(cap / "targets" / target["id"] / "task.json")
        require(digest(child) == target["task_sha256"] and child["phase"] == task["phase"], "Shared target changed")
        require(
            child["shared_implementation_sha256"] == digest(implementation), "Target lost shared implementation binding"
        )
        children[target["id"]] = child
    return task, children


def _same_prefix(before, after):
    require(len(before["attempts"]) <= len(after["attempts"]), "Shared history was truncated")
    for first, second in zip(before["attempts"], after["attempts"]):
        for key in ["number", "source_code", "source_sha256", "rationale"]:
            require(first[key] == second[key], "Shared attempt history changed")
        if first["status"] == "completed":
            require(first == second, "Completed shared history changed")
    if before["submission"] is not None:
        require(after["submission"] is not None, "Shared submission disappeared")
        for key in ["source_sha256", "rationale"]:
            require(before["submission"][key] == after["submission"][key], "Shared submission changed")
        if before["submission"]["status"] == "completed":
            require(before["submission"] == after["submission"], "Completed shared submission changed")


def _state(workspace, task, state_hash=None):
    root = Path(workspace)
    require(not (root / (LEDGER + ".writing")).exists(), "Unresolved shared ledger write")
    current = read(root / LEDGER)
    state = read(history_path(root, "shared_ledger_", state_hash)) if state_hash is not None else current
    require(set(state) == {"schema", "task_sha256", "attempts", "submission"}, "Invalid shared ledger fields")
    require(state["schema"] == SCHEMA and state["task_sha256"] == digest(task), "Shared ledger binding changed")
    require(read(root / ("shared_ledger_" + digest(state) + ".json")) == state, "Shared ledger history changed")
    if state_hash is not None:
        require(digest(state) == state_hash, "Invalid shared ledger observation")
        _same_prefix(state, current)
    require(isinstance(state["attempts"], list) and len(state["attempts"]) <= 2, "Shared development budget changed")
    ids = [t["id"] for t in task["targets"]]
    for number, item in enumerate(state["attempts"], 1):
        require(
            item["number"] == number and item["status"] in {"reserved", "completed"}, "Shared attempt order changed"
        )
        require(
            hashlib.sha256(item["source_code"].encode()).hexdigest() == item["source_sha256"], "Shared source differs"
        )
        if item["status"] == "reserved":
            require(number == len(state["attempts"]), "Shared attempt followed unresolved work")
            continue
        path = root / f"shared_attempt_{number}.json"
        record = read(path)
        require(sha256_file(path) == item["artifact_sha256"], "Shared feedback changed")
        require(
            record["source_sha256"] == item["source_sha256"] and record["rationale"] == item["rationale"],
            "Shared feedback source changed",
        )
        require([x["task_id"] for x in record["results"]] == ids, "Shared feedback target coverage changed")
        for result in record["results"]:
            history = read(history_path(root / "targets" / result["task_id"], "code_ledger_", result["ledger_sha256"]))
            require(
                digest(history) == result["ledger_sha256"] and len(history["attempts"]) == number,
                "Target history changed",
            )
            for common, target in zip(state["attempts"][:number], history["attempts"]):
                require(target["status"] == "completed", "Target evaluation remains incomplete")
                for key in ["number", "source_code", "source_sha256", "rationale"]:
                    require(common[key] == target[key], "Target did not evaluate the shared program")
        expected = {
            "schema": SCHEMA,
            "source_sha256": item["source_sha256"],
            "attempts_remaining": 2 - number,
            "status": "tested_not_published"
            if any(x["result"]["status"] == "tested_not_published" for x in record["results"])
            else "development_failed",
            "results": record["results"],
        }
        require(record["response"] == expected, "Shared feedback receipt differs")
    if state_hash is None:
        expected = {f"shared_attempt_{x['number']}.json" for x in state["attempts"] if x["status"] == "completed"}
        pending = {f"shared_attempt_{x['number']}.json" for x in state["attempts"] if x["status"] == "reserved"}
        actual = {p.name for p in root.glob("shared_attempt_*.json")}
        require(expected <= actual and not actual - expected - pending, "Shared feedback artifact coverage changed")
    submission = state["submission"]
    if submission:
        require(submission["status"] in {"reserved", "completed"}, "Invalid shared submission status")
        if submission["status"] == "completed":
            require(
                set(submission["artifact_sha256"]) == {"candidate.json", "source_provenance.json"},
                "Missing shared artifact binding",
            )
            for name, expected in submission["artifact_sha256"].items():
                read(root / name)
                require(sha256_file(root / name) == expected, "Shared submission artifact changed")
    elif state_hash is None:
        require(
            not (root / "candidate.json").exists() and not (root / "source_provenance.json").exists(),
            "Unreserved shared submission",
        )
    return state


def resolve_shared_result(name, value, *, workspace, capability, task=None):
    """Bind a compact native return to its complete local record before validation."""
    if value.get("schema") not in DISPLAY_POLICIES:
        return value
    if task is None:
        task, _ = bundle(capability)
    require(task.get(DISPLAY_FIELD) == value["schema"], "Unbound shared presentation")
    identity = value.get("result_sha256")
    path = history_path(Path(workspace), "shared_tool_response_", identity)
    require(value.get("artifact") == path.name, "Shared presentation artifact changed")
    full = read(path)
    require(digest(full) == identity, "Shared presentation record changed")
    require(
        value == envelope(name, full, policy=task[DISPLAY_FIELD]),
        "Shared presentation differs from its complete record",
    )
    return full


def validate_shared_result(name, value, *, workspace, capability):
    from .tool_contracts import _ledger, validate_generated_result

    root, cap = Path(workspace), Path(capability)
    task, children = bundle(cap)
    value = resolve_shared_result(name, value, workspace=root, capability=cap, task=task)
    ids = list(children)
    state, child_ledgers = None, {}
    if task["phase"] == "develop":
        state = _state(root, task)
        for task_id, child_task in children.items():
            child, tested = _ledger(root / "targets" / task_id, cap / "targets" / task_id, child_task)
            require(len(child["attempts"]) <= len(state["attempts"]), "Untracked target attempts")
            pending = bool(state["attempts"] and state["attempts"][-1]["status"] == "reserved")
            require(len(child["attempts"]) >= len(state["attempts"]) - int(pending), "Target attempts omitted")
            for shared, local in zip(state["attempts"], child["attempts"]):
                for key in ["number", "source_code", "source_sha256", "rationale"]:
                    require(shared[key] == local[key], "Target program differs from shared source")
                if shared["status"] == "completed":
                    require(local["status"] == "completed", "Target attempt was not completed")
            child_ledgers[task_id] = (child, tested)

    def validate_child(task_id, tool, result):
        return validate_generated_result(
            tool,
            result,
            workspace=root / "targets" / task_id,
            capability=cap / "targets" / task_id,
            _require_durable=True,
        )

    if name == "inspect_model_code_task":
        require(
            value.get("schema") == SCHEMA
            and value.get("phase") == task["phase"]
            and value.get("batch_id") == task["batch_id"],
            "Shared inspection mismatch",
        )
        require([x["task_id"] for x in value["targets"]] == ids, "Inspection omitted targets")
        for target in value["targets"]:
            validate_child(target["task_id"], name, target["contract"])
        if state is not None:
            seen = _state(root, task, value["development_state"]["ledger_sha256"])
            require(
                value["development_state"]
                == {
                    "ledger_sha256": digest(seen),
                    "attempts_used": len(seen["attempts"]),
                    "attempts_remaining": 2 - len(seen["attempts"]),
                    "submission_status": (seen["submission"] or {}).get("status", "not_submitted"),
                },
                "Shared development observation differs",
            )
            require(value.get("development_handoff") == description(task), "Shared handoff changed")
    elif name == "evaluate_model_code":
        require(
            state is not None
            and type(value.get("attempts_remaining")) is int
            and value["attempts_remaining"] in {0, 1},
            "Invalid shared evaluation phase or budget",
        )
        number = 2 - value["attempts_remaining"]
        require(
            number <= len(state["attempts"]) and state["attempts"][number - 1]["status"] == "completed",
            "Shared evaluation lacks a completed reservation",
        )
        record = read(root / f"shared_attempt_{number}.json")
        require(value == record["response"], "Shared evaluation receipt changed")
        for target in value["results"]:
            validate_child(target["task_id"], name, target["result"])
    elif name == "submit_model_code":
        require(
            state is not None and state["submission"] is not None and state["submission"]["status"] == "completed",
            "Shared submission incomplete",
        )
        require(value == state["submission"]["receipt"], "Shared submission receipt changed")
        candidate, provenance = read(root / "candidate.json"), read(root / "source_provenance.json")
        source = state["submission"]["source_sha256"]
        require(
            candidate["schema"] == "shared_model_candidates.v1"
            and candidate["batch_id"] == task["batch_id"]
            and candidate["source_sha256"] == source,
            "Shared candidate identity changed",
        )
        require([x["task_id"] for x in candidate["targets"]] == ids, "Shared candidate omitted targets")
        count = 0
        for target in candidate["targets"]:
            task_id = target["task_id"]
            child, tested = child_ledgers[task_id]
            if source in tested:
                require(target["status"] == "candidate", "Successful target was dropped")
                require(
                    sha256_file(root / "targets" / task_id / "candidate.json") == target["candidate_sha256"],
                    "Target candidate changed",
                )
                require(
                    child["submission"]
                    and child["submission"]["status"] == "completed"
                    and child["submission"]["source_sha256"] == source,
                    "Target submitted a different source",
                )
                validate_child(task_id, name, target["receipt"])
                count += 1
            else:
                require(
                    target["status"] == "training_failed" and child["submission"] is None,
                    "Unsuccessful target was counted as a candidate",
                )
        require(count > 0, "No tested target candidates")
        require(
            value
            == {
                "artifact": "candidate.json",
                "schema": SCHEMA,
                "batch_id": task["batch_id"],
                "source_sha256": source,
                "candidate_count": count,
                "allocated_targets": len(ids),
                "status": "candidate",
                "registered": False,
                **receipt_fields(task, source),
            },
            "Shared candidate count or handoff changed",
        )
        require(
            provenance
            == {
                "schema": SCHEMA,
                "origin": "shared_agent_tool_call_source_code",
                "batch_id": task["batch_id"],
                "source_sha256": source,
                "rationale": state["submission"]["rationale"],
                "task_count": len(ids),
                "shared_attempts": len(state["attempts"]),
            },
            "Shared provenance changed",
        )
    elif name == "discover_generated_models":
        require(
            task["phase"] == "reuse" and set(value) == {"schema", "batch_id", "targets"}, "Invalid shared discovery"
        )
        require(
            value["schema"] == SCHEMA
            and value["batch_id"] == task["batch_id"]
            and [x["task_id"] for x in value["targets"]] == ids,
            "Shared discovery coverage changed",
        )
        for target in value["targets"]:
            validate_child(target["task_id"], name, target["result"])
    elif name == "predict_with_generated_model":
        require(
            task["phase"] == "reuse" and [x["task_id"] for x in value["targets"]] == ids,
            "Shared prediction coverage changed",
        )
        predictions = []
        for target in value["targets"]:
            validate_child(target["task_id"], name, target["result"])
            predictions.append(
                {
                    "task_id": target["task_id"],
                    "predictions": read(root / "targets" / target["task_id"] / "predictions.json"),
                }
            )
        require(read(root / "predictions.json") == predictions, "Shared predictions differ from target outputs")
        require(
            value
            == {
                "schema": SCHEMA,
                "artifact": "predictions.json",
                "batch_id": task["batch_id"],
                "model_id": "all",
                "targets": value["targets"],
                "prediction_sha256": digest(predictions),
            },
            "Shared prediction receipt changed",
        )
    else:
        raise ValueError("Unsupported shared-program scientific operation")
