"""One native tool invocation, one shared program, explicitly separate task fits.

Installed as generated_tools.py in a versioned shared-program capability. The
unchanged single-task implementation is installed once as single_model_tools.py.
No controller, acceptance labels or publication API are present in the workspace.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

from development_handoff import description, receipt_fields
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from protein_model import digest
from shared_model_display import FIELD as DISPLAY_FIELD
from shared_model_display import POLICIES as DISPLAY_POLICIES
from shared_model_display import POLICY as COMPACT_WIRE_POLICY
from shared_model_display import envelope, wire_text

SCHEMA = "shared_model_program.v1"
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace")).absolute()
BUNDLE = Path(__file__).resolve().parent
LEDGER = ".shared_model_development.json"
mcp = FastMCP("VenusFactorySharedModelProgram")
_MODULES = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary(path):
    require(not any(p.is_symlink() for p in [path, *path.parents]), "Symlinked batch evidence")
    require(path.is_file(), "Missing batch evidence")
    return path


def read(path):
    value = json.loads(ordinary(path).read_text())
    digest(value)
    return value


def sha(path):
    return hashlib.sha256(ordinary(path).read_bytes()).hexdigest()


def history_path(root, prefix, value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid ledger history hash")
    return root / (prefix + value + ".json")


def matching_attempts(common, child):
    require(len(common) == len(child), "Target and shared attempt counts differ")
    for first, second in zip(common, child):
        for key in ["number", "source_code", "source_sha256", "rationale"]:
            require(first[key] == second[key], "Target evaluated a different shared program")
        if first["status"] == "completed":
            require(second["status"] == "completed", "Target evaluation remains incomplete")


def info():
    task = read(BUNDLE / "task.json")
    fields = {"schema", "phase", "batch_id", "targets", "implementation"}
    if task.get("phase") == "develop":
        fields.add("development_handoff")
    if DISPLAY_FIELD in task:
        require(task[DISPLAY_FIELD] in DISPLAY_POLICIES, "Unsupported shared tool presentation")
        fields.add(DISPLAY_FIELD)
    require(set(task) == fields and task["schema"] == SCHEMA, "Invalid shared-program capability")
    require(task["phase"] in {"develop", "reuse"}, "Invalid shared-program phase")
    targets = task["targets"]
    require(isinstance(targets, list) and 1 <= len(targets) <= 32, "Batch requires 1..32 targets")
    ids = [t["id"] for t in targets]
    require(len(ids) == len(set(ids)), "Repeated batch target")
    require(all(isinstance(i, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,119}", i) for i in ids), "Invalid target id")
    implementation = task["implementation"]
    require(set(implementation) == {p.name for p in BUNDLE.glob("*.py")}, "Batch implementation coverage changed")
    require(all(sha(BUNDLE / n) == h for n, h in implementation.items()), "Batch implementation changed")
    require(task["batch_id"] == digest({"phase": task["phase"], "targets": targets}), "Batch identity changed")
    for target in targets:
        require(set(target) == {"id", "task_sha256"}, "Invalid target binding")
        child = read(BUNDLE / "targets" / target["id"] / "task.json")
        require(digest(child) == target["task_sha256"], "Target definition changed")
        require(child["phase"] == task["phase"], "Target phase changed")
        require(child["shared_implementation_sha256"] == digest(implementation), "Unbound shared implementation")
    return task


def save(name, value):
    require(re.fullmatch(r"[.a-zA-Z0-9_]+\.json", name) is not None, "Invalid batch artifact name")
    path = ROOT / name
    require(not path.exists() and not path.is_symlink(), "Batch artifact already exists")
    temporary = ROOT / (name + ".writing")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    sync()


def sync():
    descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def single(target_id):
    """Each target gets its own module globals; shared Python files are not copied."""
    if target_id not in _MODULES:
        name = "_vf_shared_target_" + digest({"bundle": str(BUNDLE), "workspace": str(ROOT), "id": target_id})
        spec = importlib.util.spec_from_file_location(name, BUNDLE / "single_model_tools.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.ROOT = ROOT / "targets" / target_id
        module.BUNDLE = BUNDLE / "targets" / target_id
        require(module.ROOT.is_dir() and not module.ROOT.is_symlink(), "Missing target workspace")
        _MODULES[target_id] = module
    return _MODULES[target_id]


def call(module, name, *args):
    function = getattr(module, name)
    return getattr(function, "fn", function)(*args)


def store(state):
    require(not (ROOT / LEDGER).is_symlink(), "Symlinked shared ledger")
    history = "shared_ledger_" + digest(state) + ".json"
    if (ROOT / history).exists():
        require(read(ROOT / history) == state, "Shared ledger history changed")
    else:
        save(history, state)
    temp = ROOT / (LEDGER + ".writing")
    with temp.open("x") as stream:
        stream.write(json.dumps(state, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, ROOT / LEDGER)
    sync()


def load(task):
    require(not (ROOT / (LEDGER + ".writing")).exists(), "Incomplete shared ledger write requires host review")
    path = ROOT / LEDGER
    if not path.exists() and not path.is_symlink():
        require(
            not list(ROOT.glob("shared_attempt_*.json")) and not (ROOT / "candidate.json").exists(),
            "Unbound shared artifacts",
        )
        store({"schema": SCHEMA, "task_sha256": digest(task), "attempts": [], "submission": None})
    state = read(path)
    require(set(state) == {"schema", "task_sha256", "attempts", "submission"}, "Invalid shared ledger")
    require(state["schema"] == SCHEMA and state["task_sha256"] == digest(task), "Shared task binding changed")
    require(read(ROOT / ("shared_ledger_" + digest(state) + ".json")) == state, "Shared ledger history differs")
    require(isinstance(state["attempts"], list) and len(state["attempts"]) <= 2, "Shared attempt budget changed")
    ids = [t["id"] for t in task["targets"]]
    expected = set()
    for number, item in enumerate(state["attempts"], 1):
        require(
            item["number"] == number and item["status"] in {"reserved", "completed"}, "Shared attempt order changed"
        )
        require(
            hashlib.sha256(item["source_code"].encode()).hexdigest() == item["source_sha256"], "Shared source changed"
        )
        if item["status"] == "reserved":
            require(number == len(state["attempts"]), "Work continued after an interrupted shared attempt")
            continue
        name = f"shared_attempt_{number}.json"
        expected.add(name)
        record = read(ROOT / name)
        require(sha(ROOT / name) == item["artifact_sha256"], "Shared feedback artifact changed")
        require(
            record["source_sha256"] == item["source_sha256"] and record["rationale"] == item["rationale"],
            "Shared feedback source differs",
        )
        require([x["task_id"] for x in record["results"]] == ids, "Shared feedback omitted or reordered targets")
        for target in record["results"]:
            child = read(history_path(ROOT / "targets" / target["task_id"], "code_ledger_", target["ledger_sha256"]))
            require(digest(child) == target["ledger_sha256"], "Target ledger history changed")
            matching_attempts(state["attempts"][:number], child["attempts"])
    actual = {p.name for p in ROOT.glob("shared_attempt_*.json")}
    pending = (
        {f"shared_attempt_{len(state['attempts'])}.json"}
        if state["attempts"] and state["attempts"][-1]["status"] == "reserved"
        else set()
    )
    require(not (actual - expected - pending) and expected <= actual, "Shared feedback coverage changed")
    submission = state["submission"]
    if submission is not None:
        require(submission["status"] in {"reserved", "completed"}, "Invalid shared submission")
        if submission["status"] == "completed":
            hashes = submission["artifact_sha256"]
            require(
                set(hashes) == {"candidate.json", "source_provenance.json"}, "Incomplete shared submission evidence"
            )
            require(all(sha(ROOT / n) == h for n, h in hashes.items()), "Shared candidate evidence changed")
    elif (ROOT / "candidate.json").exists() or (ROOT / "source_provenance.json").exists():
        raise ValueError("Unreserved shared candidate")
    return state


@contextlib.contextmanager
def locked():
    require(not any(p.is_symlink() for p in [ROOT, *ROOT.parents]), "Symlinked shared workspace")
    descriptor = os.open(ROOT / ".shared_model.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield info()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def settled(state):
    require(
        not any(x["status"] == "reserved" for x in state["attempts"]),
        "Interrupted shared evaluation requires host review",
    )
    require(
        (state["submission"] or {}).get("status") != "reserved", "Interrupted shared submission requires host review"
    )


def present(name, result, task):
    if task.get(DISPLAY_FIELD) is None:
        return result
    packet = envelope(name, result, policy=task[DISPLAY_FIELD])
    artifact = packet["artifact"]
    if (ROOT / artifact).exists():
        require(read(ROOT / artifact) == result, "Shared presentation record changed")
    else:
        save(artifact, result)
    if task[DISPLAY_FIELD] == COMPACT_WIRE_POLICY:
        return ToolResult(content=[TextContent(type="text", text=wire_text(packet))], structured_content=packet)
    return packet


@mcp.tool()
def inspect_model_code_task() -> dict:
    """Inspect every admitted target and its training-only model contract. One source program is shared, but fits and outcomes remain target-specific."""
    with locked() as task:
        children = [
            {"task_id": t["id"], "contract": call(single(t["id"]), "inspect_model_code_task")} for t in task["targets"]
        ]
        result = {
            "schema": SCHEMA,
            "phase": task["phase"],
            "batch_id": task["batch_id"],
            "targets": children,
            "counting": "One shared development execution; each task requires its own independent acceptance and fresh reuse.",
        }
        if task["phase"] == "develop":
            state = load(task)
            result.update(
                development_handoff=description(task),
                development_state={
                    "ledger_sha256": digest(state),
                    "attempts_used": len(state["attempts"]),
                    "attempts_remaining": 2 - len(state["attempts"]),
                    "submission_status": (state["submission"] or {}).get("status", "not_submitted"),
                },
            )
        return present("inspect_model_code_task", result, task)


@mcp.tool()
def evaluate_model_code(source_code: str, rationale: str) -> dict:
    """Evaluate one common fit/predict program separately on every target's internal training split. At most two shared attempts; failures remain in the result."""
    with locked() as task:
        require(task["phase"] == "develop", "Shared development unavailable")
        state = load(task)
        settled(state)
        require(state["submission"] is None and len(state["attempts"]) < 2, "Shared development budget exhausted")
        number = len(state["attempts"]) + 1
        source_hash = hashlib.sha256(source_code.encode()).hexdigest()
        item = {
            "number": number,
            "source_code": source_code,
            "source_sha256": source_hash,
            "rationale": rationale,
            "status": "reserved",
        }
        state["attempts"].append(item)
        store(state)
        results = []
        for target in task["targets"]:
            worker = single(target["id"])
            result = worker.evaluate_source(source_code, rationale)
            child, _ = worker.load_ledger()
            results.append({"task_id": target["id"], "result": result, "ledger_sha256": digest(child)})
        response = {
            "schema": SCHEMA,
            "source_sha256": source_hash,
            "attempts_remaining": 2 - number,
            "status": "tested_not_published"
            if any(x["result"]["status"] == "tested_not_published" for x in results)
            else "development_failed",
            "results": results,
        }
        name = f"shared_attempt_{number}.json"
        save(name, {"source_sha256": source_hash, "rationale": rationale, "results": results, "response": response})
        item.update(status="completed", artifact_sha256=sha(ROOT / name))
        store(state)
        return present("evaluate_model_code", response, task)


@mcp.tool()
def submit_model_code(source_sha256: str, rationale: str) -> dict:
    """Fit a successfully tested common source to each eligible target's full training data. Every target is reported; independent host acceptance follows."""
    with locked() as task:
        require(task["phase"] == "develop", "Shared submission unavailable")
        state = load(task)
        settled(state)
        if state["submission"] is not None:
            previous = state["submission"]
            require(
                previous["source_sha256"] == source_sha256 and previous["rationale"] == rationale,
                "Shared submission arguments differ",
            )
            return previous["receipt"]
        eligible = {}
        for target in task["targets"]:
            child, tested = single(target["id"]).load_ledger()
            matching_attempts(state["attempts"], child["attempts"])
            eligible[target["id"]] = source_sha256 in tested
        require(any(eligible.values()), "Shared source must first pass a training-only test")
        state["submission"] = {"source_sha256": source_sha256, "rationale": rationale, "status": "reserved"}
        store(state)
        outcomes = []
        for target in task["targets"]:
            name = target["id"]
            if eligible[name]:
                receipt = call(single(name), "submit_model_code", source_sha256, rationale)
                outcomes.append(
                    {
                        "task_id": name,
                        "status": "candidate",
                        "receipt": receipt,
                        "candidate_sha256": sha(ROOT / "targets" / name / "candidate.json"),
                    }
                )
            else:
                outcomes.append(
                    {
                        "task_id": name,
                        "status": "training_failed",
                        "reason": "Selected source has no successful training-only evaluation for this target",
                    }
                )
        candidate = {
            "schema": "shared_model_candidates.v1",
            "batch_id": task["batch_id"],
            "source_sha256": source_sha256,
            "targets": outcomes,
        }
        save("candidate.json", candidate)
        save(
            "source_provenance.json",
            {
                "schema": SCHEMA,
                "origin": "shared_agent_tool_call_source_code",
                "batch_id": task["batch_id"],
                "source_sha256": source_sha256,
                "rationale": rationale,
                "task_count": len(outcomes),
                "shared_attempts": len(state["attempts"]),
            },
        )
        receipt = {
            "artifact": "candidate.json",
            "schema": SCHEMA,
            "batch_id": task["batch_id"],
            "source_sha256": source_sha256,
            "candidate_count": sum(eligible.values()),
            "allocated_targets": len(outcomes),
            "status": "candidate",
            "registered": False,
            **receipt_fields(task, source_sha256),
        }
        state["submission"].update(
            status="completed",
            receipt=receipt,
            artifact_sha256={n: sha(ROOT / n) for n in ["candidate.json", "source_provenance.json"]},
        )
        store(state)
        return receipt


@mcp.tool()
def discover_generated_models() -> dict:
    """Discover the independently accepted target models in this shared reuse episode."""
    with locked() as task:
        require(task["phase"] == "reuse", "Discovery requires a reuse episode")
        result = {
            "schema": SCHEMA,
            "batch_id": task["batch_id"],
            "targets": [
                {"task_id": t["id"], "result": call(single(t["id"]), "discover_generated_models")}
                for t in task["targets"]
            ],
        }

        return present("discover_generated_models", result, task)


@mcp.tool()
def predict_with_generated_model(model_id: str) -> dict:
    """Use model_id='all' to predict every target's unlabelled inputs with its own accepted model. Writes per-target predictions and one shared receipt."""
    with locked() as task:
        require(task["phase"] == "reuse" and model_id == "all", "Shared reuse requires model_id='all'")
        results, predictions = [], []
        for target in task["targets"]:
            worker = single(target["id"])
            packet = worker.compatible_packet()
            require(packet is not None, "Missing independently accepted target model")
            result = call(worker, "predict_with_generated_model", packet["model_id"])
            results.append({"task_id": target["id"], "result": result})
            predictions.append(
                {"task_id": target["id"], "predictions": read(ROOT / "targets" / target["id"] / "predictions.json")}
            )
        save("predictions.json", predictions)
        return {
            "schema": SCHEMA,
            "artifact": "predictions.json",
            "batch_id": task["batch_id"],
            "model_id": "all",
            "targets": results,
            "prediction_sha256": digest(predictions),
        }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
