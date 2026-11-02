"""Harness-independent sessions and host acceptance for generated paper steps."""

import json
import shutil
from pathlib import Path

from agent.harness import Profile, Task
from agent.harness.contracts import file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import execute
from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES

from .paper_step import compare, digest
from .paper_step_tool_contracts import validate_paper_step_result
from .paper_step_tool_service import PaperStepTools, source_hash
from .storage import atomic_json, file_lock, safe_child
from .versions import VersionStore

MODULES = [
    "paper_step_tool_service.py",
    "paper_step.py",
    "code_sandbox.py",
    "code_worker.py",
    "numeric_runtime.py",
    "protein_references.py",
    "storage.py",
]
PHASE_TOOLS = {
    "develop": frozenset(
        {"inspect_paper_step_task", "evaluate_paper_step", "submit_paper_step", "discover_paper_steps"}
    ),
    "reuse": frozenset({"inspect_paper_step_task", "discover_paper_steps", "execute_registered_paper_step"}),
}
REQUIRED = {
    "develop": {"inspect_paper_step_task", "evaluate_paper_step", "submit_paper_step"},
    "reuse": PHASE_TOOLS["reuse"],
}
PROMPTS = {
    "develop": "Inspect the admitted paper computation. Write the complete reproduce(inputs) Python yourself. Test it on the permitted examples with evaluate_paper_step, repair within the two-attempt budget if needed, and submit exactly one successful tested source hash. Independent acceptance and registration occur after this session. Do not claim new biology or full-paper replication.",
    "reuse": "Inspect this fresh computation task, discover the compatible registered paper step, and execute it once on all current inputs. Reuse the immutable source without editing or retraining it. Report the output artifact; independent acceptance follows after this session.",
}
INSTRUCTIONS = "Use only the admitted scientific MCP tools. Do not use shell or filesystem tools or manually write artifacts. Work from the frozen scope and permitted examples. Tool feedback is not independent scientific acceptance. Keep failed attempts and uncertainty explicit."


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked session artifact")
    return json.loads(path.read_text())


def prepare_phase(lifecycle, folder, phase, *, capability_id=None, case_id=None):
    if phase not in PHASE_TOOLS:
        raise ValueError("Unknown paper step phase")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    workspace, source = folder / "workspace", folder / "source"
    workspace.mkdir()
    source.mkdir()
    public = lifecycle.public_task()
    task = {"schema": 1, "phase": phase, "public_task": public, "development_attempts": 2}
    if phase == "reuse":
        public.pop("development_cases")
        packet = lifecycle.capability(capability_id)
        cases = [c for c in lifecycle.definition["cases"]["reuse"] if c["id"] == case_id]
        if len(cases) != 1:
            raise ValueError("Unknown host-admitted fresh input")
        task.update(case_id=case_id, inputs_sha256=digest(cases[0]["inputs"]), capability_sha256=digest(packet))
        atomic_json(workspace / "step_inputs.json", cases[0]["inputs"])
        atomic_json(source / "step_capability.json", packet)
    module_root = Path(__file__).parent
    for name in MODULES:
        target = source / "agent/rsi/engine" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module_root / name, target)
    for package in [source / "agent", source / "agent/rsi", source / "agent/rsi/engine"]:
        (package / "__init__.py").write_text("")
    shutil.copy2(module_root / "paper_step_tools.py", source / "generated_tools.py")
    atomic_json(source / "step_task.json", task)
    atomic_json(
        source / "scientific_task.json",
        {
            "schema": 2,
            "kind": "reproduction",
            "context_tool": "inspect_paper_step_task",
            "artifact_file": "step_candidate.json" if phase == "develop" else "step_reuse.json",
            "required_tools": sorted(REQUIRED[phase]),
            "completion_repair_limit": 0,
        },
    )
    versions = VersionStore(folder / "bundles")
    manifest = versions.snapshot(
        source, metadata={"kind": "paper_step", "phase": phase, "interface": public["interface"]}
    )
    declaration = {
        "phase": phase,
        "capability_version": manifest["version_id"],
        "admission_sha256": digest(lifecycle.binding),
        "capability_id": capability_id,
        "case_id": case_id,
        "prompt": PROMPTS[phase],
        "instructions": INSTRUCTIONS,
    }
    declared_files = [
        "paper_step_sessions.py",
        "paper_step_tool_contracts.py",
        "paper_step_tool_service.py",
        "reproduction_tool_contracts.py",
        "langgraph_tools.py",
    ]
    declaration["host_implementation"] = {name: file_hash(module_root / name) for name in declared_files}
    atomic_json(folder / "phase.json", declaration)
    return folder


def phase_task(folder):
    folder = Path(folder)
    declaration = read(folder / "phase.json")
    if any(
        file_hash(Path(__file__).parent / name) != expected
        for name, expected in declaration["host_implementation"].items()
    ):
        raise ValueError("Paper step session host implementation changed")
    version = declaration["capability_version"]
    if not isinstance(version, str) or len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid scientific bundle identity")
    capability = safe_child(folder, "bundles/versions/" + version)
    return Task(
        folder / "workspace",
        capability,
        folder / "runtime",
        declaration["prompt"],
        declaration["instructions"],
        PHASE_TOOLS[declaration["phase"]],
        phase="chat",
    )


async def execute_phase(folder, profile, runtimes, *, credential):
    return await execute(profile, phase_task(folder), runtimes, credential=credential)


def verify_phase(folder):
    folder = Path(folder)
    task = phase_task(folder)
    phase = read(folder / "phase.json")["phase"]
    run = read(task.output / "run.json")
    lock = read(task.output / "profile.lock.json")
    signed = dict(lock)
    signed.pop("sha256")
    if digest(signed) != lock["sha256"] or run.get("profile_sha256") != lock["sha256"]:
        raise ValueError("Runtime profile binding differs")
    if run.get("status") != "completed" or run.get("audit", {}).get("passed") is not True:
        raise ValueError("Scientific session did not complete its audited execution")
    profile = Profile.from_dict(lock["profile"])
    before = lock["task"]
    after = task.binding()
    for name in ["prompt_sha256", "instructions_sha256", "tools", "mcp_entry", "phase", "capabilities"]:
        if before[name] != after[name]:
            raise ValueError("Scientific session binding changed")
    if any(after["inputs"].get(k) != v for k, v in before["inputs"].items()):
        raise ValueError("Scientific session input changed")
    trace = audit_trace(task.output / "events.jsonl", task.tools, profile.budget.tool_calls)
    requests = audit_requests(task.output / "model_requests.jsonl")
    if (
        not trace["passed"]
        or trace["tool_errors"]
        or not requests["passed"]
        or requests["model_calls"] > profile.budget.model_calls
    ):
        raise ValueError("Unverified model requests or tool trace")
    if profile.harness in NATIVE_EXPERT_HARNESSES:
        from agent.harness.native_expert_audit import audit_native_expert

        role_audit = audit_native_expert(task.output, task.workspace, task.capability, result=run)
        if not role_audit["passed"]:
            raise ValueError("Four-role scientific episode did not pass")
    events = [json.loads(line) for line in (task.output / "events.jsonl").read_text().splitlines()]
    starts, completed, calls = {}, set(), []
    for event in events:
        if event.get("kind") == "tool_call_start":
            starts[event["tool_call_id"]] = event
        elif event.get("kind") == "tool_result":
            start = starts[event["tool_call_id"]]
            name = start["tool_name"].removeprefix("mcp__venusfactory__")
            validate_paper_step_result(name, event["output"], workspace=task.workspace, capability=task.capability)
            completed.add(name)
            calls.append((name, start, json.loads(event["output"])))
    if not REQUIRED[phase] <= completed:
        raise ValueError("Incomplete paper step tool chain")
    service = PaperStepTools(task.workspace, task.capability)
    provenance = {
        "phase": phase,
        "session_id": run["session_id"],
        "model": profile.model.model,
        "provider": profile.model.provider,
        "mode": profile.mode,
        "harness": profile.harness,
        "profile_sha256": lock["sha256"],
        "events_sha256": file_hash(task.output / "events.jsonl"),
        "requests_sha256": file_hash(task.output / "model_requests.jsonl"),
        "model_calls": requests["model_calls"],
        "tool_calls": trace["tool_calls"],
    }
    if phase == "develop":
        candidate = service.read(task.workspace, "step_candidate.json")
        sha = candidate["source_sha256"]
        evaluations = [
            (call, out)
            for name, call, out in calls
            if name == "evaluate_paper_step"
            and out.get("source_sha256") == sha
            and out.get("development_passed") is True
        ]
        submissions = [
            call for name, call, out in calls if name == "submit_paper_step" and out.get("source_sha256") == sha
        ]
        if not evaluations or len(submissions) != 1 or submissions[0]["args"] != {"source_sha256": sha}:
            raise ValueError("Missing unique tested source submission")
        if not any(
            call["args"].get("source_code") == candidate["source_code"]
            and call["args"].get("rationale") == candidate["rationale"]
            and out["attempt"] == candidate["development_attempt"]
            for call, out in evaluations
        ):
            raise ValueError("Submitted source did not originate in the recorded development tool arguments")
        provenance.update(
            source_sha256=sha,
            source_via_model_tool_call_verified=True,
            artifact_sha256=file_hash(task.workspace / "step_candidate.json"),
        )
    else:
        packet = service.packet()
        receipt = service.reuse_receipt()
        executions = [call for name, call, out in calls if name == "execute_registered_paper_step"]
        if len(executions) != 1 or executions[0]["args"] != {"capability_id": packet["capability_id"]}:
            raise ValueError("Fresh reuse must have one actual registered-source execution call")
        provenance.update(
            capability_id=packet["capability_id"],
            case_id=receipt["case_id"],
            source_sha256=source_hash(packet["source_code"]),
            artifact_sha256=file_hash(task.workspace / "step_reuse.json"),
        )
    return provenance


def accept_development(lifecycle, folder):
    folder = Path(folder)
    declaration = read(folder / "phase.json")
    if declaration["phase"] != "develop" or declaration["admission_sha256"] != digest(lifecycle.binding):
        raise ValueError("Development belongs to another host admission")
    with file_lock(safe_child(folder, ".host_acceptance.lock")):
        if (folder / "host_acceptance_reservation.json").exists():
            raise ValueError("Host acceptance was already reserved; inspect existing evidence before recovery")
        provenance = verify_phase(folder)
        atomic_json(folder / "host_acceptance_reservation.json", provenance)
        candidate = read(folder / "workspace/step_candidate.json")
        identity = lifecycle.propose(
            candidate["source_code"], declared_origin="model_tool_submission:" + provenance["session_id"]
        )
        atomic_json(folder / "host_candidate.json", {"candidate_id": identity, "provenance": provenance})
        verdict = lifecycle.assess(identity)
        result = {"candidate_id": identity, "assessment": verdict, "source_provenance": provenance}
        if verdict["passed"]:
            result["capability_id"] = lifecycle.register(identity)
        atomic_json(folder / "host_acceptance.json", result)
        return result


def accept_reuse(lifecycle, folder, *, development_session):
    folder = Path(folder)
    declaration = read(folder / "phase.json")
    if declaration["phase"] != "reuse" or declaration["admission_sha256"] != digest(lifecycle.binding):
        raise ValueError("Reuse belongs to another host admission")
    provenance = verify_phase(folder)
    if provenance["session_id"] == development_session:
        raise ValueError("Registered reuse requires a fresh session")
    identity, case_id = declaration["capability_id"], declaration["case_id"]
    if provenance["capability_id"] != identity or provenance["case_id"] != case_id:
        raise ValueError("Reuse tool result belongs to another allocation")
    with file_lock(safe_child(lifecycle.root, ".step.lock")):
        lifecycle.capability(identity)
        case = next(c for c in lifecycle.definition["cases"]["reuse"] if c["id"] == case_id)
        record = read(folder / "workspace/step_reuse.json")
        evidence = {k: record[k] for k in ["case_id", "input_sha256", "output", "execution"]}
        evidence["comparison"] = compare(record["output"], case["expected"], **lifecycle.definition["tolerance"])
        packet = lifecycle._read("registered/" + identity + ".json")
        lifecycle._verify_execution(lifecycle._candidate(packet["candidate_id"]), case, evidence)
        prefix = "reuse/" + identity + "/" + case_id
        lifecycle._save(prefix + "/reservation.json", {"capability_id": identity, "case_id": case_id})
        lifecycle._save(prefix + "/evidence.json", evidence)
        lifecycle._save(prefix + "/session_provenance.json", provenance)
    result = {"verification": lifecycle.verify_reuse(identity, case_id), "source_provenance": provenance}
    atomic_json(folder / "host_acceptance.json", result)
    return result
