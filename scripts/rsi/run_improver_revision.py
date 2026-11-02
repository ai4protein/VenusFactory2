"""Generate one untested improver-policy successor through the configured harness.

The parent revision instruction actually controls this proposal. Only verified
development history is projected; no future scores, fitted weights or labels.
No policy promotion occurs here: matched offspring tests are still required.
"""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path.insert(0, str(_entry_root))


import argparse
import ast
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path

from scripts.rsi.evaluation.audit_improver_revision import context as audit_revision_context
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_protein_lifecycle import audit_events

from agent.harness import Profile, Task, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.engine.improver_policy import (
    BASE_POLICY,
    METRICS,
    candidate,
    identity,
    revision_packet,
    validate_candidate,
    validate_packet,
)
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.numerical_evidence import verify_assessment
from agent.rsi.engine.storage import safe_child
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalStore

SYSTEM = "Revise the model-code improver policy using only the supplied development evidence. You can propose policy text through the declared tools, but cannot change evaluation, tool access, budgets or dataset partitions. Submit one candidate and report its untested status. A changed policy is not evidence of better offspring or recursive improvement."
TOOLS = frozenset({"inspect_improver_context", "submit_improver_policy"})


from agent.rsi.layout import source_relative

from agent.rsi.layout import recorded_relative

def read(path):
    return json.loads(Path(path).read_text())


def historical_plugin_hash(snapshot, hashes):
    """Interpret the published fingerprint manifests without executing code."""
    source_name = recorded_relative(hashes, "generated_features.py")
    if source_name not in hashes:
        raise ValueError("Historical plugin snapshot is incomplete")
    for name, sha in hashes.items():
        if file_hash(safe_child(snapshot, name)) != sha:
            raise ValueError("Historical implementation binding changed")
    tree = ast.parse((snapshot / source_name).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "implementation_hash"]
    if len(functions) != 1:
        raise ValueError("Historical fingerprint definition is unsupported")
    manifests = [
        ast.literal_eval(node.iter)
        for node in ast.walk(functions[0])
        if isinstance(node, ast.comprehension) and isinstance(node.iter, ast.List)
    ]
    legacy = ["generated_features.py", "generated_model.py", "code_sandbox.py", "code_worker.py"]
    allowed = [legacy, legacy + ["protein_references.py"], legacy + ["protein_references.py", "numeric_runtime.py"]]
    if len(manifests) != 1 or manifests[0] not in allowed:
        raise ValueError("Historical fingerprint manifest is unsupported")
    required = [recorded_relative(hashes, name) for name in manifests[0]]
    if not all(name in hashes for name in required):
        raise ValueError("Historical plugin snapshot is incomplete")
    return hashlib.sha256(b"".join((snapshot / name).read_bytes() for name in required)).hexdigest()


def historical_model(folder, protocol, episode, assessment):
    """Read a verified old artifact without instantiating today's evaluator.

    This imports development evidence only. It never executes archived source,
    refits a model, upgrades an acceptance decision or reads future metrics.
    """
    controller = episode / "controller"
    policy = read(controller / "policy.json")
    plugin_sha = protocol["model_plugin_sha256"]
    if (
        policy["plugin_id"] != protocol["model_plugin_id"]
        or policy["plugin_sha256"] != plugin_sha
        or policy["scope"] != protocol["scope"]
        or assessment["policy_sha256"] != digest(policy)
        or assessment["plugin_sha256"] != plugin_sha
    ):
        raise ValueError("Historical assessment policy binding changed")
    if get_model_plugin(protocol["model_plugin_id"]).source_sha256 != plugin_sha:
        archived_sha = historical_plugin_hash(folder / "implementation", protocol["implementation"])
        if archived_sha != plugin_sha:
            raise ValueError("Historical plugin fingerprint changed")
    mid = assessment["model_id"]
    versions = VersionStore(controller / "artifacts")
    manifest = versions.verify(mid)
    if manifest["metadata"]["policy_sha256"] != digest(policy):
        raise ValueError("Historical model policy binding changed")
    current = read(controller / "state.json")
    if (
        assessment["training_replayed"] is not True
        or digest(assessment) != current["candidates"][mid]["assessment_sha256"]
        or read(controller / ("assessment_" + mid + ".json")) != assessment
    ):
        raise ValueError("Development acceptance binding changed")
    if assessment["passed"] != (mid in current["published"]):
        raise ValueError("Development publication differs from acceptance")
    model = read(controller / "artifacts/versions" / mid / "model.json")
    if (
        model["plugin_id"] != policy["plugin_id"]
        or model["scope"] != policy["scope"]
        or hashlib.sha256(model["recipe"]["source_code"].encode()).hexdigest() != model["source_sha256"]
    ):
        raise ValueError("Historical model source binding changed")
    verify_assessment(controller, mid, model, policy, assessment)
    return model


def project_development_history(goals):
    entries, references = [], []
    for folder in goals:
        folder = Path(folder).resolve()
        state, protocol = GoalStore(folder).read(), read(folder / "protocol.json")
        if state["status"] not in {"completed", "budget_exhausted"} or state["inflight"] is not None:
            raise ValueError("Only terminal development histories can be projected")
        if protocol["model_plugin_id"] != "generated_features.v1":
            raise ValueError("This projection supports frozen-feature modelling histories")
        goal_sha, protocol_sha = file_hash(folder / "goal.json"), file_hash(folder / "protocol.json")
        references.append({"path": str(folder), "goal_sha256": goal_sha, "protocol_sha256": protocol_sha})
        for cycle in state["cycles"]:
            if cycle.get("phase") != "develop":
                continue
            entry = {
                "case": protocol["split"]["family"],
                "cycle": cycle["index"],
                "status": cycle["status"],
                "goal_sha256": goal_sha,
                "protocol_sha256": protocol_sha,
            }
            episode = folder / "episodes" / f"cycle_{cycle['index']:03d}"
            if (episode / "acceptance.json").exists():
                assessment = read(episode / "acceptance.json")
                mid = assessment["model_id"]
                model = historical_model(folder, protocol, episode, assessment)
                entry.update(
                    model_id=mid,
                    source_sha256=model["source_sha256"],
                    source_code=model["recipe"]["source_code"],
                    development_acceptance_metrics={k: assessment["metrics"][k] for k in METRICS},
                    acceptance_passed=assessment["passed"],
                )
            elif "error_type" in cycle:
                entry["error_type"] = cycle["error_type"]
            entries.append(entry)
    entries = entries[-12:]
    seen = 0
    for row in reversed(entries):
        if "source_code" in row:
            seen += 1
            if seen > 2:
                row.pop("source_code")
    return entries, references


def revision_prompt(packet):
    packet = validate_packet(packet)
    return (
        "Inspect the frozen improver context before submitting one successor. The existing improver's revision policy is:\n"
        + packet["parent_policy"]["revision_instruction"]
        + "\nReturn the complete revised policy via submit_improver_policy. Its plugin/schema stay fixed. You may revise both proposal_instruction and revision_instruction; do not copy performance claims into the candidate status. No automatic promotion is requested."
    )


def prepare(args):
    profile = Profile.from_dict(read(args.profile))
    if (
        profile.mode != "science_agent"
        or profile.budget.model_calls > 8
        or profile.budget.tool_calls > 4
        or profile.budget.wall_seconds > 420
    ):
        raise ValueError(
            "Policy revision requires an explicit Agent profile capped at 8 model calls, 4 tools and 420 seconds"
        )
    if not builtin_registry().preflight(profile, default_runtimes())["ready"]:
        raise ValueError("Configured policy-revision harness is unavailable")
    parent = read(args.parent_policy) if args.parent_policy else BASE_POLICY
    history, references = project_development_history(args.history_goal)
    packet = revision_packet(parent, history)
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "bundle_source"
    source.mkdir()
    for original, target in [("improver_policy.py", "improver_policy.py"), ("improver_tools.py", "generated_tools.py")]:
        (source / target).write_bytes((ROOT / source_relative(original)).read_bytes())
    write_json(source / "revision_context.json", packet)
    versions = VersionStore(args.output / "bundles")
    manifest = versions.snapshot(
        source, metadata={"parent_id": packet["parent_id"], "history_sha256": packet["history_sha256"]}
    )
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_harness_probe.py",
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        ROOT / "scripts/rsi/evaluation/audit_improver_revision.py",
        ROOT / "scripts/rsi/cases/audit_reap_generated.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        ROOT / "src/agent/kimi_client.py",
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for name in hashes:
        target = args.output / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (args.output / "workspace").mkdir()
    protocol = {
        "profile": profile.public(),
        "parent_id": packet["parent_id"],
        "history_sha256": packet["history_sha256"],
        "history_references": references,
        "bundle_id": manifest["version_id"],
        "implementation": hashes,
        "instructions_sha256": digest(SYSTEM),
        "prompt_sha256": digest(revision_prompt(packet)),
        "proposal_attempts": 1,
        "scope": "One bounded candidate-policy revision from retrospective development feedback. No offspring experiment, promotion, new model or RSI-effectiveness claim.",
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "report.json", {"status": "prepared", "protocol_sha256": digest(protocol)})
    return protocol


def preflight_history(output, protocol, report):
    """Replay donor training and validation before reading credentials or making a proposal."""
    report["status"] = "history_preflight"
    write_json(output / "report.json", report)
    receipt = {"protocol_sha256": digest(protocol), "external_model_calls": 0}
    try:
        checked, packet = audit_revision_context(output, numerical=True)
        if checked != protocol:
            raise ValueError("Protocol changed during history replay")
        receipt.update(status="passed", history_sha256=packet["history_sha256"])
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "history_preflight.json", receipt)
        report.update(
            status="history_preflight_failed",
            history_preflight_sha256=file_hash(output / "history_preflight.json"),
            error_type=type(exc).__name__,
            external_model_calls=0,
            promoted=False,
            recursive_effectiveness_verified=False,
        )
        write_json(output / "report.json", report)
        raise
    write_json(output / "history_preflight.json", receipt)
    report["history_preflight_sha256"] = file_hash(output / "history_preflight.json")
    write_json(output / "report.json", report)
    return receipt


async def _run(output, credential_toml):
    output = output.resolve()
    protocol, report = read(output / "protocol.json"), read(output / "report.json")
    if report["status"] != "prepared" or report["protocol_sha256"] != digest(protocol):
        raise ValueError("Already attempted or changed revision; no automatic retry")
    for name, sha in protocol["implementation"].items():
        if file_hash(ROOT / name) != sha or file_hash(output / "implementation" / name) != sha:
            raise ValueError("Frozen revision implementation changed")
    for ref in protocol["history_references"]:
        if (
            file_hash(Path(ref["path"]) / "goal.json") != ref["goal_sha256"]
            or file_hash(Path(ref["path"]) / "protocol.json") != ref["protocol_sha256"]
        ):
            raise ValueError("Source development goal changed")
    versions = VersionStore(output / "bundles")
    versions.verify(protocol["bundle_id"])
    capability = output / "bundles/versions" / protocol["bundle_id"]
    packet = validate_packet(read(capability / "revision_context.json"))
    profile = Profile.from_dict(protocol["profile"])
    prompt = revision_prompt(packet)
    if digest(prompt) != protocol["prompt_sha256"] or digest(SYSTEM) != protocol["instructions_sha256"]:
        raise ValueError("Parent revision instruction was not bound to this task")
    preflight_history(output, protocol, report)
    report["status"] = "running"
    write_json(output / "report.json", report)
    key = credential_from_toml(credential_toml, profile.model)
    task = Task(output / "workspace", capability, output / "execution", prompt, SYSTEM, TOOLS, phase="chat")
    result = await execute(profile, task, default_runtimes(), credential=key)
    scientific = audit_events(task.output, "revision", result, set(TOOLS), set())
    report.update(
        status="revision_failed",
        run=result,
        scientific_audit=scientific,
        recursive_effectiveness_verified=False,
        promoted=False,
    )
    artifact = task.workspace / "improver_candidate.json"
    if (
        result["status"] == "completed"
        and result["audit"]["passed"]
        and scientific["passed"]
        and artifact.is_file()
        and not artifact.is_symlink()
    ):
        proposed = validate_candidate(read(artifact), packet)
        events = [json.loads(line) for line in (task.output / "events.jsonl").read_text().splitlines()]
        calls = [
            e
            for e in events
            if e.get("kind") == "tool_call_start" and e.get("tool_name", "").split("__")[-1] == "submit_improver_policy"
        ]
        if (
            len(calls) != 1
            or candidate(json.loads(calls[0]["args"]["policy_json"]), calls[0]["args"]["rationale"], packet) != proposed
        ):
            raise ValueError("Policy candidate does not match the single recorded model tool call")
        report.update(
            status="candidate_submitted",
            policy_id=proposed["policy_id"],
            parent_id=proposed["parent_id"],
            candidate_sha256=identity(proposed),
        )
    versions.verify(protocol["bundle_id"])
    write_json(output / "report.json", report)
    return report


async def run(output, credential_toml):
    with (output / ".revision.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return await _run(output, credential_toml)
        except Exception as exc:
            report = read(output / "report.json")
            if report["status"] == "running":
                report.update(
                    status="revision_error",
                    error_type=type(exc).__name__,
                    promoted=False,
                    recursive_effectiveness_verified=False,
                )
                write_json(output / "report.json", report)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--parent-policy", type=Path)
    parser.add_argument("--history-goal", type=Path, action="append")
    parser.add_argument("--credential-toml", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.profile or not args.history_goal:
            parser.error("Preparation needs profile and terminal development history")
        result = prepare(args)
        print(
            json.dumps(
                {"status": "prepared", "parent_id": result["parent_id"], "history_sha256": result["history_sha256"]}
            )
        )
    else:
        if not args.credential_toml:
            parser.error("Execution requires credential source")
        result = asyncio.run(run(args.output, args.credential_toml))
        print(json.dumps({k: result[k] for k in ["status", "promoted", "recursive_effectiveness_verified"]}))
