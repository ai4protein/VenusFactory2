"""Read-only policy-origin audit, independent of the revision producer.

No LLM call, policy promotion or performance conclusion is made here.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

from scripts.rsi.cases.audit_reap_generated import compare_metrics, independent_metrics, read, require

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.rsi.engine.numerical_evidence import verify_assessment
from agent.rsi.engine.versions import VersionStore

TOOLS = {"inspect_improver_context", "submit_improver_policy"}
METRICS = {"rows", "rmse", "training_mean_rmse", "mse_skill_vs_training_mean", "spearman"}


from agent.rsi.layout import recorded_relative, recorded_path

def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def policy(value):
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "plugin_id", "proposal_instruction", "revision_instruction"},
        "Unexpected policy fields",
    )
    require(
        type(value["schema"]) is int and value["schema"] == 1 and value["plugin_id"] == "model_improver.prompt.v1",
        "Unknown policy schema",
    )
    for name in ["proposal_instruction", "revision_instruction"]:
        require(
            isinstance(value[name], str) and 40 <= len(value[name].encode()) <= 8000,
            "Invalid policy instruction length",
        )
    return canonical(value)


def history_projection(references, numerical=False):
    rows = []
    for ref in references:
        root = Path(ref["path"])
        for name in ["goal", "protocol"]:
            require(file_hash(root / (name + ".json")) == ref[name + "_sha256"], "History reference changed")
        state, protocol = read(root / "goal.json"), read(root / "protocol.json")
        require(
            state["status"] in {"completed", "budget_exhausted"} and state["inflight"] is None,
            "History was not terminal",
        )
        require(protocol["model_plugin_id"] == "generated_features.v1", "Unexpected history model type")
        for cycle in state["cycles"]:
            if cycle.get("phase") != "develop":
                continue
            row = {
                "case": protocol["split"]["family"],
                "cycle": cycle["index"],
                "status": cycle["status"],
                "goal_sha256": ref["goal_sha256"],
                "protocol_sha256": ref["protocol_sha256"],
            }
            episode = root / "episodes" / f"cycle_{cycle['index']:03d}"
            if (episode / "acceptance.json").exists():
                assessment = read(episode / "acceptance.json")
                mid, controller = assessment["model_id"], episode / "controller"
                VersionStore(controller / "artifacts").verify(mid)
                model = read(controller / "artifacts/versions" / mid / "model.json")
                published = read(controller / "state.json")
                require(
                    assessment["training_replayed"] is True
                    and assessment == read(controller / ("assessment_" + mid + ".json"))
                    and digest(assessment) == published["candidates"][mid]["assessment_sha256"]
                    and assessment["passed"] == (mid in published["published"]),
                    "History assessment changed",
                )
                source = model["recipe"]["source_code"]
                require(hashlib.sha256(source.encode()).hexdigest() == model["source_sha256"], "History source changed")
                verify_assessment(controller, mid, model, read(controller / "policy.json"), assessment)
                if numerical:
                    from agent.rsi.engine.generated_features import fit_candidate

                    rebuilt = fit_candidate(read(root / "data/training.json"), protocol["scope"], model["recipe"])
                    require(digest(rebuilt) == digest(model), "History training replay identity differs")
                    _, metrics = independent_metrics(model, read(root / "data/validation.json"), protocol["scope"])
                    compare_metrics(assessment["metrics"], metrics)
                row.update(
                    model_id=mid,
                    source_sha256=model["source_sha256"],
                    source_code=source,
                    development_acceptance_metrics={k: assessment["metrics"][k] for k in METRICS},
                    acceptance_passed=assessment["passed"],
                )
            elif "error_type" in cycle:
                row["error_type"] = cycle["error_type"]
            rows.append(row)
    rows = rows[-12:]
    require(1 <= len(rows) <= 12, "Missing development history")
    sources = [i for i, row in enumerate(rows) if "source_code" in row]
    for i in sources[:-2]:
        rows[i].pop("source_code")
    return rows


def context(root, numerical=False):
    protocol = read(root / "protocol.json")
    VersionStore(root / "bundles").verify(protocol["bundle_id"])
    packet = read(root / "bundles/versions" / protocol["bundle_id"] / "revision_context.json")
    require(
        set(packet)
        == {
            "schema",
            "parent_id",
            "parent_policy",
            "history_sha256",
            "development_history",
            "editable_fields",
            "authority",
            "status",
        },
        "Unexpected context fields",
    )
    require(
        packet["schema"] == 1 and packet["parent_id"] == policy(packet["parent_policy"]) == protocol["parent_id"],
        "Parent policy identity differs",
    )
    history = history_projection(protocol["history_references"], numerical)
    require(
        packet["development_history"] == history
        and packet["history_sha256"] == canonical(history) == protocol["history_sha256"],
        "History projection contains altered, omitted or non-development evidence",
    )
    require(
        packet["editable_fields"] == ["proposal_instruction", "revision_instruction"]
        and packet["status"] == "revision_context_not_performance_evidence"
        and packet["authority"]
        == "Candidate prompt data only. Fixed host evaluation, tools, budgets and final-label isolation override generated instructions.",
        "Revision authority changed",
    )
    prompt = (
        "Inspect the frozen improver context before submitting one successor. The existing improver's revision policy is:\n"
        + packet["parent_policy"]["revision_instruction"]
        + "\nReturn the complete revised policy via submit_improver_policy. Its plugin/schema stay fixed. You may revise both proposal_instruction and revision_instruction; do not copy performance claims into the candidate status. No automatic promotion is requested."
    )
    archived = ast.parse((root / "implementation/scripts/rsi/run_improver_revision.py").read_text())
    systems = [
        ast.literal_eval(node.value)
        for node in archived.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM" for t in node.targets)
    ]
    require(
        len(systems) == 1
        and digest(systems[0]) == protocol["instructions_sha256"]
        and digest(prompt) == protocol["prompt_sha256"],
        "Parent revision instruction was not bound to the prompt",
    )
    require(protocol["proposal_attempts"] == 1, "Revision retry allowance changed")
    for name, sha in protocol["implementation"].items():
        require(file_hash(root / "implementation" / name) == sha, "Revision implementation archive changed")
    return protocol, packet


def usage(paths):
    result = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "responses_with_usage": 0,
        "responses": 0,
        "requests": 0,
        "complete": True,
    }
    for path in paths:
        if not path.exists():
            result["complete"] = False
            continue
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        req = [r for r in records if r.get("event") == "request"]
        res = [r for r in records if r.get("event") == "response"]
        result["requests"] += len(req)
        result["responses"] += len(res)
        for row in res:
            u = row.get("usage") or {}
            if all(type(u.get(k)) is int and u[k] >= 0 for k in ["prompt_tokens", "completion_tokens"]):
                result["responses_with_usage"] += 1
                for k in ["prompt_tokens", "completion_tokens"]:
                    result[k] += u[k]
            else:
                result["complete"] = False
    result["complete"] &= result["requests"] == result["responses"]
    result["interpretation"] = (
        "Provider-reported tokens summed where available; missing usage is unknown, not zero cost."
    )
    return result


def submitted_candidate(root, protocol, packet):
    report = read(root / "report.json")
    require(report["status"] == "candidate_submitted", "No completed revision candidate")
    require(
        report["protocol_sha256"] == digest(protocol)
        and report["promoted"] is False
        and report["recursive_effectiveness_verified"] is False,
        "Revision claims or binding differ",
    )
    path = root / "workspace/improver_candidate.json"
    require(path.is_file() and not path.is_symlink(), "Missing regular candidate artifact")
    candidate = read(path)
    require(
        set(candidate)
        == {"schema", "policy", "policy_id", "parent_id", "history_sha256", "rationale", "status", "promoted"},
        "Unexpected candidate fields",
    )
    require(
        candidate["schema"] == 1
        and candidate["policy_id"] == policy(candidate["policy"])
        and candidate["policy_id"] != packet["parent_id"]
        and candidate["parent_id"] == packet["parent_id"]
        and candidate["history_sha256"] == packet["history_sha256"]
        and candidate["status"] == "untested_candidate"
        and candidate["promoted"] is False,
        "Candidate identity or untested status differs",
    )
    require(
        isinstance(candidate["rationale"], str) and 20 <= len(candidate["rationale"].encode()) <= 6000,
        "Invalid candidate rationale",
    )
    require(
        report["candidate_sha256"] == canonical(candidate)
        and report["policy_id"] == candidate["policy_id"]
        and report["parent_id"] == candidate["parent_id"],
        "Candidate report differs",
    )
    execution = root / "execution"
    events = [json.loads(s) for s in (execution / "events.jsonl").read_text().splitlines() if s.strip()]
    starts = [e for e in events if e.get("kind") == "tool_call_start"]
    submitted = [e for e in starts if e.get("tool_name", "").split("__")[-1] == "submit_improver_policy"]
    require(len(submitted) == 1, "Exactly one model submission required")
    call = submitted[0]
    require(
        json.loads(call["args"]["policy_json"]) == candidate["policy"]
        and call["args"]["rationale"] == candidate["rationale"],
        "Candidate not emitted by recorded model call",
    )
    attempt = read(root / "workspace/improver_attempt.json")
    require(
        attempt
        == {
            "proposal_text_sha256": canonical(call["args"]["policy_json"]),
            "status": "candidate_submitted",
            "candidate_sha256": canonical(candidate),
        },
        "Durable single-attempt receipt differs",
    )
    # Require a successful, source-bound context observation before submission.
    ends = {e["tool_call_id"]: e for e in events if e.get("kind") == "tool_result"}
    inspections = [
        e for e in starts[: starts.index(call)] if e.get("tool_name", "").split("__")[-1] == "inspect_improver_context"
    ]
    require(
        inspections
        and any(
            not ends[e["tool_call_id"]].get("is_error")
            and events.index(ends[e["tool_call_id"]]) < events.index(call)
            and json.loads(ends[e["tool_call_id"]]["output"]) == packet
            for e in inspections
        ),
        "No verified context observation before revision",
    )
    require(not ends[call["tool_call_id"]].get("is_error"), "Candidate submission failed")
    return candidate


def audit(root):
    root = Path(root).resolve()
    protocol, packet = context(root, numerical=True)
    report = read(root / "report.json")
    require(report["status"] == "candidate_submitted", "No completed revision candidate")
    profile, execution = protocol["profile"], root / "execution"
    cap = profile["budget"]
    require(
        profile["mode"] == "science_agent"
        and 0 < cap["model_calls"] <= 8
        and 0 < cap["tool_calls"] <= 4
        and 0 < cap["wall_seconds"] <= 420,
        "Revision budget changed",
    )
    requests = audit_requests(execution / "model_requests.jsonl")
    trace = audit_trace(execution / "events.jsonl", TOOLS, cap["tool_calls"])
    run, lock = read(execution / "run.json"), read(execution / "profile.lock.json")
    require(
        requests["passed"]
        and trace["passed"]
        and trace["tool_errors"] == 0
        and requests["model_calls"] <= cap["model_calls"],
        "Physical revision execution failed",
    )
    require(
        run["status"] == "completed"
        and run["audit"]["passed"]
        and requests == run["audit"]["requests"]
        and trace == run["audit"]["events"]
        and run == report["run"]
        and report["scientific_audit"]["passed"],
        "Revision run snapshot differs",
    )
    require(
        lock["profile"] == profile
        and digest({k: v for k, v in lock.items() if k != "sha256"}) == lock["sha256"] == run["profile_sha256"],
        "Revision profile lock differs",
    )
    task = lock["task"]
    require(
        task["inputs"] == {}
        and set(task["tools"]) == TOOLS
        and task["prompt_sha256"] == protocol["prompt_sha256"]
        and task["instructions_sha256"] == protocol["instructions_sha256"],
        "Revision task contract differs",
    )
    bundle = root / "bundles/versions" / protocol["bundle_id"]
    require(
        task["capabilities"] == {str(p.relative_to(bundle)): file_hash(p) for p in bundle.rglob("*") if p.is_file()},
        "Observed revision context differs",
    )
    for section, prefix in [("adapter_source", "src/agent/harness/"), ("supporting_source", "src/agent/")]:
        for name, sha in lock[section].items():
            require(protocol["implementation"].get(recorded_path(protocol["implementation"], prefix + name)) == sha, "Revision runtime source changed")
    for name, origin in [("generated_tools.py", "improver_tools.py"), ("improver_policy.py", "improver_policy.py")]:
        require(
            file_hash(bundle / name) == protocol["implementation"][recorded_relative(protocol["implementation"], origin)],
            "Revision capability source changed",
        )
    proposed = submitted_candidate(root, protocol, packet)
    elapsed = run["host_wall_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "Invalid revision elapsed time")
    names = [
        "protocol.json",
        "report.json",
        "workspace/improver_candidate.json",
        "execution/model_requests.jsonl",
        "execution/events.jsonl",
    ]
    receipt = {
        "path": str(root),
        "files": {n: file_hash(root / n) for n in names},
        "policy_id": proposed["policy_id"],
        "parent_id": proposed["parent_id"],
        "history_sha256": proposed["history_sha256"],
        "observed_model_calls": requests["model_calls"],
        "observed_tool_calls": trace["tool_calls"],
        "host_wall_seconds": elapsed,
        "charged_once_per_campaign": True,
        "token_usage": usage([execution / "model_requests.jsonl"]),
    }
    return {
        "status": "passed",
        "receipt": receipt,
        "candidate": proposed,
        "attempt_sha256": file_hash(root / "workspace/improver_attempt.json"),
        "history_validation_metrics_replayed": True,
        "improved_performance_verified": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.resolve().is_relative_to(a.revision.resolve()), "Keep audit outside historical evidence")
    result = audit(a.revision)
    a.output.mkdir(parents=True, exist_ok=False)
    result["auditor_sha256"] = file_hash(Path(__file__))
    (a.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (a.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != "candidate"}))
