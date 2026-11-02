"""Independent terminal evidence audit for frozen REAP generated-model matrices.

No LLM calls. Refit candidate code in the existing isolated worker, reconstruct
public feature allocations and calculate metrics independently of the evaluator.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.rsi import generated_features as feature_model
from agent.rsi.engine.experience import LIMITATION
from agent.rsi.engine.numerical_evidence import verify_assessment
from agent.rsi.cases.reap_experience import load_library, project_for_target
from agent.rsi.cases.reap_model_case import prepare_reap_model_case
from agent.rsi.engine.versions import VersionStore


from agent.rsi.layout import recorded_relative, recorded_path

def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def independent_metrics(model, rows, scope):
    predictions = feature_model.predict(model, rows, scope)
    y = np.asarray([r["target"] for r in rows], dtype=float)
    p = np.asarray([r["prediction"] for r in predictions], dtype=float)
    mse = float(np.dot(y - p, y - p) / len(y))
    baseline = float(np.dot(y - model["training_mean"], y - model["training_mean"]) / len(y))
    ry, rp = rankdata(y), rankdata(p)
    rho = 0.0 if min(ry.std(), rp.std()) == 0 else float(np.corrcoef(ry, rp)[0, 1])
    return predictions, {
        "rows": len(rows),
        "rmse": mse**0.5,
        "training_mean_rmse": baseline**0.5,
        "mse_skill_vs_training_mean": 1 - mse / baseline,
        "spearman": rho,
        "predictions_sha256": digest(predictions),
    }


def compare_metrics(observed, expected):
    require(set(observed) == set(expected), "Metric fields differ")
    for key, value in expected.items():
        if key in {"rows", "predictions_sha256"}:
            require(observed[key] == value, "Metric identity differs: " + key)
        else:
            require(np.isclose(observed[key], value, rtol=1e-9, atol=1e-12), "Independent metric differs: " + key)


def verify_code_origin(work, events, model):
    """Bind fitted source to successful agent tool input and archived attempts."""
    starts = {e["tool_call_id"]: e for e in events if e.get("kind") == "tool_call_start"}
    ends = {e["tool_call_id"]: e for e in events if e.get("kind") == "tool_result"}
    tested, submitted, attempt_count = {}, [], 0
    evaluation_calls, budget_rejections = 0, 0
    for cid, event in starts.items():
        name = event["tool_name"].split("__")[-1]
        result = ends.get(cid)
        if name == "evaluate_model_code":
            evaluation_calls += 1
            if (
                result
                and result.get("is_error")
                and result.get("output", "").endswith(": code-development budget exhausted")
            ):
                # The pinned tool checks its cap before executing or incrementing
                # the source-attempt counter. Denied requests remain physical tool
                # calls but cannot be interpreted as an additional model fit.
                require(attempt_count == 2, "Budget denial before the declared attempt cap")
                require(not (work / "code_attempt_3.json").exists(), "Third attempt artifact after budget denial")
                budget_rejections += 1
                continue
            attempt_count += 1
            require(attempt_count <= 2, "Admitted code-attempt budget differs")
            if not result or result.get("is_error"):
                continue
            output = json.loads(result["output"])
            code = event["args"]["source_code"]
            sha = hashlib.sha256(code.encode()).hexdigest()
            require(output["source_sha256"] == sha, "Tool source hash differs")
            require(output["attempts_remaining"] == 2 - attempt_count, "Tool attempt reservation differs")
            attempt = read(work / f"code_attempt_{attempt_count}.json")
            require(attempt["source_code"] == code, "Archived attempt source differs")
            require(attempt["training_feedback"] == output["training_only_feedback"], "Attempt feedback differs")
            if output["status"] == "tested_not_published":
                tested[sha] = code
        if name == "submit_model_code" and result and not result.get("is_error"):
            output = json.loads(result["output"])
            sha = event["args"]["source_sha256"]
            require(sha in tested and output["source_sha256"] == sha, "Submission was not a tested source")
            submitted.append(sha)
    require(1 <= attempt_count <= 2, "Code-attempt budget differs")
    require(submitted == [model["source_sha256"]], "Successful submission identity differs")
    require(tested[submitted[0]] == model["recipe"]["source_code"], "Fitted source differs from agent tool input")
    require(model["recipe"]["seed"] == 314159, "Tool recipe seed differs")
    provenance = read(work / "source_provenance.json")
    require(
        provenance["origin"] == "agent_tool_call_source_code"
        and provenance["source_sha256"] == submitted[0]
        and provenance["training_feedback_attempts"] == attempt_count,
        "Source provenance differs",
    )
    return {
        "attempts": attempt_count,
        "evaluation_tool_calls": evaluation_calls,
        "budget_rejected_calls": budget_rejections,
        "successful_tested_sources": len(tested),
        "source_from_agent_tool": True,
    }


def verify_phase(root, arm, phase, protocol, case, rows):
    folder = root / f"arm_{arm['id']:03d}"
    execution = folder / phase / "execution"
    run, lock = read(execution / "run.json"), read(execution / "profile.lock.json")
    require(
        digest({k: v for k, v in lock.items() if k != "sha256"}) == lock["sha256"] == run["profile_sha256"],
        "Profile binding changed",
    )
    expected = json.loads(json.dumps(protocol["profiles"][arm["mode"]]))
    if phase == "reuse":
        expected["budget"]["wall_seconds"] = 420
    require(lock["profile"] == expected, "Model/harness/budget differs from allocation")
    require(
        run["mode"] == arm["mode"] and run["phase"] == phase and run["model"] == expected["model"]["model"],
        "Execution identity differs",
    )
    require(lock["task"]["instructions_sha256"] == protocol["system_sha256"], "System instructions changed")
    if phase == "develop":
        require(lock["task"]["prompt_sha256"] == digest(case["prompt"]), "Development prompt changed")
    bundle = folder / ("development_bundle" if phase == "develop" else "reuse_bundle") / "bundles"
    versions = list((bundle / "versions").iterdir())
    require(len(versions) == 1, "Ambiguous capability package")
    cap = versions[0]
    manifest = VersionStore(bundle).verify(cap.name)
    require(
        {str(p.relative_to(cap)): file_hash(p) for p in cap.rglob("*") if p.is_file()} == lock["task"]["capabilities"],
        "Capability package changed",
    )
    require(manifest["metadata"] == {"phase": phase, "model": expected["model"]}, "Capability metadata changed")
    for name, sha in manifest["files"].items():
        if name.endswith(".py"):
            require(protocol["implementation"].get(recorded_relative(protocol["implementation"], name)) == sha, "Bundled implementation differs")
    for section, prefix in [("adapter_source", "src/agent/harness/"), ("supporting_source", "src/agent/")]:
        for name, sha in lock[section].items():
            require(protocol["implementation"].get(recorded_path(protocol["implementation"], prefix + name)) == sha, "Runtime source differs")
    inputs = (
        {"training.json": rows["training"]}
        if phase == "develop"
        else {"inputs.json": [{k: r[k] for k in ["id", "sequence", "features"]} for r in rows["future"]]}
    )
    task = read(cap / "task.json")
    require(
        task["phase"] == phase and task["scope"] == case["scope"] and task["plugin_id"] == protocol["plugin"],
        "Scientific contract changed",
    )
    require(task["inputs"] == {n: digest(v) for n, v in inputs.items()}, "Scientific input binding differs")
    require(set(lock["task"]["inputs"]) == set(inputs), "Unexpected runtime input")
    for name, values in inputs.items():
        path = folder / phase / "workspace" / name
        require(read(path) == values and file_hash(path) == lock["task"]["inputs"][name], "Runtime data changed")
    if phase == "develop":
        empty = case.get(
            "experience_projection", {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
        )
        require(not (cap / "registered").exists(), "Unplanned pretrained model")
        require(
            read(cap / "experience.json") == empty and task["experience_sha256"] == digest(empty),
            "Cold-start experience changed",
        )
    else:
        packet = read(cap / "registered/capability.json")
        model = read(folder / "controller/artifacts/versions" / arm["model_id"] / "model.json")
        require(
            digest(packet) == task["capability_sha256"]
            and packet["model_id"] == arm["model_id"]
            and packet["model"] == model
            and packet["model_sha256"] == digest(model)
            and packet["assessment"] == arm["assessment"],
            "Reuse capability changed",
        )
    requests = audit_requests(execution / "model_requests.jsonl")
    trace = audit_trace(execution / "events.jsonl", lock["task"]["tools"], expected["budget"]["tool_calls"])
    require(requests == run["audit"]["requests"] and trace == run["audit"]["events"], "Terminal audit snapshot differs")
    recorded = arm["development" if phase == "develop" else "reuse"]
    require(recorded["run"] == run, "Campaign runtime summary differs")
    # Reuse the frozen event-chain auditor only for structural checks; numerical
    # scores and generated-source provenance are independently checked below.
    from run_protein_lifecycle import audit_events

    required = (
        {"inspect_model_code_task", "read_model_experience", "submit_model_code"}
        if phase == "develop"
        else {"discover_generated_models", "predict_with_generated_model"}
    )
    scientific = audit_events(execution, phase, run, required, {"evaluate_model_code"} if phase == "develop" else None)
    require(scientific == recorded["scientific_audit"], "Scientific event audit differs")
    if arm["status"] == "completed" or (phase == "develop" and "assessment" in arm):
        require(
            run["status"] == "completed"
            and run["audit"]["passed"]
            and requests["passed"]
            and trace["passed"]
            and scientific["passed"],
            "Successful stage contains failed evidence",
        )
        if arm["mode"] == "science_expert":
            require(
                run.get("four_role_path_completed") and run.get("finalizer_succeeded"),
                "Scientist role/finalizer evidence incomplete",
            )
    return {
        "session_id": run["session_id"],
        "model_calls": requests["model_calls"],
        "tool_calls": trace["tool_calls"],
        "request_passed": requests["passed"],
        "trace_passed": trace["passed"],
        "scientific_passed": scientific["passed"],
    }


def audit(root, implementation_root):
    root, implementation_root = Path(root).resolve(), Path(implementation_root).resolve()
    report, protocol = read(root / "report.json"), read(root / "protocol.json")
    require(report["status"] == "campaign_completed", "Terminal campaign required")
    require(digest(protocol) == report["protocol_sha256"], "Protocol binding changed")
    for name, sha in protocol["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Frozen source differs: " + name,
        )
    require(feature_model.implementation_hash() == protocol["plugin_sha256"], "Imported feature plugin differs")
    require(report["source_intact"], "Campaign reported source drift")
    if protocol.get("experience_library"):
        manifest, _ = load_library(root / "donor_library")
        require(manifest == protocol["experience_library"], "Donor library differs")
    cases = {}
    for family, case in protocol["cases"].items():
        source = case["source"]
        rebuilt = prepare_reap_model_case(
            source["admission"], source["features"], source["baseline"], family, case["split"]["seed"]
        )
        require(
            (rebuilt[0], rebuilt[4], rebuilt[5]) == (case["scope"], source, case["split"]),
            "Public source reconstruction changed",
        )
        rows = {n: read(root / "data" / family / (n + ".json")) for n in ["training", "acceptance", "future"]}
        for name, values in zip(rows, rebuilt[1:4], strict=True):
            require(rows[name] == values and digest(values) == case["data_hashes"][name], "Frozen allocation differs")
        cases[family] = rows
    require(len(report["arms"]) == len(protocol["allocations"]) == 4, "Allocation removed")
    checked = []
    for planned, arm in zip(protocol["allocations"], report["arms"], strict=True):
        require(all(arm[k] == planned[k] for k in ["id", "family", "mode", "condition"]), "Allocation changed")
        require(arm["status"] not in {"allocated", "running"}, "Nonterminal allocation")
        folder = root / f"arm_{arm['id']:03d}"
        require(read(folder / "report.json") == arm, "Arm report differs")
        case, rows = protocol["cases"][arm["family"]], cases[arm["family"]]
        if protocol.get("experience_library"):
            expected_projection = project_for_target(root / "donor_library", case["scope"], arm["condition"])
            require(case["experience_projection"] == expected_projection, "Cross-family allocation projection differs")
        record = {k: arm[k] for k in ["id", "family", "mode", "status"]}
        for phase in ["develop", "reuse"]:
            if (folder / phase / "execution/run.json").exists():
                record[phase] = verify_phase(root, arm, phase, protocol, case, rows)
        work = folder / "develop/workspace"
        record["candidate_written"] = (work / "candidate.json").exists()
        if record["candidate_written"]:
            candidate = read(work / "candidate.json")
            events = [json.loads(x) for x in (folder / "develop/execution/events.jsonl").read_text().splitlines()]
            record["code_origin"] = verify_code_origin(work, events, candidate)
        if "assessment" not in arm:
            require("future_metrics" not in arm, "Unassessed allocation has scientific score")
            checked.append(record)
            continue
        mid = arm["model_id"]
        controller = folder / "controller"
        VersionStore(controller / "artifacts").verify(mid)
        model = read(controller / "artifacts/versions" / mid / "model.json")
        require(model == candidate and model["source_sha256"] == arm["source_sha256"], "Candidate identity changed")
        require(
            digest(feature_model.fit_candidate(rows["training"], case["scope"], model["recipe"])) == digest(model),
            "Independent training replay differs",
        )
        assessment = read(controller / ("assessment_" + mid + ".json"))
        state = read(controller / "state.json")
        policy = read(controller / "policy.json")
        verify_assessment(controller, mid, model, policy, assessment)
        expected_policy = {
            "schema": 1,
            "scope": case["scope"],
            "plugin_id": protocol["plugin"],
            "plugin_sha256": protocol["plugin_sha256"],
            "training_sha256": digest(rows["training"]),
            "validation_sha256": digest(rows["acceptance"]),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        }
        require(
            policy == expected_policy and assessment["policy_sha256"] == digest(policy), "Acceptance policy differs"
        )
        require(
            assessment == arm["assessment"] and digest(assessment) == state["candidates"][mid]["assessment_sha256"],
            "Assessment identity changed",
        )
        _, metrics = independent_metrics(model, rows["acceptance"], case["scope"])
        compare_metrics(assessment["metrics"], metrics)
        accepted = metrics["mse_skill_vs_training_mean"] > 0
        require(assessment["passed"] == accepted == (mid in state["published"]), "Publication gate differs")
        require(
            assessment["training_replayed"] and assessment["plugin_sha256"] == protocol["plugin_sha256"],
            "Independent assessment evidence differs",
        )
        if accepted:
            eid = arm["registered_experience_id"]
            entry = read(folder / "experience/entries" / (eid + ".json"))
            require(
                digest(entry) == eid
                and entry["parents"]
                == [e["experience_id"] for e in case.get("experience_projection", {"entries": []})["entries"]],
                "Cold-start experience ancestry changed",
            )
        if arm.get("future_scoring", {}).get("status") == "failed":
            try:
                independent_metrics(model, rows["future"], case["scope"])
            except Exception as exc:
                require(type(exc).__name__ == arm["future_scoring"]["error_type"], "Future failure replay differs")
                record.update(
                    training_replayed=True,
                    acceptance_replayed=True,
                    accepted=accepted,
                    future_scoring_failed=True,
                    error_type=type(exc).__name__,
                )
                checked.append(record)
                continue
            raise ValueError("Recorded future failure now produces valid predictions")
        predictions, future_metrics = independent_metrics(model, rows["future"], case["scope"])
        compare_metrics(arm["future_metrics"], future_metrics)
        record.update(
            training_replayed=True,
            acceptance_replayed=True,
            accepted=accepted,
            acceptance_metrics=metrics,
            future_metrics=future_metrics,
        )
        if arm["status"] == "completed":
            require(accepted and record["develop"]["session_id"] != record["reuse"]["session_id"], "Fresh reuse absent")
            require(predictions == read(folder / "reuse/workspace/predictions.json"), "Reuse predictions differ")
            inputs = [{k: r[k] for k in ["id", "sequence", "features"]} for r in rows["future"]]
            receipt = arm["reuse_event"]
            require(
                receipt in state["events"]
                and receipt["event"] == "reused"
                and receipt["model_id"] == mid
                and receipt["session_id"] == record["reuse"]["session_id"]
                and receipt["input_sha256"] == digest(inputs)
                and receipt["prediction_sha256"] == digest(predictions),
                "Reuse receipt differs",
            )
            record["reuse_replayed"] = True
        checked.append(record)
    return {
        "status": "passed",
        "report_sha256": file_hash(root / "report.json"),
        "protocol_sha256": file_hash(root / "protocol.json"),
        "auditor_sha256": file_hash(Path(__file__)),
        "arms": checked,
        "scope": "Frozen numerical replay, public-source allocation reconstruction, independent metric calculation and tool-source binding. Not an independent experimental replicate, full-goal mode comparison or RSI efficacy evidence.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--implementation-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.campaign, args.implementation_root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
