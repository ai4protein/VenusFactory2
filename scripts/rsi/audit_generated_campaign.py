"""Read-only audit and deterministic replay of completed model campaigns.

Outputs go to a fresh directory. Failed runtime allocations remain failed;
audit agreement is not a new acceptance decision or an experimental replicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_protein_lifecycle import prepare_case

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.versions import VersionStore

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def require_equal(a, b, label):
    if digest(a) != digest(b):
        raise ValueError(label)


def verify_snapshot(folder):
    manifest = read(folder / "manifest.json")
    VersionStore(folder.parent.parent).verify(manifest["version_id"])
    return manifest


def audit_arm(campaign, arm, data, matrix):
    folder = (
        campaign / ("arm_{:03d}".format(arm["id"]))
        if matrix
        else campaign / arm["case"] / arm["condition"] / arm["mode"]
    )
    out = {k: arm[k] for k in ["id", "case", "condition", "mode", "harness", "status"] if k in arm}
    out["original_status"] = out.pop("status")
    out["phases"] = {}
    for phase, key in [("develop", "development"), ("reuse", "reuse")]:
        if key not in arm:
            continue
        execution = folder / phase / "execution" if matrix else folder / phase
        raw_run = read(execution / "run.json")
        require_equal(raw_run, arm[key]["run"], "report/runtime disagreement")
        caps_key = ("development" if phase == "develop" else "reuse") + ("_bundle" if matrix else "_capability")
        caps_dirs = list((folder / caps_key / "bundles/versions").iterdir())
        if len(caps_dirs) != 1:
            raise ValueError("ambiguous capability snapshot")
        caps = caps_dirs[0]
        verify_snapshot(caps)
        task = read(caps / "task.json")
        for name, sha in task["inputs"].items():
            require_equal(digest(read(folder / phase / "workspace" / name)), sha, "task input changed")
        if matrix:
            lock = read(execution / "profile.lock.json")
            require_equal(
                digest({k: v for k, v in lock.items() if k != "sha256"}), lock["sha256"], "profile lock changed"
            )
            require_equal(raw_run["profile_sha256"], lock["sha256"], "run profile binding changed")
            for name, sha in lock["task"]["inputs"].items():
                require_equal(file_hash(folder / phase / "workspace" / name), sha, "bound input changed")
            for name, sha in lock["task"]["capabilities"].items():
                require_equal(file_hash(caps / name), sha, "bound capability changed")
            allowed, limit = lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"]
        else:
            binding = read(folder / phase / "binding.json")
            allowed, limit = binding["allowed_scientific_tools"], binding["max_tool_calls"]
        events = audit_trace(execution / "events.jsonl", allowed, limit)
        requests = audit_requests(execution / "model_requests.jsonl")
        original_audit = arm[key].get("scientific_audit", arm[key].get("audit"))
        if matrix:
            require_equal(events["passed"], raw_run["audit"]["events"]["passed"], "event audit changed")
            require_equal(requests["passed"], raw_run["audit"]["requests"]["passed"], "request audit changed")
        else:
            require_equal(requests["passed"], original_audit["request_gateway"]["passed"], "request audit changed")
        records = [json.loads(line) for line in (execution / "model_requests.jsonl").read_text().splitlines()]
        errors = [
            {k: row.get(k) for k in ["event", "call", "error_type"]}
            for row in records
            if row["event"] in {"transport_error", "policy_rejected", "budget_rejected"}
        ]
        out["phases"][phase] = {
            "runtime_status": raw_run["status"],
            "events_passed": events["passed"],
            "requests_passed": requests["passed"],
            "scientific_audit_passed": original_audit["passed"],
            "errors": errors,
            "candidate_written": (folder / phase / "workspace/candidate.json").is_file(),
        }
        if out["original_status"] == "completed" and not (
            events["passed"] and requests["passed"] and original_audit["passed"]
        ):
            raise ValueError("completed arm has failed evidence gate")
    if "assessment" in arm:
        controller = folder / "controller"
        policy, state = read(controller / "policy.json"), read(controller / "state.json")
        if len(state["candidates"]) != 1:
            raise ValueError("candidate allocation changed")
        mid = next(iter(state["candidates"]))
        model_dir = controller / "artifacts/versions" / mid
        verify_snapshot(model_dir)
        model = read(model_dir / "model.json")
        plugin = get_model_plugin(policy["plugin_id"])
        require_equal(plugin.source_sha256, policy["plugin_sha256"], "evaluation implementation drift")
        scope = policy["scope"]
        train, val, future = data
        require_equal(digest(train), policy["training_sha256"], "training changed")
        require_equal(digest(val), policy["validation_sha256"], "acceptance changed")
        require_equal(read(folder / "develop/workspace/candidate.json"), model, "submitted candidate changed")
        provenance = read(folder / "develop/workspace/source_provenance.json")
        sha = hashlib.sha256(model["recipe"]["source_code"].encode()).hexdigest()
        for value in [model["source_sha256"], arm["source_sha256"], provenance["source_sha256"]]:
            require_equal(value, sha, "source provenance changed")
        require_equal(plugin.fit(train, scope, model["recipe"]), model, "independent training replay differs")
        assessment = read(controller / ("assessment_" + mid + ".json"))
        require_equal(assessment, arm["assessment"], "assessment changed")
        require_equal(digest(assessment), state["candidates"][mid]["assessment_sha256"], "assessment hash changed")
        metrics = plugin.evaluate(model, val, scope)
        require_equal(metrics, assessment["metrics"], "acceptance replay differs")
        passed = metrics["mse_skill_vs_training_mean"] > policy["minimum_skill"]
        require_equal(passed, assessment["passed"], "acceptance decision differs")
        require_equal(mid in state["published"], passed, "publication decision differs")
        out.update(source_sha256=sha, training_replayed=True, acceptance_metrics=metrics, acceptance_passed=passed)
        if "future_metrics" in arm:
            require_equal(plugin.evaluate(model, future, scope), arm["future_metrics"], "future replay differs")
            out["future_metrics"] = arm["future_metrics"]
        if out["original_status"] == "completed":
            inputs = [{"id": x["id"], "sequence": x["sequence"]} for x in future]
            predictions = read(folder / "reuse/workspace/predictions.json")
            require_equal(plugin.predict(model, inputs, scope), predictions, "reuse predictions differ")
            require_equal(
                arm["reuse"]["run"]["session_id"] != arm["development"]["run"]["session_id"],
                True,
                "reuse session not fresh",
            )
            reuse = [x for x in state["events"] if x["event"] == "reused"]
            if len(reuse) != 1:
                raise ValueError("reuse event missing or duplicated")
            require_equal(reuse[0]["prediction_sha256"], digest(predictions), "reuse trace differs")
            require_equal(reuse[0]["input_sha256"], digest(inputs), "reuse inputs differ")
            out["fresh_session_reuse_replayed"] = True
    out["audit_passed"] = True
    return out


def main(args):
    report, protocol = read(args.campaign / "report.json"), read(args.campaign / "protocol.json")
    if report["status"] != "campaign_completed":
        raise ValueError("Only terminal campaigns may be audited")
    require_equal(digest(protocol), report["protocol_sha256"], "protocol hash mismatch")
    matrix = "matrix" in protocol
    hashes = protocol["implementation"] if matrix else read(args.campaign / "implementation/manifest.json")
    for name, sha in hashes.items():
        require_equal(file_hash(args.campaign / "implementation" / name), sha, "archived implementation changed")
    data = {}
    for case in protocol["data"]:
        if matrix:
            scope, train, val, future, source, _ = prepare_case(case, args.source_cache)
            declared = protocol["data"][case]
            for name, value in [("training", train), ("acceptance", val), ("future", future)]:
                require_equal(digest(value), declared[name + "_sha256"], "source split reconstruction differs")
            require_equal(scope, declared["scope"], "scope changed")
            require_equal(source, declared["source"], "source provenance changed")
        else:
            train, val, future = [
                read(args.campaign / "frozen_data" / case / (n + ".json")) for n in ["training", "validation", "future"]
            ]
            for name, value in zip(["training", "validation", "future"], [train, val, future], strict=True):
                require_equal(digest(value), protocol["data_hashes"][case][name], "frozen data changed")
        data[case] = (train, val, future)
    if len(protocol["allocations"]) != len(report["arms"]):
        raise ValueError("allocation missing")
    for allocated, actual in zip(protocol["allocations"], report["arms"], strict=True):
        for key in allocated:
            if key != "status":
                require_equal(allocated[key], actual[key], "allocation identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    output = {
        "status": "running",
        "campaign": str(args.campaign),
        "source_report_sha256": file_hash(args.campaign / "report.json"),
        "protocol_sha256": digest(protocol),
        "auditor_sha256": file_hash(Path(__file__)),
        "scope": "Evidence integrity and numerical replay, not a new scientific replicate",
        "arms": [],
    }
    for arm in report["arms"]:
        try:
            result = audit_arm(args.campaign, arm, data[arm["case"]], matrix)
        except Exception as exc:
            result = {
                "case": arm["case"],
                "mode": arm["mode"],
                "condition": arm.get("condition"),
                "id": arm.get("id"),
                "audit_passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        output["arms"].append(result)
        write_json(args.output / "report.json", output)
        print(
            json.dumps(
                {k: result[k] for k in ["case", "mode", "id", "condition", "audit_passed", "error"] if k in result}
            ),
            flush=True,
        )
    output["status"] = "audit_completed" if all(x["audit_passed"] for x in output["arms"]) else "audit_failed"
    write_json(args.output / "report.json", output)
    if output["status"] != "audit_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    main(parser.parse_args())
