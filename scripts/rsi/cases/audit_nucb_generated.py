"""Read-only terminal audit of the full-sequence NucB generated-model matrix."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

import numpy as np

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi import generated_classifier as classifier
from agent.rsi.cases.nucb_case import prepare_nucb_case
from agent.rsi.engine.versions import VersionStore

ROOT = Path(__file__).resolve().parents[3]


from agent.rsi.layout import recorded_relative, recorded_path

def load(path):
    return json.loads(path.read_text())


def independent_metrics(model, rows, scope):
    predictions = classifier.predict(model, rows, scope)
    p = np.asarray([x["probabilities"] for x in predictions])
    y = np.asarray([x["target"] for x in rows], dtype=int)
    truth = np.zeros_like(p)
    truth[np.arange(len(y)), y] = 1
    # Calculate proper-score averages by class and then sum, independently of the evaluator.
    brier = float(np.sum(np.mean((p - truth) ** 2, axis=0)))
    baseline = float(np.sum(np.mean((truth - np.asarray(model["training_class_prior"])) ** 2, axis=0)))
    return predictions, {
        "brier": brier,
        "training_prior_brier": baseline,
        "brier_skill_vs_training_prior": 1 - brier / baseline,
        "log_loss": float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)))),
    }


def compare_metrics(actual, expected):
    for key, value in expected.items():
        if not np.isclose(actual[key], value, rtol=1e-9, atol=1e-12):
            raise ValueError("Independent score differs: " + key)


def verify_bundle(folder, phase, lock, protocol, arm, training, future):
    """Bind the observed runtime capability and inputs to the allocation."""
    bundle_root = folder / ("development_bundle" if phase == "develop" else "reuse_bundle") / "bundles"
    versions = list((bundle_root / "versions").iterdir())
    if len(versions) != 1:
        raise ValueError("Expected one frozen capability package per phase")
    cap = versions[0]
    manifest = VersionStore(bundle_root).verify(cap.name)
    observed = {str(p.relative_to(cap)): file_hash(p) for p in cap.rglob("*") if p.is_file()}
    if observed != lock["task"]["capabilities"]:
        raise ValueError("Runtime capability package differs from immutable bundle")
    for name, sha in manifest["files"].items():
        if name.endswith(".py") and protocol["implementation"].get(recorded_relative(protocol["implementation"], name)) != sha:
            raise ValueError("Capability implementation differs from frozen protocol")
    if manifest["metadata"] != {"phase": phase, "model": lock["profile"]["model"]}:
        raise ValueError("Capability metadata binding differs")
    task = load(cap / "task.json")
    if task["phase"] != phase or task["scope"] != protocol["scope"] or task["plugin_id"] != protocol["plugin"]:
        raise ValueError("Capability scientific contract differs")
    expected_inputs = (
        {"training.json": training}
        if phase == "develop"
        else {"inputs.json": [{"id": row["id"], "sequence": row["sequence"]} for row in future]}
    )
    if task["inputs"] != {name: digest(rows) for name, rows in expected_inputs.items()}:
        raise ValueError("Capability input content binding differs")
    if set(lock["task"]["inputs"]) != set(expected_inputs):
        raise ValueError("Unexpected runtime input, including possible future labels")
    for name, rows in expected_inputs.items():
        if load(folder / phase / "workspace" / name) != rows:
            raise ValueError("Observed inputs differ from frozen allocation")
    if phase == "develop":
        if (cap / "registered").exists():
            raise ValueError("Development received an unplanned fitted model")
        expected = protocol["experience_projections"][arm["condition"]]
        if digest(load(cap / "experience.json")) != expected or task["experience_sha256"] != expected:
            raise ValueError("Experience projection differs from assigned condition")
    else:
        packet = load(cap / "registered/capability.json")
        if digest(packet) != task["capability_sha256"] or packet["model_id"] != arm["model_id"]:
            raise ValueError("Reuse capability identity changed")
        model = load(folder / "controller/artifacts/versions" / arm["model_id"] / "model.json")
        if (
            packet["model"] != model
            or packet["model_sha256"] != digest(model)
            or packet["assessment"] != arm["assessment"]
        ):
            raise ValueError("Reuse did not receive the independently accepted model")
    return cap.name


def verify_development_guidance(protocol, lock):
    # Historical protocols predate explicit prompt fields; their immutable source
    # remains the evidence. New protocols also bind the exact observed prompt.
    if "development_guidance" not in protocol:
        return
    if protocol["development_guidance"] not in {"standard", "compact_probability"}:
        raise ValueError("Unregistered development guidance")
    expected = digest(protocol["development_prompt"])
    if expected != protocol["development_prompt_sha256"] or lock["task"]["prompt_sha256"] != expected:
        raise ValueError("Development prompt differs from assigned guidance")
    if lock["task"]["instructions_sha256"] != protocol["development_instructions_sha256"]:
        raise ValueError("Development instructions differ")


def runtime_snapshot_discrepancy(run, requests, events):
    """Retain a narrowly identified shutdown defect, never convert it to a passed audit."""
    recorded = run["audit"]
    if events != recorded["events"]:
        raise ValueError("Runtime event audit differs")
    old = recorded["requests"]
    if requests == old:
        return None
    before, after = old["errors"], requests["errors"]
    if (
        run["status"] != "timeout"
        or old["passed"] is not False
        or requests["passed"] is not False
        or {k: v for k, v in old.items() if k != "errors"} != {k: v for k, v in requests.items() if k != "errors"}
        or len(after) <= len(before)
        or after[: len(before)] != before
        or any(error != "transport_error" for error in after[len(before) :])
    ):
        raise ValueError("Runtime request audit differs")
    return {
        "kind": "timeout_terminal_transport_errors_added",
        "recorded_requests": old,
        "terminal_requests": requests,
        "additional_transport_errors": len(after) - len(before),
        "interpretation": "Stored runtime snapshot and terminal log disagree; both audits remain failed. Compatible with late shutdown logging, not proof of the event timing.",
    }


def verify_implementation(protocol, campaign, implementation_root):
    for name, sha in protocol["implementation"].items():
        if file_hash(campaign / "implementation" / name) != sha or file_hash(implementation_root / name) != sha:
            raise ValueError("Use the frozen campaign implementation for numerical replay")


def audit(args):
    campaign = args.campaign.resolve()
    report = load(campaign / "report.json")
    protocol = load(campaign / "protocol.json")
    if report["status"] != "campaign_completed":
        raise ValueError("Terminal campaign required; do not audit an unobserved live prefix")
    if digest(protocol) != report["protocol_sha256"]:
        raise ValueError("Protocol binding changed")
    implementation_root = (getattr(args, "implementation_root", None) or ROOT).resolve()
    verify_implementation(protocol, campaign, implementation_root)
    if classifier.implementation_hash() != protocol["plugin_sha256"]:
        raise ValueError("Classifier implementation differs")
    scope = protocol["scope"]
    train, val, future = [load(campaign / (name + ".json")) for name in ("training", "acceptance", "future")]
    for name, rows in (("training", train), ("acceptance", val), ("future", future)):
        if digest(rows) != protocol["data_hashes"][name]:
            raise ValueError("Data changed")
        classifier.validate_rows(rows, scope, labels=True)
    groups = [{r["sequence"] for r in x} for x in (train, val, future)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Temporal sequence overlap")
    rebuilt = Path(protocol["source"]["generation_files"]["g1"]["path"]).parents[2]
    assets = (getattr(args, "assets", None) or implementation_root / "research_runs/rsi/case_assets/nuclease_design" / protocol["source"]["commit"]).resolve()
    rebuilt_case = prepare_nucb_case(assets, rebuilt)
    if rebuilt_case[0] != scope or any(
        digest(a) != digest(b) for a, b in zip(rebuilt_case[1:4], (train, val, future), strict=True)
    ):
        raise ValueError("Generation-specific source reconstruction differs")
    donor = load(campaign / "donor_evidence/entry.json")
    if digest(donor) != protocol["donor_id"]:
        raise ValueError("Donor source differs")
    for condition, expected in protocol["experience_projections"].items():
        if digest(load(campaign / "donor_evidence" / (condition + ".json"))) != expected:
            raise ValueError("Archived donor projection changed")
    if len(report["arms"]) != len(protocol["allocations"]):
        raise ValueError("Allocation removed")
    checked = []
    for planned, arm in zip(protocol["allocations"], report["arms"], strict=True):
        if any(arm[k] != planned[k] for k in ("id", "condition", "mode")):
            raise ValueError("Allocation identity changed")
        folder = campaign / ("arm_{:03d}".format(arm["id"]))
        record = {"id": arm["id"], "condition": arm["condition"], "mode": arm["mode"], "status": arm["status"]}
        if arm["status"] in {"running", "allocated"}:
            raise ValueError("Nonterminal arm")
        for phase in ("develop", "reuse"):
            path = folder / phase / "execution"
            if not path.exists():
                continue
            run = load(path / "run.json")
            lock = load(path / "profile.lock.json")
            sha = lock.pop("sha256")
            if digest(lock) != sha or run["profile_sha256"] != sha:
                raise ValueError("Profile lock changed")
            expected = json.loads(json.dumps(protocol["profiles"][arm["mode"]]))
            if phase == "reuse":
                expected["budget"]["wall_seconds"] = 420
            if lock["profile"] != expected:
                raise ValueError("Model, harness, workflow or budget drift")
            if phase == "develop":
                verify_development_guidance(protocol, lock)
            cap_id = verify_bundle(folder, phase, lock, protocol, arm, train, future)
            for section, prefix in (("adapter_source", "src/agent/harness/"), ("supporting_source", "src/agent/")):
                for name, bound_sha in lock[section].items():
                    if protocol["implementation"].get(recorded_path(protocol["implementation"], prefix + name)) != bound_sha:
                        raise ValueError("Runtime source differs from frozen campaign")
            req = audit_requests(path / "model_requests.jsonl")
            events = audit_trace(path / "events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"])
            if run != arm["development" if phase == "develop" else "reuse"]["run"]:
                raise ValueError("Campaign runtime summary differs from stored run")
            discrepancy = runtime_snapshot_discrepancy(run, req, events)
            for name, h in lock["task"]["inputs"].items():
                if file_hash(folder / phase / "workspace" / name) != h:
                    raise ValueError("Bound input changed")
            if arm["status"] == "completed" and (
                run["status"] != "completed" or not req["passed"] or not events["passed"]
            ):
                raise ValueError("Completed arm contains failed runtime")
            record[phase] = {
                "capability_version": cap_id,
                "model_calls": req["model_calls"],
                "tool_calls": events["tool_calls"],
                "request_passed": req["passed"],
                "events_passed": events["passed"],
                "runtime_snapshot_consistent": discrepancy is None,
                "snapshot_discrepancy": discrepancy,
            }
        if "assessment" in arm:
            mid = arm["model_id"]
            registry = folder / "controller"
            VersionStore(registry / "artifacts").verify(mid)
            model = load(registry / "artifacts/versions" / mid / "model.json")
            if digest(classifier.fit_candidate(train, scope, model["recipe"])) != digest(model):
                raise ValueError("Independent training replay differs")
            if model["source_sha256"] != arm["source_sha256"]:
                raise ValueError("Source identity changed")
            predictions, metrics = independent_metrics(model, val, scope)
            assessment = load(registry / ("assessment_" + mid + ".json"))
            if assessment != arm["assessment"]:
                raise ValueError("Assessment differs")
            compare_metrics(assessment["metrics"], metrics)
            if digest(predictions) != assessment["metrics"]["predictions_sha256"]:
                raise ValueError("Acceptance predictions differ")
            passed = metrics[classifier.ACCEPTANCE_METRIC] > 0
            if passed != assessment["passed"]:
                raise ValueError("Promotion gate differs")
            state = load(registry / "state.json")
            if (mid in state["published"]) != passed:
                raise ValueError("Registry publication gate differs")
            record.update(training_replayed=True, acceptance_replayed=True, accepted=passed)
            if passed:
                packet = load(folder / "experience/entries" / (arm["registered_experience_id"] + ".json"))
                if digest(packet) != arm["registered_experience_id"]:
                    raise ValueError("Published experience changed")
                expected_parents = [] if arm["condition"] == "none" else [protocol["donor_id"]]
                if packet["parents"] != expected_parents:
                    raise ValueError("Experience ancestry differs")
            if arm["status"] == "completed":
                future_predictions, future_metrics = independent_metrics(model, future, scope)
                if future_predictions != load(folder / "reuse/workspace/predictions.json"):
                    raise ValueError("Fresh-session probabilities differ")
                compare_metrics(arm["future_metrics"], future_metrics)
                if arm["development"]["run"]["session_id"] == arm["reuse"]["run"]["session_id"]:
                    raise ValueError("Fresh session missing")
                record.update(reuse_replayed=True, future_replayed=True)
        checked.append(record)
    consistent = all(
        phase["runtime_snapshot_consistent"]
        for arm in checked
        for name, phase in arm.items()
        if name in {"develop", "reuse"}
    )
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "passed" if consistent else "evidence_verified_with_runtime_snapshot_discrepancies",
        "runtime_snapshot_consistent": consistent,
        "auditor_sha256": file_hash(Path(__file__)),
        "implementation_root": str(implementation_root),
        "assets": str(assets),
        "campaign": str(campaign),
        "report_sha256": file_hash(campaign / "report.json"),
        "protocol_sha256": file_hash(campaign / "protocol.json"),
        "rows": protocol["split"]["counts"],
        "arms": checked,
        "scope": "independent score calculation and evidence binding; fitting and prediction reuse the frozen numerical implementation; not a new experimental replicate or independent efficacy study",
    }
    (args.output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--implementation-root", type=Path, help="Frozen numerical source root; all protocol hashes remain enforced")
    p.add_argument("--assets", type=Path, help="Public author asset directory, separate from numerical source snapshot")
    audit(p.parse_args())
