"""Adversarial checks for the independent campaign evidence auditor."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness.contracts import digest, file_hash
from agent.rsi.versions import VersionStore

spec = importlib.util.spec_from_file_location(
    "nucb_auditor", Path(__file__).resolve().parents[2] / "scripts/rsi/audit_nucb_generated.py"
)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bundle(tmp_path, phase="develop"):
    training = [{"id": "train", "sequence": "ACDE", "target": 1}]
    future = [{"id": "future", "sequence": "ACDF", "target": 0}]
    scope = {"protein": "unit-test-scope"}
    model_binding = {"provider": "test", "model": "test"}
    experience = {"condition": "none", "entries": []}
    protocol = {
        "scope": scope,
        "plugin": "generated_classifier.v1",
        "experience_projections": {"none": digest(experience)},
    }
    arm = {"condition": "none", "model_id": "a" * 64, "assessment": {"passed": True}}
    folder = tmp_path / "arm"
    base = folder / ("development_bundle" if phase == "develop" else "reuse_bundle")
    source = base / "source"
    inputs = (
        {"training.json": training} if phase == "develop" else {"inputs.json": [{"id": "future", "sequence": "ACDF"}]}
    )
    task = {
        "phase": phase,
        "scope": scope,
        "plugin_id": protocol["plugin"],
        "inputs": {n: digest(v) for n, v in inputs.items()},
    }
    if phase == "develop":
        write(source / "experience.json", experience)
        task["experience_sha256"] = digest(experience)
    else:
        model = {"source": "accepted"}
        packet = {
            "model_id": arm["model_id"],
            "model": model,
            "model_sha256": digest(model),
            "assessment": arm["assessment"],
        }
        write(source / "registered/capability.json", packet)
        write(folder / "controller/artifacts/versions" / arm["model_id"] / "model.json", model)
        task["capability_sha256"] = digest(packet)
    write(source / "task.json", task)
    (source / "model.py").write_text("# frozen source fixture")
    protocol["implementation"] = {"src/agent/rsi/model.py": file_hash(source / "model.py")}
    store = VersionStore(base / "bundles")
    manifest = store.snapshot(source, metadata={"phase": phase, "model": model_binding})
    cap = base / "bundles/versions" / manifest["version_id"]
    for name, rows in inputs.items():
        write(folder / phase / "workspace" / name, rows)
    lock = {
        "profile": {"model": model_binding},
        "task": {
            "inputs": {name: file_hash(folder / phase / "workspace" / name) for name in inputs},
            "capabilities": {str(p.relative_to(cap)): file_hash(p) for p in cap.rglob("*") if p.is_file()},
        },
    }
    return folder, phase, lock, protocol, arm, training, future


@pytest.mark.parametrize("phase", ["develop", "reuse"])
def test_bound_capability_and_label_free_reuse(tmp_path, phase):
    assert len(auditor.verify_bundle(*bundle(tmp_path, phase))) == 64


def test_wrong_experience_condition_rejected(tmp_path):
    args = bundle(tmp_path)
    args[3]["experience_projections"]["none"] = digest({"condition": "source"})
    with pytest.raises(ValueError, match="Experience projection"):
        auditor.verify_bundle(*args)


def test_extra_label_input_rejected(tmp_path):
    args = bundle(tmp_path, "reuse")
    args[2]["task"]["inputs"]["future.json"] = digest(args[-1])
    with pytest.raises(ValueError, match="Unexpected runtime input"):
        auditor.verify_bundle(*args)


def test_bound_input_content_substitution_rejected(tmp_path):
    args = bundle(tmp_path, "reuse")
    write(args[0] / "reuse/workspace/inputs.json", [{"id": "future", "sequence": "ACDF", "target": 0}])
    with pytest.raises(ValueError, match="Observed inputs"):
        auditor.verify_bundle(*args)


def test_runtime_capability_substitution_rejected(tmp_path):
    args = bundle(tmp_path)
    args[2]["task"]["capabilities"]["task.json"] = "0" * 64
    with pytest.raises(ValueError, match="Runtime capability package"):
        auditor.verify_bundle(*args)


def test_different_accepted_model_rejected(tmp_path):
    args = bundle(tmp_path, "reuse")
    write(args[0] / "controller/artifacts/versions" / args[4]["model_id"] / "model.json", {"source": "substituted"})
    with pytest.raises(ValueError, match="independently accepted model"):
        auditor.verify_bundle(*args)


def test_live_campaign_cannot_receive_terminal_audit(tmp_path):
    write(tmp_path / "report.json", {"status": "running"})
    write(tmp_path / "protocol.json", {})
    with pytest.raises(ValueError, match="Terminal campaign required"):
        auditor.audit(SimpleNamespace(campaign=tmp_path, output=tmp_path / "audit"))
    assert not (tmp_path / "audit").exists()


def test_self_consistent_but_unplanned_source_rejected(tmp_path):
    args = bundle(tmp_path)
    args[3]["implementation"]["src/agent/rsi/model.py"] = "0" * 64
    with pytest.raises(ValueError, match="Capability implementation"):
        auditor.verify_bundle(*args)


def snapshot_fixture():
    requests = {"passed": False, "model_calls": 7, "errors": ["transport_error"], "usage": []}
    events = {"passed": True, "tool_calls": 3, "tool_errors": 0, "problems": []}
    return {"status": "timeout", "audit": {"requests": requests, "events": events}}


def test_late_timeout_errors_are_retained_as_a_discrepancy():
    run = snapshot_fixture()
    assert auditor.runtime_snapshot_discrepancy(run, run["audit"]["requests"], run["audit"]["events"]) is None
    terminal = {**run["audit"]["requests"], "errors": ["transport_error", "transport_error"]}
    result = auditor.runtime_snapshot_discrepancy(run, terminal, run["audit"]["events"])
    assert result["additional_transport_errors"] == 1
    assert result["recorded_requests"] != result["terminal_requests"]
    assert not result["recorded_requests"]["passed"] and not result["terminal_requests"]["passed"]


@pytest.mark.parametrize(
    "change",
    [
        {"passed": True},
        {"model_calls": 8},
        {"usage": [{"total_tokens": 1}]},
        {"errors": []},
        {"errors": ["policy_rejected", "transport_error"]},
        {"errors": ["transport_error", "budget_rejected"]},
    ],
)
def test_other_runtime_snapshot_drift_remains_fatal(change):
    run = snapshot_fixture()
    terminal = {**run["audit"]["requests"], **change}
    with pytest.raises(ValueError, match="Runtime request audit differs"):
        auditor.runtime_snapshot_discrepancy(run, terminal, run["audit"]["events"])


def test_completed_runtime_cannot_explain_away_appended_errors():
    run = snapshot_fixture()
    run["status"] = "completed"
    terminal = {**run["audit"]["requests"], "errors": ["transport_error", "transport_error"]}
    with pytest.raises(ValueError, match="Runtime request audit differs"):
        auditor.runtime_snapshot_discrepancy(run, terminal, run["audit"]["events"])
    with pytest.raises(ValueError, match="Runtime event audit differs"):
        auditor.runtime_snapshot_discrepancy(run, terminal, {**run["audit"]["events"], "tool_calls": 4})


def test_development_guidance_is_bound_to_observed_runtime():
    protocol = {
        "development_guidance": "compact_probability",
        "development_prompt": "fixed task",
        "development_prompt_sha256": digest("fixed task"),
        "development_instructions_sha256": digest("fixed instructions"),
    }
    lock = {"task": {"prompt_sha256": digest("fixed task"), "instructions_sha256": digest("fixed instructions")}}
    auditor.verify_development_guidance(protocol, lock)
    lock["task"]["prompt_sha256"] = digest("different instruction")
    with pytest.raises(ValueError, match="prompt differs"):
        auditor.verify_development_guidance(protocol, lock)
    lock["task"]["prompt_sha256"] = digest("fixed task")
    lock["task"]["instructions_sha256"] = digest("different system")
    with pytest.raises(ValueError, match="instructions differ"):
        auditor.verify_development_guidance(protocol, lock)


def test_explicit_frozen_root_keeps_source_hash_gate(tmp_path):
    campaign = tmp_path / "campaign"
    frozen = tmp_path / "elsewhere" / "source"
    relative = "src/agent/rsi/model.py"
    for root in (campaign / "implementation", frozen):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text("verified implementation")
    protocol = {"implementation": {relative: file_hash(frozen / relative)}}
    auditor.verify_implementation(protocol, campaign, frozen)
    (frozen / relative).write_text("changed numerical implementation")
    with pytest.raises(ValueError, match="frozen campaign implementation"):
        auditor.verify_implementation(protocol, campaign, frozen)
