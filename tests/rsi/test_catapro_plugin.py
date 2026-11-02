"""Full-cohort admission and resource handling for the CataPro plugin."""

import csv
import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from agent.rsi import catapro_evidence as evidence
from agent.rsi.engine.case_evidence import canonical_claims
from agent.rsi.campaigns.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.cases.catapro_case import file_sha
from agent.rsi.engine.reproduction_goal_contract import allocation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.run_catapro_case import gpu_lease


def call(root, parameters):
    return invocation(
        "catapro.full_cv.v1",
        parameters,
        root=root,
        output=root / "research_runs/rsi/new",
        controller_python=Path("/python"),
    )


def test_plugin_admits_only_full_allocations_and_distinguishes_origins(tmp_path):
    assert call(tmp_path, {"operation": "reproduce"})[-2:] == ["--operation", "reproduce"]
    params = {"operation": "verify_existing", "campaign": "catapro_full_20261017"}
    assert call(tmp_path, params)[-4:] == ["--operation", "verify_existing", "--campaign", "catapro_full_20261017"]
    assert resource_for("catapro.full_cv.v1", params) == "cpu"
    assert resource_for("catapro.full_cv.v1", {"operation": "reproduce"}) == "gpu"
    for wrong in [
        {},
        {"operation": "reproduce", "dataset": "sample"},
        {"operation": "verify_existing", "campaign": "../other"},
        {"operation": "reproduce", "campaign": "catapro_full_20261017"},
    ]:
        with pytest.raises(ValueError):
            call(tmp_path, wrong)
    plugin = next(p for p in list_case_plugins() if p["id"] == "catapro.full_cv.v1")
    assert plugin["parameters"]["additionalProperties"] is False
    assert allocation("catapro_full_cohorts")["parameters"] == {"operation": "reproduce"}
    # Evidence inspection remains available to both modes, but cannot satisfy
    # the long-goal requirement for new computation and fresh execution.
    with pytest.raises(ValueError, match="Unknown"):
        allocation("catapro_full_evidence_reuse")


def facts_report():
    datasets = {}
    for name, rows in evidence.COUNTS.items():
        base, remainder = divmod(rows, 10)
        datasets[name] = {
            "rows": rows,
            "independent_predictions_passed": True,
            "folds": [{"fold": f, "rows": base + (f < remainder)} for f in range(10)],
            "metrics": {"rows": rows, "pcc": -0.2, "scc": -0.1, "rmse": 1.2},
        }
    return {
        "status": "completed",
        "execution_origin": "verified_existing",
        "source_campaign": "research_runs/rsi/fixture",
        "audit_evidence": {
            "source_commit": evidence.COMMIT,
            "total_rows": 95507,
            "fold_checkpoints_replayed": 30,
            "all_rows_independently_replayed": True,
            "datasets": datasets,
        },
    }


def test_full_scientific_claims_preserve_negative_results_and_scope():
    facts = canonical_claims("catapro.full_cv.v1", facts_report())
    assert facts["metrics"]["kcat"]["pcc"] == -0.2
    assert facts["training_reproduced"] is False
    assert facts["new_model_or_self_evolution_established"] is False
    assert facts["full_paper_reproduced"] is False
    assert facts["author_validation_used_for_checkpoint_selection"] is True


@pytest.mark.parametrize("mutation", ["missing_cohort", "missing_fold", "wrong_rows", "nan", "wrong_source"])
def test_incomplete_or_invalid_evidence_cannot_be_registered(mutation):
    report = facts_report()
    audit = report["audit_evidence"]
    if mutation == "missing_cohort":
        del audit["datasets"]["activity"]
    elif mutation == "missing_fold":
        audit["datasets"]["km"]["folds"].pop()
    elif mutation == "wrong_rows":
        audit["datasets"]["kcat"]["rows"] = 5
    elif mutation == "nan":
        audit["datasets"]["kcat"]["metrics"]["rmse"] = float("nan")
    else:
        audit["source_commit"] = "different"
    with pytest.raises(ValueError):
        canonical_claims("catapro.full_cv.v1", report)


def test_inherited_gpu_lease_is_not_released_by_child_context(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    with path.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("VENUS_RSI_CASE_LOCK_FD", str(owner.fileno()))
        with gpu_lease(path):
            pass
        with path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wrong = tmp_path / "wrong.lock"
        wrong.touch()
        with pytest.raises(ValueError, match="different"):
            with gpu_lease(wrong):
                pass


def fixture_cohort(tmp_path, monkeypatch):
    import torch

    source, folder = tmp_path / "source", tmp_path / "cohort"
    (source / "datasets").mkdir(parents=True)
    folder.mkdir()
    raw = source / "datasets" / evidence.DATASETS["kcat"]
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["", "Sequence", "Smiles", "fold", "kcat(s^-1)"])
        writer.writeheader()
        for i in range(40):
            writer.writerow(
                {"": str(i), "Sequence": "ACD", "Smiles": "CCO", "fold": i % 10, "kcat(s^-1)": float(10 ** (i / 100))}
            )
    for name in ["admission.json", "checkpoints_admission_api.json"]:
        (source / name).write_text("{}")
    weights = {}
    for fold in range(10):
        name = "models/kcat_models/" + str(fold) + "_bestmodel.pth"
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-fixture")
        weights[name] = file_sha(path)
    predicted = (np.arange(40) / 100 + 0.01 * (np.arange(40) % 3)).astype("f4")
    features = np.zeros((40, 1959), dtype="f4")
    features[:, 0] = predicted
    np.save(folder / "features.npy", features)
    np.save(folder / "predictions.npy", predicted[:, None])
    rows = evidence.dataset_rows(source, "kcat")
    metric = evidence.regression_metrics(predicted, np.array([r["targets"]["kcat"] for r in rows]))
    protocol = {
        "rows": 40,
        "dataset": "kcat",
        "source_commit": evidence.COMMIT,
        "input_sha256": file_sha(raw),
        "source_admission_sha256": file_sha(source / "admission.json"),
        "checkpoint_admission_sha256": file_sha(source / "checkpoints_admission_api.json"),
        "checkpoints": weights,
    }
    report = {
        "status": "completed",
        "dataset": "kcat",
        "rows_encoded": 40,
        "features_sha256": file_sha(folder / "features.npy"),
        "predictions_sha256": file_sha(folder / "predictions.npy"),
        "result": {"metrics": metric},
    }
    for name, value in [
        ("protocol.json", protocol),
        ("report.json", report),
        ("results.json", {}),
        ("checkpoint_parity.json", []),
    ]:
        (folder / name).write_text(json.dumps(value))
    monkeypatch.setitem(evidence.COUNTS, "kcat", 40)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {})
    monkeypatch.setattr(evidence, "numpy_forward", lambda x, state, activity=False: x[:, 0])
    return source, folder


def test_audit_replays_every_row_and_rejects_changed_saved_prediction(tmp_path, monkeypatch):
    source, folder = fixture_cohort(tmp_path, monkeypatch)
    checked = evidence.audit_cohort(folder, source, "kcat")
    assert checked["rows"] == 40 and len(checked["folds"]) == 10
    pred = np.load(folder / "predictions.npy")
    pred[-1, 0] += 0.05
    np.save(folder / "predictions.npy", pred)
    report = json.loads((folder / "report.json").read_text())
    # Even a report with a refreshed byte hash cannot bypass numerical replay.
    report["predictions_sha256"] = file_sha(folder / "predictions.npy")
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replay differs"):
        evidence.audit_cohort(folder, source, "kcat")


def test_running_campaign_cannot_be_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "checked_manifest", lambda *args: {"commit": evidence.COMMIT})
    (tmp_path / "execution.json").write_text('{"status":"running"}')
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="terminal"):
        evidence.audit_campaign(tmp_path, tmp_path)
