import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_context_v2 import make_task, save

from agent.rsi import context_baseline_refit as refit
from agent.rsi import context_controls, context_v2_controls, context_v3_controls, shared_context_evaluation


def fixture(kind="ordinary"):
    scope = {
        "protein": "synthetic",
        "assay": "response",
        "target": "measured",
        "unit": "ratio",
        "context": {"preparation": "synthetic fixture"},
        "grouping": {"level": "biological_replicate", "evidence_sha256": "0" * 64},
        "covariates": [
            {"name": "temperature" if kind != "numeric" else "input", "unit": "degC", "minimum": 0, "maximum": 100}
        ],
    }
    if kind == "reference":
        scope["unit"] = "ratio_to_within_replicate_reference"
        scope["covariates"].append({"name": "reference_temperature", "unit": "degC", "minimum": 0, "maximum": 100})
    rows = []
    for replicate in range(3):
        for index, temperature in enumerate(range(20, 71, 5)):
            target = 0.1 + 0.9 / (1 + __import__("math").exp((temperature - 45) / 5))
            if kind == "reference":
                target /= 0.1 + 0.9 / (1 + __import__("math").exp((20 - 45) / 5))
            rows.append(
                {
                    "id": f"r{replicate}_{index}",
                    "protein": "synthetic",
                    "group": f"g{replicate}",
                    "values": [temperature, 20] if kind == "reference" else [temperature],
                    "target": target,
                }
            )
    return scope, rows[:22], rows[22:]


@pytest.mark.parametrize("kind", ["numeric", "ordinary", "reference"])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_independent_refits_match_production_without_calling_it(kind, version, monkeypatch):
    scope, training, future = fixture(kind)
    controls = {1: context_controls, 2: context_v2_controls, 3: context_v3_controls}[version]
    if version > 1:
        scope["interface_version"] = f"generated_context.v{version}"
    saved = controls.score_rows(training, future, scope)
    monkeypatch.setattr(controls, "score_rows", lambda *a, **k: pytest.fail("Production evaluator called"))
    monkeypatch.setattr(context_controls, "score_validated_rows", lambda *a, **k: pytest.fail("Shared evaluator called"))
    actual = refit.refit_predictions(training, refit.query_rows(future), scope)
    assert set(actual) == {r["method"] for r in saved}
    for row in saved:
        assert actual[row["method"]]["status"] == "completed"
        assert [r["prediction"] for r in actual[row["method"]]["predictions"]] == pytest.approx(
            [r["prediction"] for r in row["predictions"]], rel=1e-7, abs=1e-9
        )


@pytest.mark.parametrize("change", ["label", "group", "id", "protein", "dimension", "nan", "domain", "duplicate"])
def test_invalid_or_leaking_queries_fail(change):
    scope, training, future = fixture("numeric")
    query = refit.query_rows(future)
    if change == "label":
        query[0]["target"] = 123
    elif change in {"group", "id"}:
        query[0][change] = training[0][change]
    elif change == "protein":
        query[0]["protein"] = "another"
    elif change == "dimension":
        query[0]["values"] = [1, 2]
    elif change == "nan":
        query[0]["values"] = [float("nan")]
    elif change == "domain":
        query[0]["values"] = [200]
    elif change == "duplicate":
        query.append(copy.deepcopy(query[0]))
    with pytest.raises(ValueError):
        refit.refit_predictions(training, query, scope)


def test_constant_training_and_multiple_query_conditions():
    scope, training, future = fixture("numeric")
    for row in training:
        row["target"] = 3.5
    result = refit.refit_predictions(training, refit.query_rows(future), scope)
    assert all(r["status"] == "completed" for r in result.values())
    assert all(v["prediction"] == 3.5 for r in result.values() for v in r["predictions"])


def test_failed_thermal_fit_is_preserved():
    scope, training, future = fixture("reference")
    training[0]["target"] = -1
    result = refit.refit_predictions(training, refit.query_rows(future), scope)
    assert all(result[n]["status"] == "completed" for n in refit.NUMERICAL)
    assert all(result[n]["status"] == "failed" for n in refit.thermal_spec(scope)[0])


@pytest.mark.parametrize("change", ["prediction", "id", "missing", "duplicate", "status", "group"])
def test_saved_corruption_rejected(tmp_path, change):
    task = make_task(tmp_path / "case")[0]
    result = shared_context_evaluation.evaluate_task(task, None)
    controls = copy.deepcopy(result["controls"])
    if change == "prediction":
        controls[0]["predictions"][0]["prediction"] += 1
    elif change == "id":
        controls[0]["predictions"][0]["id"] = "other"
    elif change == "missing":
        controls.pop()
    elif change == "duplicate":
        controls[-1] = copy.deepcopy(controls[0])
    elif change == "status":
        controls[0] = {"method": controls[0]["method"], "status": "failed"}
    elif change == "group":
        rows = refit.read_bound(task["data"]["future"])
        rows[0]["group"] = refit.read_bound(task["data"]["validation"])[0]["group"]
        task["data"]["future"] = save(Path(task["data"]["future"]["path"]), rows)
    with pytest.raises(ValueError):
        refit.verify_controls(task, controls)


def test_future_labels_do_not_affect_refitting(tmp_path):
    task = make_task(tmp_path / "case")[0]
    saved = shared_context_evaluation.evaluate_task(task, None)
    first = refit.verify_controls(task, saved["controls"])
    rows = refit.read_bound(task["data"]["future"])
    for row in rows:
        row["target"] += 10000
    task["data"]["future"] = save(Path(task["data"]["future"]["path"]), rows)
    second = refit.verify_controls(task, saved["controls"])
    assert first["prediction_sha256"] == second["prediction_sha256"]


def test_cli_complete_coverage_and_stale_report(tmp_path):
    task = make_task(tmp_path / "case")[0]
    result = shared_context_evaluation.evaluate_task(task, None)
    root = tmp_path / "evaluation"
    root.mkdir()
    bound = save(root / (task["id"] + ".json"), result)
    evidence = save(root / "evidence.json", {"fixture": True})
    protocol = {
        "fit_partition": "training",
        "score_partition": "future",
        "hyperparameter_search": False,
        "source_bindings": {evidence["path"]: evidence["sha256"]},
        "implementation": {},
        "tasks": [task],
        "task_bindings": {task["id"]: refit.canonical(task)},
    }
    protocol_ref = save(root / "protocol.json", protocol)
    report = {
        "schema": "shared_context_final_evaluation.v1",
        "final_labels_scored": True,
        "protocol_sha256": protocol_ref["sha256"],
        "task_reports": {task["id"]: bound["sha256"]},
        "totals": {"allocated": 1, "completed_control_fits": 4, "failed_control_fits": 0},
    }
    report_ref = save(root / "report.json", report)
    cli = Path(__file__).resolve().parents[2] / "scripts/rsi/evaluation/refit_context_baselines.py"
    spec = importlib.util.spec_from_file_location("context_refit_cli_test", cli)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(evaluation=root, report_sha256=report_ref["sha256"], output=root / "refit.json")
    output = module.main(args)
    assert output["completed_refits"] == 4 and output["external_model_calls"] == 0
    args.output = root / "refit_again.json"
    report["totals"]["allocated"] = 2
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Bound input changed"):
        module.main(args)


@pytest.mark.parametrize("change", ["valid", "wrong_digest", "source_override", "conflict"])
def test_historical_implementation_archive_is_exact_and_code_only(tmp_path, change):
    cli = Path(__file__).resolve().parents[2] / "scripts/rsi/evaluation/refit_context_baselines.py"
    spec = importlib.util.spec_from_file_location("context_archive_cli_test", cli)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = str(tmp_path / "original.py")
    archive = save(tmp_path / "historical.py", {"original_bytes": True})
    protocol = {"source_bindings": {}, "implementation": {original: archive["sha256"]}}
    reference = copy.deepcopy(archive)
    if change == "wrong_digest":
        reference["sha256"] = "0" * 64
    if change == "source_override":
        protocol["source_bindings"][original] = archive["sha256"]
    if change == "conflict":
        protocol["source_bindings"][original] = "0" * 64
    manifest = save(
        tmp_path / "archive.json", {"schema": "context_implementation_archive.v1", "files": {original: reference}}
    )
    if change == "valid":
        assert module.evidence_bindings(protocol, manifest)[original] == archive
    else:
        with pytest.raises(ValueError):
            module.evidence_bindings(protocol, manifest)
