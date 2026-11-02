import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_generated_assay import make_task
from test_sabio_assay_tasks import synthetic_sources

from agent.rsi import assay_baseline_refit as refit
from agent.rsi import assay_baselines, sabio_assay_tasks, sabio_domain_controls, shared_assay_evaluation
from agent.rsi.engine.storage import atomic_json, sha256_file


def enrich(controls, rows):
    result = copy.deepcopy(controls)
    for control in result:
        for prediction, row in zip(control["predictions"], rows, strict=True):
            prediction.update(protein=row["protein"], group=row["group"])
    return result


def test_refit_reconstructs_controls_without_production_fitting(tmp_path, monkeypatch):
    task, parts = make_task(tmp_path / "data")
    report = shared_assay_evaluation.evaluate_task(task, None)
    monkeypatch.setattr(assay_baselines, "score_rows", lambda *a, **kw: pytest.fail("Production fit called"))
    monkeypatch.setattr(
        sabio_domain_controls, "predictions", lambda *a, **kw: pytest.fail("Production reference called")
    )
    result = refit.verify_controls(task, report["controls"])
    assert result["baseline_refit_performed"] and len(result["methods"]) == 4
    assert result["maximum_absolute_prediction_error"] < 1e-9
    assert not result["descriptor_encoding_reimplemented"]


def test_validation_refit_does_not_read_final_partition(tmp_path):
    task, parts = make_task(tmp_path / "data")
    controls = enrich(assay_baselines.run_controls(task)["outcomes"], parts["validation"])
    Path(task["data"]["future"]["path"]).unlink()
    assert refit.verify_controls(task, controls, partition="validation")["score_partition"] == "validation"


@pytest.mark.parametrize("change", ["missing", "duplicate", "failure", "prediction", "input", "row_order"])
def test_incomplete_or_corrupted_controls_do_not_pass(tmp_path, change):
    task, _ = make_task(tmp_path / "data")
    controls = shared_assay_evaluation.evaluate_task(task, None)["controls"]
    if change == "missing":
        controls.pop()
    elif change == "duplicate":
        controls[-1] = copy.deepcopy(controls[0])
    elif change == "failure":
        controls[0]["status"] = "failed"
    elif change == "prediction":
        controls[0]["predictions"][0]["prediction"] += 1
    elif change == "input":
        controls[0]["predictions"][0]["inputs_sha256"] = "0" * 64
    else:
        controls[0]["predictions"].reverse()
    with pytest.raises(ValueError):
        refit.verify_controls(task, controls)


def test_changed_training_file_rejected(tmp_path):
    task, _ = make_task(tmp_path / "data")
    controls = shared_assay_evaluation.evaluate_task(task, None)["controls"]
    path = Path(task["data"]["training"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Bound input changed"):
        refit.verify_controls(task, controls)


def test_queries_are_unlabelled_and_scaling_uses_training_only(tmp_path):
    _, parts = make_task(tmp_path / "data")
    queries = refit.query_rows(parts["validation"])
    with pytest.raises(ValueError, match="unlabelled"):
        refit.refit_predictions(parts["training"], parts["validation"])
    original = refit.refit_predictions(parts["training"], queries[:2])
    extended = copy.deepcopy(queries)
    extended[-1]["values"] = [1000000.0, -1000000.0]
    after = refit.refit_predictions(parts["training"], extended)
    for method in original:
        np.testing.assert_allclose(
            [r["prediction"] for r in original[method]],
            [r["prediction"] for r in after[method][:2]],
            rtol=0,
            atol=1e-10,
        )


@pytest.mark.parametrize("field", ["id", "protein", "sequence", "study", "group"])
def test_overlap_is_rejected_even_with_renamed_groups(tmp_path, field):
    _, parts = make_task(tmp_path / "data")
    queries = refit.query_rows(parts["validation"])
    queries[0][field] = parts["training"][0][field]
    with pytest.raises(ValueError, match="Partition overlap"):
        refit.refit_predictions(parts["training"], queries)


def test_constant_training_labels_are_finite(tmp_path):
    _, parts = make_task(tmp_path / "data")
    for row in parts["training"]:
        row["target"] = 7.0
    results = refit.refit_predictions(parts["training"], refit.query_rows(parts["validation"]))
    assert all(row["prediction"] == 7.0 for rows in results.values() for row in rows)


def test_domain_reference_weights_groups_and_deduplicates_neighbours():
    training = [{"id": f"a{i}", "group": "a", "target": 10.0} for i in range(100)]
    training += [{"id": "b", "group": "b", "target": 0.0}, {"id": "c", "group": "c", "target": 2.0}]
    inputs = [{"id": "query"}, {"id": "unseen"}]
    common = {"ec": ("1", "1", "1", "1"), "substrate": "s", "ph": (7, 7, False), "temperature": (25, 25, False)}
    metadata = {row["id"]: copy.deepcopy(common) for row in training + inputs}
    metadata["unseen"]["ec"] = ("7", "9", "9", "9")
    result = refit.domain_predictions(training, inputs, metadata)
    assert result[refit.DOMAIN[0]][0] == (2.0, "ec_prefix_4", 3)
    assert result[refit.DOMAIN[0]][1] == (2.0, "global_group_median", 3)
    assert result[refit.DOMAIN[1]][0] == (2.0, "same_ec_prefix_2_substrate_nearest_conditions", 3)
    assert result[refit.DOMAIN[1]][1][0] == 2.0


@pytest.fixture(scope="module")
def source_tasks(tmp_path_factory):
    root = tmp_path_factory.mktemp("refit_domain")
    source, sequences, _, _ = synthetic_sources(root)
    manifest = sabio_assay_tasks.prepare(source, [sequences], [], root / "prepared")
    return [json.loads(Path(row["path"]).read_text()) for row in manifest["tasks"]]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_all_six_controls_agree_for_each_assay_endpoint(source_tasks, index):
    task = source_tasks[index]
    rows = refit.read_bound(task["data"]["validation"])
    controls = enrich(assay_baselines.run_controls(task)["outcomes"], rows)
    result = refit.verify_controls(task, controls, partition="validation")
    assert len(result["methods"]) == 6 and result["source_metadata_reconstructed"]
    assert result["maximum_absolute_prediction_error"] < 1e-9


def test_intervals_preserve_missing_ranges_and_reject_nonfinite():
    assert refit.condition_interval(None) == (0, 0, True)
    assert refit.condition_interval({"start_value": 2, "end_value": 4}) == (2, 4, False)
    assert refit._distance((2, 4, False), (3, 6, False), 14) == 0
    assert refit._distance((0, 0, True), (3, 6, False), 14) == 1
    with pytest.raises(ValueError):
        refit.condition_interval({"start_value": float("nan")})


def test_cli_binds_report_and_refits_all_allocations(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/rsi/evaluation/refit_assay_baselines.py"
    spec = importlib.util.spec_from_file_location("refit_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    task, _ = make_task(tmp_path / "data")
    frozen = shared_assay_evaluation.freeze_protocol([task], {str(script): sha256_file(script)})
    frozen_path = tmp_path / "frozen.json"
    atomic_json(frozen_path, frozen)
    evaluation = tmp_path / "synthetic_evaluation"
    evaluation.mkdir()
    row = shared_assay_evaluation.evaluate_task(task, None)
    task_path = evaluation / (task["id"] + ".json")
    atomic_json(task_path, row)
    terminal = tmp_path / "synthetic_terminal.json"
    atomic_json(terminal, {"synthetic_software_test_only": True})
    protocol = evaluation / "protocol.json"
    atomic_json(
        protocol,
        {
            "frozen_protocol": str(frozen_path),
            "frozen_protocol_sha256": sha256_file(frozen_path),
            "terminal_and_reuse_bindings": {str(terminal): sha256_file(terminal)},
        },
    )
    report = evaluation / "report.json"
    atomic_json(
        report,
        {
            "schema": shared_assay_evaluation.SCHEMA,
            "final_labels_scored": True,
            "protocol_sha256": sha256_file(protocol),
            "task_reports": {task["id"]: sha256_file(task_path)},
            "totals": shared_assay_evaluation.summarize([row]),
        },
    )
    args = SimpleNamespace(evaluation=evaluation, report_sha256=sha256_file(report), output=tmp_path / "refit.json")
    cli.main(args)
    outcome = json.loads(args.output.read_text())
    assert outcome["status"] == "independent_assay_baseline_refit_passed" and outcome["completed_refits"] == 4
    assert outcome["new_autonomous_models"] == outcome["external_model_calls"] == 0
    terminal.write_text("{}")
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        cli.main(SimpleNamespace(**{**vars(args), "output": tmp_path / "rejected.json"}))
    assert not (tmp_path / "rejected.json").exists()
