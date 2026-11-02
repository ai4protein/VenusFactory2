"""Synthetic numerical and source-bound tests; these do not create research models."""

import copy
import json
from pathlib import Path

import pytest

from agent.rsi import generated_assay as api
from agent.rsi import model_tasks
from agent.rsi.assay_baselines import score_rows
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.storage import atomic_json, sha256_file

CODE = """import numpy as np
def fit(features, targets, seed):
    x = np.asarray([[1.0] + row for row in features], dtype=float)
    return {"weights": np.linalg.lstsq(x, np.asarray(targets), rcond=None)[0].tolist()}
def predict(model, features):
    x = np.asarray([[1.0] + row for row in features], dtype=float)
    return (x @ np.asarray(model["weights"])).tolist()
"""


def fixture_parts():
    parts = {phase: [] for phase in model_tasks.PHASES}
    for group in range(14):
        phase = "training" if group < 8 else "validation" if group < 11 else "future"
        sequence = "ACDEFGHIKLMNPQRSTVWY" + "ACD" * group
        for sample in range(8):
            values = [float(sample), float(group % 4)]
            parts[phase].append(
                {
                    "id": f"observation_{group}_{sample}",
                    "protein": f"synthetic_protein_{group}",
                    "sequence": sequence,
                    "substrate": f"synthetic_substrate_{sample % 2}",
                    "study": f"synthetic_publication_{group}",
                    "group": f"component_{group}",
                    "values": values,
                    "target": 2 + 0.3 * values[0] + 0.2 * values[1],
                }
            )
    return parts


def make_task(root):
    root = Path(root).resolve()
    root.mkdir(parents=True)
    parts = fixture_parts()
    raw = root / "source.json"
    atomic_json(raw, {"synthetic_test_only": True, "rows": parts})
    encoder = root / "encoder.py"
    encoder.write_text("# Synthetic numerical covariates for software validation only.\n")

    def reference(path):
        return {"path": str(path), "sha256": sha256_file(path)}

    scope = {
        "protein": "synthetic_protein_assay_panel",
        "assay": "synthetic observed conditions",
        "target": "synthetic_response",
        "unit": "arbitrary_units",
        "context": {"source": "synthetic_test_only"},
        "interface_version": api.PLUGIN_ID,
        "covariates": [{"name": f"x{i}", "unit": "arbitrary_units", "minimum": 0, "maximum": 100} for i in range(2)],
        "representation": {"name": "synthetic_numeric.v1", "dimensions": 2, "encoder_sha256": sha256_file(encoder)},
        "grouping": {"level": "protein_publication_component", "evidence_sha256": "a" * 64},
    }
    evidence = {
        "schema": "assay_groups.v1",
        "level": scope["grouping"]["level"],
        "study_doi": "synthetic_assay_fixture",
        "sources": [reference(raw)],
        "observations": {
            row["id"]: {"group": row["group"], "phase": phase, "input_sha256": api.input_digest(row)}
            for phase, rows in parts.items()
            for row in rows
        },
    }
    grouping = root / "groups.json"
    atomic_json(grouping, evidence)
    scope["grouping"]["evidence_sha256"] = sha256_file(grouping)
    data = {}
    for phase, rows in parts.items():
        path = root / f"{phase}.json"
        atomic_json(path, rows)
        data[phase] = reference(path)
    definition = {
        "schema": 1,
        "id": "synthetic_assay",
        "stage": "kinetics",
        "plugin_id": api.PLUGIN_ID,
        "scope": scope,
        "paper": {"title": "Synthetic software test", "doi": "synthetic_assay_fixture"},
        "references": [reference(raw), reference(encoder), reference(grouping)],
        "data": data,
        "minimum_skill": 0.0,
    }
    atomic_json(root / "task.json", definition)
    return definition, parts


def test_repeated_variable_length_proteins_and_source_bound_inputs(tmp_path):
    task, parts = make_task(tmp_path / "data")
    assert model_tasks.load_task(task) == parts
    assert len({r["sequence"] for r in parts["training"]}) < len(parts["training"])
    assert len({len(r["sequence"]) for r in parts["training"]}) > 1
    inputs = model_tasks.inference_inputs(parts["future"], api.PLUGIN_ID)
    assert all(set(row) == set(api.INPUT_FIELDS) and "target" not in row for row in inputs)
    fit, dev = api.development_split(parts["training"])
    api.require_disjoint_parts({"fit": fit, "dev": dev}, task["scope"])


@pytest.mark.parametrize("field", ["protein", "sequence", "study"])
def test_renaming_group_cannot_hide_dependency(field, tmp_path):
    task, parts = make_task(tmp_path / "data")
    parts["validation"][0][field] = parts["training"][0][field]
    with pytest.raises(ValueError, match="crosses partitions|multiple sequences across partitions"):
        api.require_disjoint_parts(parts, task["scope"])


def test_protein_identity_is_consistent_across_publication_partitions(tmp_path):
    task, parts = make_task(tmp_path / "data")
    task["scope"]["grouping"]["level"] = "publication"
    parts["validation"][0]["protein"] = parts["training"][0]["protein"]
    with pytest.raises(ValueError, match="multiple sequences across partitions"):
        api.require_disjoint_parts(parts, task["scope"])


@pytest.mark.parametrize(
    "field,value", [("substrate", "changed_substrate"), ("study", "changed_study"), ("values", [99.0, 1.0])]
)
def test_task_rejects_input_tampering_even_if_partition_file_is_rehashed(field, value, tmp_path):
    task, parts = make_task(tmp_path / "data")
    rows = parts["future"]
    rows[0][field] = value
    path = Path(task["data"]["future"]["path"])
    atomic_json(path, rows)
    task["data"]["future"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="source-bound evidence"):
        model_tasks.load_task(task)


def test_encoder_is_bound_and_representation_cannot_change_silently(tmp_path):
    task, _ = make_task(tmp_path / "data")
    task["scope"]["representation"]["encoder_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="encoder source"):
        model_tasks.load_task(task)
    task["scope"]["representation"]["dimensions"] = 3
    with pytest.raises(ValueError, match="representation identity"):
        api.validate_scope(task["scope"])


@pytest.mark.parametrize("change", ["sequence", "group", "nan", "duplicate_id", "target"])
def test_invalid_assay_rows_are_rejected(change, tmp_path):
    task, parts = make_task(tmp_path / "data")
    rows = copy.deepcopy(parts["training"])
    if change == "sequence":
        rows[1]["sequence"] += "A"
    if change == "group":
        rows[1]["group"] = "invented_independence"
    if change == "nan":
        rows[1]["values"][0] = float("nan")
    if change == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    if change == "target":
        rows[1]["target"] = True
    with pytest.raises(ValueError):
        api.validate_rows(rows, task["scope"], labels=True)


def test_real_isolated_fit_assessment_registration_and_fresh_reuse(tmp_path):
    task, parts = make_task(tmp_path / "data")
    plugin = get_model_plugin(api.PLUGIN_ID)
    model = plugin.fit(parts["training"], task["scope"], {"source_code": CODE, "seed": 19})
    # Publication packets traverse JSON storage; exact numerical replay must survive it.
    assert json.loads(json.dumps(model)) == model
    controller = ModelLifecycle(tmp_path / "controller", model_tasks.policy_for_task(task, parts))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, model)
    mid = controller.submit(candidate, session_id="synthetic_scientist", mode="science_expert")
    report = controller.assess(mid, training=parts["training"], validation=parts["validation"])
    assert report["passed"] and report["metrics"]["mse_skill_vs_training_mean"] > 0.999999
    packet = controller.export(mid, tmp_path / "registered")
    assert packet["model"]["plugin_id"] == api.PLUGIN_ID
    inputs = api.inference_inputs(parts["future"])
    predictions = plugin.predict(packet["model"], inputs, task["scope"])
    assert api.prediction_rows_match(predictions, inputs)
    receipt = controller.record_reuse(
        mid,
        session_id="independent_agent",
        mode="science_agent",
        inputs=inputs,
        predictions=predictions,
        scope=task["scope"],
    )
    assert receipt["event"] == "reused"
    damaged = copy.deepcopy(inputs)
    damaged[0]["substrate"] = "different_substrate"
    assert not api.prediction_rows_match(predictions, damaged)
    with pytest.raises(ValueError, match="does not match"):
        controller.record_reuse(
            mid,
            session_id="other_agent",
            mode="science_agent",
            inputs=damaged,
            predictions=predictions,
            scope=task["scope"],
        )


def test_classical_controls_do_not_fit_assessment_labels(tmp_path):
    task, parts = make_task(tmp_path / "data")
    first = score_rows(parts["training"], parts["validation"], task["scope"])
    changed = copy.deepcopy(parts["validation"])
    for row in changed:
        row["target"] += 1000
    second = score_rows(parts["training"], changed, task["scope"])
    assert len(first) == 4 and all(row["status"] == "completed" for row in first)
    assert [r["predictions"] for r in first] == [r["predictions"] for r in second]
    assert first[1]["metrics"] != second[1]["metrics"]
