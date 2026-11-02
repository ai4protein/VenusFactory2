"""Synthetic public-table fixtures; these are never research model publications."""

import csv
import hashlib
import json

import numpy as np
import pytest

from agent.rsi import arabidopsis_context as source
from agent.rsi.model_tasks import read_task
from agent.rsi.reference_thermal_baselines import METHODS, predict, ratio_curve
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.traditional_baselines import run_controls


@pytest.fixture
def raw_fixture(tmp_path):
    raw, methods = tmp_path / "raw", tmp_path / "methods"
    raw.mkdir()
    methods.mkdir()
    (methods / "methods.Rnw").write_text("Synthetic source methods; test fixture only")
    channels = [f"TMT10-{n}" for n in ["126", "127N", "127C", "128N", "128C", "129N", "129C", "130N", "130C", "131"]]
    tree, retrieved, metadata_records = {}, [], {}
    for replicate in source.REPLICATES:
        temperatures = (
            [21.8 + 4 * i for i in range(10)] if replicate in {"C1", "C2"} else [24.8 + 4 * i for i in range(10)]
        )
        if replicate in {"C2", "C4", "C6"}:
            temperatures.reverse()
        metadata = methods / f"R/meta/{replicate}.meta"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            "channel\ttemp\n"
            + "".join(f"{c.replace('TMT10-', 'TMT10.')}\t{t}\n" for c, t in zip(channels, temperatures))
        )
        metadata_records[replicate] = {
            "sha256": sha256_file(metadata),
            "channel_names": [c.replace("TMT10-", "TMT10.") for c in channels],
            "temperatures": temperatures,
        }
        relative = f"R/data.rmbaseline/{replicate}.pp.f0.01.tsv"
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write("## synthetic filtered PSM input\n")
            writer = csv.DictWriter(
                handle, fieldnames=["id", "protein", "peptide", "coelute_inf", *channels], delimiter="\t"
            )
            writer.writeheader()
            for protein in [source.BSA, "AT1G00010", "AT1G00020", "AT1G00010|AT1G00020", "CONTAMINANT"]:
                count = 5 if protein == "AT1G00020" and replicate == "C6" else 10
                for number in range(count):
                    values = [100 * (i + 1) for i in range(10)]
                    if protein != source.BSA:
                        shape = 0.1 + 0.9 / (1 + np.exp((np.asarray(temperatures) - 48) / 5))
                        values = (np.asarray(values) * shape * 2).tolist()
                    writer.writerow(
                        {
                            "id": f"{protein}_{number}",
                            "protein": protein,
                            "peptide": "P" + str(number % 2),
                            "coelute_inf": 0.1,
                            **dict(zip(channels, values)),
                        }
                    )
        data = path.read_bytes()
        sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        tree[relative] = {"sha": sha, "size": len(data)}
        retrieved.append({"path": relative, "sha256": sha256_file(path)})
    atomic_json(raw / "source_tree_selection.json", tree)
    atomic_json(raw / "retrieval_api.json", retrieved)
    atomic_json(
        raw / "protocol.json",
        {
            "source_commit": source.COMMIT,
            "paper_doi": source.DOI,
            "proposed_partition_before_model_fitting": source.protocol()["partition"],
        },
    )
    atomic_json(methods / "temperature_mapping.json", metadata_records)
    return raw, methods


def test_real_table_export_and_catalog_admission(raw_fixture, tmp_path):
    raw, methods = raw_fixture
    output = tmp_path / "prepared"
    report = source.export_tasks(raw, methods, output)
    assert report["training_qualified_proteins"] == 2
    assert report["exported_tasks"] == 1 and report["catalog_statuses"] == {"admitted": 1}
    assert report["excluded_training_candidates"] == [{"protein": "AT1G00020", "missing_qualified_replicates": ["C6"]}]
    task, data = read_task(output / "tasks/at1g00010/task.json")
    assert [len(data[p]) for p in source.PARTITIONS] == [40, 10, 10]
    assert data["future"][0]["values"] == [24.8, 24.8, 0.0]
    assert data["future"][0]["target"] == 1.0
    expected = ratio_curve(np.stack([[60.8], [24.8]]), 0.1, 48.0, 5.0)[0]
    assert data["future"][-1]["target"] == pytest.approx(expected)
    assert {row["values"][2] for row in data["training"]} == {0.0, 1.0}
    controls = run_controls(task)
    outcomes = {r["method"]: r for r in controls["outcomes"]}
    assert set(METHODS) <= set(outcomes)
    assert all(row["status"] == "completed" for row in outcomes.values())
    assert outcomes["thermal_reference_sigmoid"]["metrics"]["rmse"] < 1e-5
    assert outcomes["thermal_reference_sigmoid"]["non_reference_metrics"]["rows"] == 9
    assert not controls["future_scored"]


@pytest.mark.parametrize("changed", ["raw", "metadata", "partition"])
def test_export_rejects_source_changes(raw_fixture, tmp_path, changed):
    raw, methods = raw_fixture
    if changed == "raw":
        path = raw / "R/data.rmbaseline/C1.pp.f0.01.tsv"
        path.write_text(path.read_text() + "\n")
    elif changed == "metadata":
        path = methods / "R/meta/C1.meta"
        path.write_text(path.read_text().replace("21.8", "22.8"))
    else:
        path = raw / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["proposed_partition_before_model_fitting"]["validation"] = ["C6"]
        atomic_json(path, protocol)
    with pytest.raises(ValueError):
        source.export_tasks(raw, methods, tmp_path / "prepared")


def test_reference_uses_temperature_mapping_and_channel_control(raw_fixture):
    raw, methods = raw_fixture
    channels, temperatures = source.read_mapping(methods / "R/meta/C6.meta")
    proteins, counts = source.aggregate_psms(raw / "R/data.rmbaseline/C6.pp.f0.01.tsv", channels)
    assert counts["excluded_ambiguous_or_nonplant"] == 20
    rows = source.normalized_observations("AT1G00010", "C6", proteins, channels, temperatures)
    assert rows[0]["target"] == 1 and rows[0]["id"].endswith("TMT10-131")
    for item in proteins.values():
        item["intensities"] = [v * (i + 2) for i, v in enumerate(item["intensities"])]
    again = source.normalized_observations("AT1G00010", "C6", proteins, channels, temperatures)
    assert [r["target"] for r in again] == pytest.approx([r["target"] for r in rows])


def test_reference_controls_handle_new_reference_but_reject_unseen_preparation():
    x = np.asarray([[t, 20, c] for c in [0, 1] for t in range(20, 70, 5)], dtype=float)
    targets = ratio_curve(np.stack([x[:, 0], x[:, 1]]), 0.1, 48, 5)
    query = np.asarray([[25, 25, 0], [50, 25, 0]], dtype=float)
    for name in METHODS:
        result = predict(name, x, targets, query, (0, 1))
        assert result[0] == pytest.approx(1.0)
        assert 0 < result[1] < 1
        with pytest.raises(ValueError, match="preparation condition"):
            predict(name, x, targets, np.asarray([[50, 25, 2]]), (0, 1))


def test_generated_context_reports_non_reference_error_and_binary_conditions(raw_fixture, tmp_path):
    from copy import deepcopy

    from test_generated_context import CODE

    from agent.rsi import generated_context as api

    raw, methods = raw_fixture
    source.export_tasks(raw, methods, tmp_path / "prepared")
    task, data = read_task(tmp_path / "prepared/tasks/at1g00010/task.json")
    model = api.fit_candidate(data["training"], task["scope"], {"source_code": CODE, "seed": 0})
    metrics, predictions = api.evaluate_with_predictions(model, data["validation"], task["scope"])
    errors = [
        (r["target"] - p["prediction"]) ** 2
        for r, p in zip(data["validation"], predictions)
        if r["values"][0] != r["values"][1]
    ]
    assert metrics["non_reference_metrics"]["rows"] == 9
    assert metrics["non_reference_metrics"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(errors))))
    bad = deepcopy(data["validation"])
    bad[0]["values"][2] = 0.5
    with pytest.raises(ValueError, match="Binary condition"):
        api.validate_rows(bad, task["scope"], labels=True)
