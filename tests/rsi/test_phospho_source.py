import csv
import json
from pathlib import Path

import numpy as np
import pytest

from agent.rsi import phospho_source as source
from agent.rsi import phospho_tasks as tasks
from agent.rsi.model_tasks import load_task
from agent.rsi.protein_model import digest
from agent.rsi.storage import sha256_file


def annotations(n=80):
    conditions = [
        (i + 2, {"cond": f"C{i}", "description": f"condition {i}", "type": f"class{i % 8}", "time": "5min"})
        for i in range(n)
    ]
    runs = [
        (
            i * 3 + rep + 1,
            {"sample": f"C{i}.{i + 1}_{rep}", "run": f"x{10000 + i * 3 + rep}", "batch": "2", "block": "2A"},
        )
        for i in range(n)
        for rep in (1, 2, 3)
    ]
    return conditions, runs


def tiny_quant():
    _, runs, samples, _ = source.metadata(*annotations(2))
    sample = next(iter(samples))
    row = {"sample": sample, "run": samples[sample]["run"], "reference": "YAL001W", "var": "YAL001W_S_2", "value": "12"}
    return row, runs, samples


def test_condition_code_is_used_instead_of_disagreeing_numeric_suffix():
    condition_rows, run_rows = annotations(2)
    run_rows[0][1]["sample"] = "C0.999_1"
    _, _, samples, excluded = source.metadata(condition_rows, run_rows)
    assert samples["C0.999_1"]["condition"] == "C0" and not excluded


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"sample": "block.pool_2A"}, "not_a_resolved_biological_sample"),
        ({"sample": "DP.36_1"}, "no_exact_condition_code"),
        ({"batch": "1"}, "outside_author_retained_batches"),
    ],
)
def test_qc_and_unsupported_samples_are_excluded(change, reason):
    condition_rows, run_rows = annotations(2)
    run_rows[0][1].update(change)
    _, _, samples, excluded = source.metadata(condition_rows, run_rows)
    assert len(samples) == 5 and excluded[0]["reason"] == reason


@pytest.mark.parametrize("time", ["120min", "static"])
def test_different_durations_and_backgrounds_are_not_mixed(time):
    condition_rows, run_rows = annotations(2)
    condition_rows[0][1]["time"] = time
    _, _, samples, excluded = source.metadata(condition_rows, run_rows)
    assert len(samples) == len(excluded) == 3
    assert all(item["reason"] == "different_duration_or_background" for item in excluded)


def test_annotations_and_biological_replicates_cannot_be_duplicated():
    conditions, runs = annotations(2)
    conditions.append((90, dict(conditions[0][1])))
    accepted, _, _, _ = source.metadata(conditions, runs)
    assert accepted["C0"]["source_rows"] == [2, 90]
    conditions[-1][1]["type"] = "different"
    with pytest.raises(ValueError, match="Conflicting"):
        source.metadata(conditions, runs)
    conditions.pop()
    runs.append((100, {**runs[0][1], "run": "x999999"}))
    with pytest.raises(ValueError, match="Repeated biological"):
        source.metadata(conditions, runs)


def test_duplicate_or_mismatched_quantification_is_rejected():
    row, runs, samples = tiny_quant()
    with pytest.raises(ValueError, match="Duplicate sample-site"):
        source.parse_quantitative([(2, row), (3, row)], runs, samples)
    with pytest.raises(ValueError, match="metadata disagree"):
        source.parse_quantitative([(2, {**row, "run": "x99999"})], runs, samples)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "bad"])
def test_malformed_log_intensities_fail(value):
    row, runs, samples = tiny_quant()
    with pytest.raises(ValueError, match="log intensity"):
        source.parse_quantitative([(2, {**row, "value": value})], runs, samples)


@pytest.mark.parametrize("value", ["", "NA", "NaN"])
def test_missing_values_stay_missing_and_retain_witness(value):
    row, runs, samples = tiny_quant()
    _, _, values, witnesses, report = source.parse_quantitative([(2, {**row, "value": value})], runs, samples)
    assert np.isnan(values).all() and witnesses.max() == 2
    assert report["counts"]["explicit_missing_measurements"] == 1


def test_ambiguous_sites_are_excluded_and_negative_log_values_are_valid():
    row, runs, samples = tiny_quant()
    rows = [
        (2, {**row, "value": "-2"}),
        (3, {**row, "var": "YAL001W;YAL002W_S_2"}),
        (4, {**row, "reference": "YAL002W"}),
    ]
    _, sites, values, _, report = source.parse_quantitative(rows, runs, samples)
    assert sites == ["YAL001W_S_2"] and np.nanmin(values) == -2
    assert report["counts"]["unresolved_site_rows"] == 2


def test_centering_is_sample_local_and_removes_additive_run_offsets():
    values = np.arange(4 * 110, dtype=float).reshape(4, 110)
    values[1, :4] = np.nan
    original, _, _, admitted = source.center_runs(values)
    shifted, _, _, _ = source.center_runs(values + np.array([2, -5, 100, 3])[:, None])
    np.testing.assert_allclose(original, shifted, equal_nan=True)
    assert admitted.all()
    changed = values.copy()
    changed[3, :10] += 999
    result, _, _, _ = source.center_runs(changed)
    np.testing.assert_array_equal(result[:3], original[:3])
    changed[2, :11] = np.nan
    result, _, coverage, admitted = source.center_runs(changed)
    assert coverage[2] == 99 and not admitted[2] and np.isnan(result[2]).all()


def test_partitions_ignore_site_values_and_all_replicates_stay_together():
    _, _, samples, _ = source.metadata(*annotations())
    order = list(samples)
    partitions = tasks.partition_conditions(samples, 17)
    assert partitions == tasks.partition_conditions(dict(reversed(list(samples.items()))), 17)
    features, values = np.ones((len(order), 1)), np.tile([1, 2, 3], 80).astype(float)
    witnesses = np.arange(len(order)) + 2
    one, evidence = tasks.aggregate_site("YAL001W_S_2", samples, order, features, values, witnesses, partitions)
    two, _ = tasks.aggregate_site("YAL001W_S_3", samples, order, features, values * 99, witnesses, partitions)
    assert sum(map(len, one.values())) == 80 and all(item["replicates_used"] == 3 for item in evidence.values())
    for phase in tasks.PHASES:
        assert {row["group"] for row in one[phase]} == {row["group"] for row in two[phase]}
        assert all(row["target"] == 2 for row in one[phase])
    values[0] = np.nan
    incomplete, _ = tasks.aggregate_site("YAL001W_S_2", samples, order, features, values, witnesses, partitions)
    assert sum(map(len, incomplete.values())) == 79
    samples[order[1]]["replicate"] = samples[order[0]]["replicate"]
    with pytest.raises(ValueError, match="Repeated biological"):
        tasks.aggregate_site("YAL001W_S_2", samples, order, features, np.ones(len(order)), witnesses, partitions)


@pytest.fixture
def full_source(tmp_path, monkeypatch):
    condition_rows, run_rows = annotations()
    tables = {
        "meta_conditions.csv": (
            ["id", "description", "type", "description 2", "cond", "time"],
            [{"id": f"{r['cond']}.999", "description 2": "fixture", **r} for _, r in condition_rows],
        ),
        "meta_runs_samples.csv": (["sample", "run", "batch", "block"], [r for _, r in run_rows]),
    }
    sites = [f"YAL{protein:03d}W_S_{position}" for protein in (1, 2) for position in range(1, 56)]
    quantitative = []
    for i, (_, run) in enumerate(run_rows):
        for j, site in enumerate(sites):
            quantitative.append(
                {
                    "sample": run["sample"],
                    "run": run["run"],
                    "reference": site.split("_")[0],
                    "var": site,
                    "value": 10 + j / 50 + np.sin(i // 3 + j) + (i % 3) / 10,
                }
            )
    tables["data_psites.csv"] = (["sample", "run", "reference", "var", "value"], quantitative)
    for name, (header, rows) in tables.items():
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    pins = {name: sha256_file(tmp_path / name) for name in tables}
    monkeypatch.setattr(source, "SOURCE_HASHES", pins)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"fixture":true}\n')
    return {
        "schema": source.SCHEMA,
        "paper": source.PAPER,
        "record_doi": source.RECORD,
        "files": {name: {"path": str(tmp_path / name), "sha256": value} for name, value in pins.items()},
        "references": [tasks.binding(receipt)],
    }


def test_source_to_native_tasks_counts_sites_and_proteins_separately(full_source, tmp_path):
    prepared = tmp_path / "prepared"
    report = source.prepare(full_source, prepared)
    assert report["samples"] == 240 and report["finite_observations"] == 240 * 110
    assert report["distinct_proteins"] == 2 and report["site_scopes_before_task_admission"] == 110
    result = tasks.prepare(prepared, tmp_path / "tasks")
    assert result["scientific_scopes"] == 110 and result["distinct_proteins"] == 2 and result["models_published"] == 0
    scopes = set()
    for entry in result["tasks"]:
        definition = json.loads(Path(entry["task"]["path"]).read_text())
        data = load_task(definition)
        assert sum(len(rows) for rows in data.values()) == 80
        scopes.add(digest(definition["scope"]))
        assert definition["scope"]["context"]["site"] == entry["site"]
    assert len(scopes) == 110
    sample_task = json.loads(Path(result["tasks"][0]["task"]["path"]).read_text())
    future = Path(sample_task["data"]["future"]["path"])
    future.write_bytes(future.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        load_task(sample_task)


def test_pinned_sources_and_prepared_assets_reject_tampering(full_source, tmp_path):
    prepared = tmp_path / "prepared"
    source.prepare(full_source, prepared)
    with pytest.raises(ValueError, match="New absolute"):
        source.prepare(full_source, prepared)
    asset = prepared / "samples.json"
    asset.write_bytes(asset.read_bytes() + b" ")
    with pytest.raises(ValueError, match="asset changed"):
        tasks.prepare(prepared, tmp_path / "tampered_tasks")
    raw = Path(full_source["files"]["data_psites.csv"]["path"])
    raw.write_bytes(raw.read_bytes() + b" ")
    full_source["files"]["data_psites.csv"]["sha256"] = sha256_file(raw)
    with pytest.raises(ValueError, match="pinned nonimputed"):
        source.prepare(full_source, tmp_path / "tampered_source")


def test_csv_headers_and_record_width_are_strict(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,a\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        list(source.read_rows(path, ["a", "b"]))
    path.write_text("a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="width"):
        list(source.read_rows(path, ["a", "b"]))


def test_orf_suffixes_have_valid_distinct_native_task_names():
    assert tasks.task_identifier("YAL001W-A_S_2") == "yeast_phospho_yal001w_a_s_2"
    assert tasks.task_identifier("YAL001W_S_2") != tasks.task_identifier("YAL001W-A_S_2")
    with pytest.raises(ValueError, match="site identity"):
        tasks.task_identifier("YAL001W;YAL002W_S_2")
