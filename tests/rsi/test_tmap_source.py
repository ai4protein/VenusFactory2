import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from agent.rsi import tmap_source as source
from agent.rsi import tmap_tasks as tasks
from agent.rsi.datasets.model_tasks import load_task
from agent.rsi.engine.storage import sha256_file


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def quantitative_fixture():
    samples = {f"GENE{i}.1": {"gene": f"GENE{i}"} for i in range(4)}
    header = ["Protein IDs", "Number of proteins", *source.FLAGS]
    for sample in samples:
        header.extend(
            f"{field} {sample}"
            for field in ["Ratio H/L", "Ratio H/L count", "Peptides", "Unique peptides", "Ratio H/L normalized"]
        )
    rows = []
    for accession, ratio in [("P00001", "2"), ("P00002", "4")]:
        rows.append([accession, "1", "", "", ""] + [value for _ in samples for value in [ratio, "2", "2", "1", "9999"]])
    return header, rows, samples, {"P00001": {}, "P00002": {}}


def parse_fixture(operation=None):
    header, rows, samples, proteins = quantitative_fixture()
    if operation:
        operation(header, rows)
    return source.parse_quantitative(header, rows, samples, proteins)


def test_unnormalized_ratio_is_used_and_row_columns_are_traceable():
    sample_ids, proteins, values, witness = parse_fixture()
    assert values.tolist() == [[1.0, 2.0]] * 4
    assert proteins == ["P00001", "P00002"]
    assert witness["columns"][sample_ids[0]]["Ratio H/L"] == 6
    assert witness["proteins"]["P00001"]["row"] == 2


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "garbage"])
def test_invalid_measurements_fail_instead_of_becoming_missing(value):
    with pytest.raises(ValueError, match="observation"):
        parse_fixture(lambda header, rows: rows[0].__setitem__(5, value))


@pytest.mark.parametrize("column,value", [(5, ""), (5, "0"), (6, "1"), (7, "1"), (8, "0")])
def test_missing_and_insufficient_evidence_are_not_imputed(column, value):
    _, _, values, _ = parse_fixture(lambda header, rows: rows[0].__setitem__(column, value))
    assert np.isnan(values[0, 0])
    assert np.isfinite(values[1:, 0]).all()


@pytest.mark.parametrize("flag", source.FLAGS)
def test_reverse_contaminants_and_site_only_records_are_excluded(flag):
    _, proteins, _, witness = parse_fixture(lambda header, rows: rows[0].__setitem__(header.index(flag), "+"))
    assert proteins == ["P00002"]
    assert witness["excluded_rows"][0]["reason"] == "reverse_contaminant_or_site_only"


@pytest.mark.parametrize("identity", ["P00001;P00003", "002893", "P00001-2"])
def test_multiple_and_unresolved_accessions_cannot_become_extra_models(identity):
    _, proteins, _, _ = parse_fixture(lambda header, rows: rows[0].__setitem__(0, identity))
    assert proteins == ["P00002"]


def test_duplicate_quantitative_accession_excludes_all_of_its_rows():
    _, proteins, _, witness = parse_fixture(lambda header, rows: rows.append(list(rows[0])))
    assert proteins == ["P00002"]
    assert [item["row"] for item in witness["excluded_rows"]] == [4, 2]


def test_normalized_column_cannot_substitute_for_missing_raw_ratio():
    def change(header, rows):
        header[5] = "unexpected"

    with pytest.raises(ValueError, match="unnormalized"):
        parse_fixture(change)


def test_duplicate_source_column_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_fixture(lambda header, rows: header.__setitem__(5, header[6]))


def test_duplicate_annotations_are_not_counted_twice_or_guessed():
    screen = source.screen_index([(1, "GENE1", "class"), (2, "GENE1", "class")])
    assert screen["GENE1"]["source_rows"] == [2, 3]
    with pytest.raises(ValueError, match="Conflicting"):
        source.screen_index([(1, "GENE1", "class"), (2, "GENE1", "different")])
    accepted, excluded = source.protein_index(
        [(1, "P00001", "A", "locus", "name"), (2, "P00001", "B", "locus", "name")]
    )
    assert accepted == {} and len(excluded) == 2


def test_unresolved_sample_prefix_or_allele_is_preserved_as_exclusion():
    screen = {f"GENE{i}": {"function": "class", "source_rows": [i + 2]} for i in range(4)}
    runs = {f"GENE{i}.1": [f"raw{i}"] for i in range(4)}
    accepted, excluded = source.resolve_samples(set(runs) | {"hGENE1.1", "GENE1d.1"}, screen, runs)
    assert len(accepted) == 4
    assert {item["sample"] for item in excluded} == {"hGENE1.1", "GENE1d.1"}


def test_acquisition_summary_rows_are_not_independent_runs(tmp_path):
    path = tmp_path / "summary.tsv"
    path.write_text("Raw file\tExperiment\nrun1\tGENE1.1\nrun2\tGENE1.1\nGENE1.1\t\nTotal\t\n")
    assert source.acquisition_index(path) == {"GENE1.1": ["run1", "run2"]}
    path.write_text(path.read_text() + "UNKNOWN\t\n")
    with pytest.raises(ValueError, match="Unresolved"):
        source.acquisition_index(path)


def test_repeat_aggregation_and_global_gene_split_ignore_target_values():
    samples = {f"GENE{i}.{rep}": {"gene": f"GENE{i}"} for i in range(80) for rep in (1, 2)}
    order = list(samples)
    features = np.ones((160, 1))
    values = np.tile([1.0, 3.0], 80)
    partition = tasks.partition_genes(samples, 7)
    first, evidence = tasks.aggregate_protein("P00001", samples, order, features, values, partition)
    second, _ = tasks.aggregate_protein("P00002", samples, order, features, values * 100, partition)
    assert sum(map(len, first.values())) == 80
    assert all(row["target"] == 2 for rows in first.values() for row in rows)
    assert all(item["replicates_used"] == 2 for item in evidence.values())
    for phase in tasks.PHASES:
        assert {row["group"] for row in first[phase]} == {row["group"] for row in second[phase]}
    assert {row["group"] for row in first["training"]}.isdisjoint(row["group"] for row in first["future"])
    assert tasks.partition_genes(dict(reversed(list(samples.items()))), 7) == partition


@pytest.fixture
def full_source(tmp_path):
    book = openpyxl.Workbook()
    screen = book.active
    screen.title = "Screen description"
    screen.append(
        ["# (number of strain)", "Strain (name of gene deleted in strain)", "Function (encoded by the deleted gene)"]
    )
    annotations = book.create_sheet("Protein abundance")
    annotations.append(
        ["abundance (copy number per cell)", "Uniprot ID", "Gene Name", "ENSG (systematic name)", "Protein name"]
    )
    scores = book.create_sheet("Turnover robust Z-Scores")
    scores.append(["Strains", "unused normalized score"])
    sample_ids = [f"GENE{i}.{rep}" for i in range(80) for rep in (1, 2)]
    for i in range(80):
        screen.append([i + 1, f"GENE{i}", f"function{i % 8}"])
    for sample in sample_ids:
        scores.append([sample, 999999])
    for accession in ["P00001", "P00002"]:
        annotations.append([999, accession, datetime(2000, 3, 1), "locus", "fixture"])
    workbook = tmp_path / "table.xlsx"
    book.save(workbook)
    raw = tmp_path / "proteinGroups.txt"
    header = ["Protein IDs", "Number of proteins", *source.FLAGS]
    for sample in sample_ids:
        header.extend(
            f"{field} {sample}"
            for field in ["Ratio H/L", "Ratio H/L count", "Peptides", "Unique peptides", "Ratio H/L normalized"]
        )
    header.append("ignored long peptide evidence")
    with raw.open("w", newline="") as writer:
        table = csv.writer(writer, delimiter="\t")
        table.writerow(header)
        for accession in ["P00001", "P00002"]:
            table.writerow(
                [accession, "1", "", "", ""]
                + [value for i in range(160) for value in [2 ** (1 + i % 7 / 4), 2, 2, 1, 99999]]
                + ["x" * 140000]
            )
    summary = tmp_path / "summary.txt"
    summary.write_text(
        "Raw file\tExperiment\n"
        + "".join(f"{sample}a\t{sample}\n{sample}b\t{sample}\n{sample}\t\n" for sample in sample_ids)
        + "Total\t\n"
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"fixture": true}\n')
    return {
        "schema": source.SCHEMA,
        "paper": source.PAPER,
        "protein_groups": binding(raw),
        "table_s1": binding(workbook),
        "ms_summary": binding(summary),
        "references": [binding(receipt)],
        "unavailable_samples": [],
    }


def test_source_to_native_scientist_task_and_tamper_checks(full_source, tmp_path):
    original_limit = csv.field_size_limit()
    prepared = tmp_path / "prepared"
    report = source.prepare(full_source, prepared)
    assert csv.field_size_limit() == original_limit
    assert (report["samples"], report["independent_genes"], report["resolved_proteins"]) == (160, 80, 2)
    assert report["finite_observations"] == 320 and report["external_model_calls"] == 0
    task_root = tmp_path / "tasks"
    result = tasks.prepare(prepared, task_root)
    assert result["scientific_scopes"] == 2 and result["models_published"] == 0
    for entry in result["tasks"]:
        definition = json.loads(Path(entry["task"]["path"]).read_text())
        data = load_task(definition)
        assert sum(len(value) for value in data.values()) == 80
        assert definition["scope"]["unit"] == "log2_H_L_ratio"
        future = Path(definition["data"]["future"]["path"])
        saved = future.read_bytes()
        future.write_bytes(saved + b" ")
        with pytest.raises(ValueError, match="changed"):
            load_task(definition)
        future.write_bytes(saved)
    with pytest.raises(ValueError, match="New absolute"):
        source.prepare(full_source, prepared)
    path = Path(full_source["protein_groups"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        source.prepare(full_source, tmp_path / "changed_source")


def test_missing_archive_sample_requires_explicit_declaration():
    screen = {f"GENE{i}": {"function": "class", "source_rows": [i + 2]} for i in range(5)}
    runs = {f"GENE{i}.1": [f"raw{i}"] for i in range(4)}
    published = set(runs) | {"GENE4.1"}
    with pytest.raises(ValueError, match="absent"):
        source.resolve_samples(published, screen, runs)
    accepted, excluded = source.resolve_samples(published, screen, runs, ["GENE4.1"])
    assert len(accepted) == 4
    assert excluded == [{"sample": "GENE4.1", "reason": "explicitly_unavailable_in_bound_archive"}]


def test_unavailable_sample_cannot_hide_an_existing_archive_measurement():
    screen = {f"GENE{i}": {"function": "class", "source_rows": [i + 2]} for i in range(4)}
    runs = {f"GENE{i}.1": [f"raw{i}"] for i in range(4)}
    with pytest.raises(ValueError, match="exists"):
        source.resolve_samples(set(runs), screen, runs, ["GENE0.1"])


def test_source_native_nan_is_missing_and_recorded_without_imputation():
    _, _, values, witness = parse_fixture(lambda header, rows: rows[0].__setitem__(5, "NaN"))
    assert np.isnan(values[0, 0])
    assert witness["counts"]["source_nan_ratio"] == 1
    assert source.measurement("NaN", count=True) is None


def test_excel_date_metadata_is_preserved_without_guessing_a_gene():
    proteins, excluded = source.protein_index([(1, "P00001", datetime(2000, 3, 1), "YAA001W", "name")])
    assert not excluded
    assert proteins["P00001"]["gene"] == {
        "unresolved_original_type": "datetime",
        "original_value": "2000-03-01T00:00:00",
    }
    assert proteins["P00001"]["accession"] == "P00001"
