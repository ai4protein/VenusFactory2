"""Workbook/source contracts and complete task ABI using synthetic fixtures."""

import importlib.util
import json
from pathlib import Path

import openpyxl
import pytest

from agent.rsi.datasets.model_tasks import read_task
from agent.rsi.datasets.mutation_landscape_tasks import make_record, write_mutation_task
from agent.rsi.datasets.published_sequence_tasks import audit_catalog, bound

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_mutation_landscapes", ROOT / "scripts/rsi/data_preparation/prepare_mutation_landscape_tasks.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def workbook(path, rows, header=None):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "aminoacid_genotypes"
    sheet.append(header or ["aa_genotype_native", "gene", "replicates_mean_brightness"])
    for row in rows:
        sheet.append(row)
    book.save(path)
    book.close()


def test_workbook_export_preserves_source_row_identity_and_raw_value(tmp_path):
    source, output = tmp_path / "source.xlsx", tmp_path / "export.tsv"
    workbook(source, [["A1C", "amacGFP", 100], ["A1D", "amacGFP", 10]])
    report = prep.export_gfp(source, output)
    assert report["rows"] == 2
    assert output.read_text().splitlines()[1:] == ["2\tA1C\tamacGFP\t100", "3\tA1D\tamacGFP\t10"]
    with pytest.raises(FileExistsError):
        prep.export_gfp(source, output)


def test_duplicate_workbook_genotype_fails_instead_of_reweighting(tmp_path):
    source = tmp_path / "source.xlsx"
    workbook(source, [["A1C", "amacGFP", 100], ["A1C", "amacGFP", 10]])
    with pytest.raises(ValueError, match="Duplicated publisher genotype"):
        prep.export_gfp(source, tmp_path / "export.tsv")


def test_wrong_source_digest_fails_before_creating_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "input").write_text("changed")
    monkeypatch.setattr(prep, "EXPECTED", {"input": "0" * 64})
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Source digest mismatch"):
        prep.prepare(source, output)
    assert not output.exists()


def test_written_tasks_share_phase_by_sequence_and_reject_changed_asset(tmp_path):
    records = []
    for i in range(20):
        for j in range(i + 1, 20):
            sequence = list("A" * 20)
            sequence[i] = sequence[j] = "C"
            records.append(make_record(f"d2:{i}:{j}", "".join(sequence), 2, float(i + j), "fixture"))
            sequence[(j + 1) % 20] = "D"
            distance = sum(letter != "A" for letter in sequence)
            if distance == 3:
                records.append(make_record(f"d3:{i}:{j}", "".join(sequence), 3, float(i - j), "fixture"))
    reference = tmp_path / "source.json"
    reference.write_text(json.dumps(records))
    args = {
        "family": "fixture",
        "paper": "gfp2022",
        "protein": "Synthetic test fixture",
        "target": "synthetic",
        "unit": "synthetic",
        "records": records,
        "references": [bound(reference)],
    }
    entries = [write_mutation_task(tmp_path, identity=name, **args) for name in ("fixture_one", "fixture_two")]
    audit = audit_catalog(entries)
    assert audit["global_exact_sequence_separation"]
    definition, data = read_task(entries[0]["task"]["path"])
    assert len(data["training"]) >= 10 and data["future"]
    asset = Path(definition["data"]["training"]["path"])
    asset.write_text("[]")
    with pytest.raises(ValueError):
        read_task(entries[0]["task"]["path"])
