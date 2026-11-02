import copy
import json
import math
import zipfile

import openpyxl
import pytest

from agent.rsi import yeast_population as population
from agent.rsi.engine.storage import sha256_file


def samples():
    records = []
    populations = [
        "1. Wine/European",
        "3. Brazilian bioethanol",
        "5. French dairy",
        "6. African beer",
        "13. African palm wine",
        "25. Sake",
        "26. Asian fermentation",
    ]
    for index, name in enumerate(populations):
        for j in range(12):
            records.append(
                {
                    "Standardized_name": "A" + chr(65 + index) + chr(65 + j),
                    "Eco_Origin": "Wine",
                    "Clades": name,
                    "Ploidy": 2,
                    "OD_RNA_seq": "0.5",
                    "OD_proteomic": "1.0",
                    "in_Proteomic": "Yes",
                    "in_Transcriptomic": "Yes",
                }
            )
    return records


def test_stable_partition_keeps_wine_subclades_together():
    rows = samples()
    rows[0]["Clades"] = "1. Wine/European (subclade 4)"
    forward = population.population_plan(rows)
    assert forward == population.population_plan(list(reversed(rows)))
    assert forward["rows"] == {"training": 60, "validation": 12, "future": 12}
    assert len({r["phase"] for r in forward["samples"] if r["population"] == "1. Wine/European"}) == 1
    assert forward["new_trained_models"] == 0


def test_origin_and_ambiguous_population_filters():
    rows = samples()
    extras = [
        ("Human clinical", "25. Sake"),
        ("Unknown", "25. Sake"),
        ("Wine", "M1. Mosaic region 1"),
        ("Wine", "8. Mixed origin"),
        ("Wine", None),
    ]
    for i, (origin, clade) in enumerate(extras):
        rows.append(dict(rows[0], Standardized_name="ZZ" + chr(65 + i), Eco_Origin=origin, Clades=clade))
    plan = population.population_plan(rows)
    assert len(plan["samples"]) == 84
    assert plan["excluded"]["outside_food_fermentation_origins"] == 2
    assert plan["excluded"]["mixed_mosaic_or_unassigned_population"] == 3


def test_target_fields_cannot_influence_metadata_split():
    original = samples()
    modified = copy.deepcopy(original)
    for index, row in enumerate(modified):
        row["protein_target"] = index * 1000000
        row["RNA_target"] = -index
    assert population.population_plan(original) == population.population_plan(modified)


def test_duplicates_and_insufficient_populations_fail():
    with pytest.raises(ValueError, match="Repeated"):
        population.population_plan(samples() + samples()[:1])
    with pytest.raises(ValueError, match="six eligible"):
        population.population_plan(samples()[:60])


@pytest.mark.parametrize("value", [None, "", "bad", math.nan, math.inf, -1, 11, True])
def test_covariates_reject_nonfinite_and_invalid(value):
    with pytest.raises(ValueError):
        population.numeric_metadata(value, 0, 10)


def test_source_binding_and_refuse_overwrite(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = samples()
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    headers = list(records[0])
    sheet.append(headers)
    for row in records:
        sheet.append([row[k] for k in headers])
    workbook.save(raw / population.SAMPLE_TABLE)
    archive = raw / "supplementary.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(raw / population.SAMPLE_TABLE, population.SAMPLE_TABLE)
    monkeypatch.setattr(population, "SUPPLEMENT_SHA256", sha256_file(archive))
    output = tmp_path / "frozen"
    report = population.freeze_source(raw, output)
    assert report["populations"] == 7
    assert report["external_model_calls"] == 0
    saved = json.loads((output / "population_plan.json").read_text())
    assert saved["metadata_source"]["sha256"] == sha256_file(raw / population.SAMPLE_TABLE)
    with pytest.raises(FileExistsError):
        population.freeze_source(raw, output)
    with (raw / population.SAMPLE_TABLE).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="differs"):
        population.freeze_source(raw, tmp_path / "other")


def test_documented_sace_alias_is_preserved_without_suffix_merging():
    assert population.canonical_sample("SACE.YBY") == "SACE_YBY"
    assert population.canonical_sample("SACE_YBY") == "SACE_YBY"
    assert population.canonical_sample("YBY") == "YBY"
    assert population.canonical_sample("SACE/YBY") is None
    rows = samples()
    rows[0]["Standardized_name"] = "SACE_YBY"
    plan = population.population_plan(rows)
    assert any(r["id"] == "SACE_YBY" for r in plan["samples"])


def test_alias_collision_is_rejected():
    rows = samples()
    rows[0]["Standardized_name"] = "SACE_YBY"
    rows.append(dict(rows[0], Standardized_name="SACE.YBY"))
    with pytest.raises(ValueError, match="Repeated"):
        population.population_plan(rows)


def test_report_requires_raw_signal_and_available_quality_fields():
    columns = [
        "File.Name",
        "Run",
        "Genes",
        "Proteotypic",
        "Precursor.Id",
        "Stripped.Sequence",
        "Precursor.Quantity",
        *population.target_protocol()["quality_fields"],
    ]
    assert "Protein.Q.Value" in population.validate_report_header(columns)
    assert "Global.Q.Value" in population.validate_report_header(columns)
    assert "Global.Protein.Q.Value" not in columns
    missing_raw = [c for c in columns if c != "Precursor.Quantity"] + ["Precursor.Normalised"]
    with pytest.raises(ValueError, match="Raw report"):
        population.validate_report_header(missing_raw)
    with pytest.raises(ValueError, match="Raw report"):
        population.validate_report_header(columns + [columns[0]])
