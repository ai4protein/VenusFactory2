import json
import math

import pytest

from agent.rsi import sabio_kinetics_source as source


def parameter(kind="kcat", value=60.0, raw_unit="min^(-1)", normalized=1.0):
    item = {
        "name": kind,
        "role": "Constant",
        "parameter_type": {"name": kind},
        "start_value": value,
        "unit": {"name": raw_unit, "n_name": source.CANONICAL_UNITS[kind]},
        "n_start_value": normalized,
        "species": {"species_ref_type": "species"},
    }
    if kind != "kcat":
        item["species"]["species_key"] = "1 | L-Malate | Substrate"
    return item


def entry():
    return {
        "id": 1,
        "general": {
            "organism": {"name": "Arabidopsis thaliana", "ncbi_taxonomy_id": 3702},
            "experiment_type": "in_vitro",
            "publication_id": 5,
        },
        "enzyme_description": {
            "ec_number": "1.1.1.40",
            "wildtype": "wildtype",
            "proteins": [{"uniprot_id": "O82191"}, {"uniprot_id": "(O82191)*n"}],
        },
        "publication": {"publication_id": 5, "pubmed_id": "16113210"},
        "reaction": {"id": 142, "species": [{"role": "Substrate", "compound": {"id": 1918, "name": "L-Malate"}}]},
        "experimental_conditions": {"envvar_ph": {"start_value": 7.5}, "envvar_temperature": {}},
        "kineticlaw": {"parameter": [parameter()]},
    }


@pytest.mark.parametrize(
    "kind,value,unit,normalized",
    [
        ("kcat", 60, "min^(-1)", 1),
        ("kcat", 3600, "h^(-1)", 1),
        ("kcat", 2, "s^(-1)", 2),
        ("Km", 250, "\u00b5M", 0.00025),
        ("Km", 2, "mM", 0.002),
        ("Km", 1e6, "nM", 0.001),
        ("kcat/Km", 6, "mM^(-1)*min^(-1)", 100),
    ],
)
def test_physical_unit_conversion(kind, value, unit, normalized):
    result = source.parameter_value(parameter(kind, value, unit, normalized))
    assert result["value"] == pytest.approx(normalized)
    assert result["log10_value"] == pytest.approx(math.log10(normalized))


def test_inverted_time_conversion_is_rejected():
    with pytest.raises(ValueError, match="normalized_value_mismatch"):
        source.parameter_value(parameter(normalized=3600))


@pytest.mark.parametrize(
    "edit,reason",
    [
        ({"start_value": 0}, "nonpositive"),
        ({"start_value": -1}, "nonpositive"),
        ({"start_value": True}, "nonpositive"),
        ({"start_value": float("nan")}, "nonpositive"),
        ({"end_value": 61}, "interval_parameter"),
        ({"role": "Variable"}, "nonconstant"),
        ({"n_start_value": None}, "normalized_value_mismatch"),
    ],
)
def test_invalid_measurement_is_not_silently_coerced(edit, reason):
    item = parameter()
    item.update(edit)
    with pytest.raises(ValueError, match=reason):
        source.parameter_value(item)


def test_unknown_units_and_inconsistent_uncertainty_are_rejected():
    item = parameter(raw_unit="-")
    with pytest.raises(ValueError, match="unsupported_unit"):
        source.parameter_value(item)
    item = parameter()
    item.update(standard_deviation=6, n_standard_deviation=360)
    with pytest.raises(ValueError, match="normalized_sd_mismatch"):
        source.parameter_value(item)
    item["n_standard_deviation"] = 0.1
    assert source.parameter_value(item)["standard_deviation"] == pytest.approx(0.1)


def test_conditions_and_publication_remain_attached_without_imputation():
    raw = entry()
    raw["kineticlaw"]["parameter"].append(parameter("Km", 250, "\u00b5M", 0.00025))
    rows, errors = source.extract_entry(raw, {"path": "/source/page.json", "sha256": "a" * 64})
    assert not errors and len(rows) == 2
    assert rows[1]["associated_substrate"] == {"id": 1918, "name": "L-Malate"}
    assert rows[0]["associated_substrate"] is None
    assert rows[0]["conditions"]["envvar_temperature"] == {}
    assert rows[0]["publication_group"] == "pubmed:16113210"
    assert rows[1]["sources"][0]["parameter_index"] == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ("taxon", "outside_plant_panel"),
        ("mutant", "not_wildtype"),
        ("enzyme", "outside_enzyme_panel"),
        ("publication", "publication_mismatch"),
        ("multiple_proteins", "ambiguous_or_missing_protein"),
        ("protein_expression", "unresolved_protein_expression"),
    ],
)
def test_unreviewed_or_ambiguous_entries_are_excluded(change, reason):
    raw = entry()
    if change == "taxon":
        raw["general"]["organism"]["ncbi_taxonomy_id"] = 1
    if change == "mutant":
        raw["enzyme_description"]["wildtype"] = "mutant"
    if change == "enzyme":
        raw["enzyme_description"]["ec_number"] = "0.0.0.0"
    if change == "publication":
        raw["publication"]["publication_id"] = 6
    if change == "multiple_proteins":
        raw["enzyme_description"]["proteins"].append({"uniprot_id": "P47998"})
    if change == "protein_expression":
        raw["enzyme_description"]["proteins"].append({"uniprot_id": "unresolved"})
    rows, errors = source.extract_entry(raw, {})
    assert not rows and errors[0]["reason"] == reason


def test_substrate_ambiguity_does_not_relabel_a_km():
    raw = entry()
    raw["kineticlaw"]["parameter"] = [parameter("Km", 1, "mM", 0.001)]
    raw["reaction"]["species"].append({"role": "Substrate", "compound": {"id": 55, "name": "L-Malate"}})
    rows, errors = source.extract_entry(raw, {})
    assert not rows and errors[0]["reason"] == "ambiguous_substrate"


def test_exact_repeated_measurement_retains_both_source_coordinates():
    first, second = entry(), entry()
    second["id"] = 2
    rows, errors, duplicates = source.build_observations(
        [(first, {"sha256": "a" * 64}), (second, {"sha256": "b" * 64})]
    )
    assert not errors and duplicates == 1 and len(rows) == 1
    assert [s["entry_id"] for s in rows[0]["sources"]] == [1, 2]
    second["experimental_conditions"]["envvar_ph"]["start_value"] = 8.0
    rows, _, duplicates = source.build_observations([(first, {}), (second, {})])
    assert len(rows) == 2 and duplicates == 0


def snapshot(tmp_path):
    raw_dir = tmp_path / "source"
    raw_dir.mkdir()
    raw = raw_dir / "page_0000.json"
    raw.write_text(
        json.dumps({"response": {"numFound": 1, "start": 0, "docs": [{"EntryID": 1, "Json": json.dumps(entry())}]}})
    )
    manifest = {
        "schema": "sabio_source_pages.v1",
        "complete": True,
        "endpoint": source.ENDPOINT,
        "organisms": list(source.ORGANISMS),
        "ec_numbers": sorted(source.EC_NUMBERS),
        "enzyme_type": "wildtype",
        "query": "panel",
        "total_entries": 1,
        "pages": [
            {"path": str(raw), "sha256": source.sha256(raw), "offset": 0, "entries": 1, "parameters": {"q": "panel"}}
        ],
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, raw, manifest


def test_export_is_source_bound_and_never_counts_rows_as_models(tmp_path):
    path, raw, _ = snapshot(tmp_path)
    output = tmp_path / "observations"
    report = source.export_source(path, source.sha256(path), output)
    assert report["summary"]["retained_observations"] == 1
    assert report["summary"]["new_models"] == 0
    assert report["policy"]["task_admission"] is False
    assert report["inventory"][0]["ready_for_acquisition"] is False
    with pytest.raises(ValueError, match="output_already_exists"):
        source.export_source(path, source.sha256(path), output)
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="source_page_changed"):
        source.read_entries(path, source.sha256(path))


def test_partial_page_and_changed_manifest_are_rejected(tmp_path):
    path, _, manifest = snapshot(tmp_path)
    original_hash = source.sha256(path)
    manifest["total_entries"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_manifest_changed"):
        source.read_entries(path, original_hash)
    with pytest.raises(ValueError, match="source_page_boundary_mismatch"):
        source.read_entries(path, source.sha256(path))


def test_inverse_millisecond_conversion():
    result = source.parameter_value(parameter("kcat", 0.03, "ms^(-1)", 30.0))
    assert result["value"] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="normalized_value_mismatch"):
        source.parameter_value(parameter("kcat", 0.03, "ms^(-1)", 0.00003))
