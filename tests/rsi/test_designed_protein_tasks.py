"""All observations in these admission tests are synthetic software fixtures."""

import copy
import json
from pathlib import Path

import pytest

from agent.rsi.designed_protein_tasks import VERSION, mutation_site, prepare, select_records, site_assignments
from agent.rsi.model_tasks import load_task
from agent.rsi.published_sequence_tasks import bound
from agent.rsi.storage import atomic_json


@pytest.fixture
def source(tmp_path):
    records, parents, assignments = [], {}, {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for name, phase, residue in [
        ("fixture_training", "training", "A"),
        ("fixture_validation", "validation", "C"),
        ("fixture_future", "future", "D"),
    ]:
        parent = residue * 40
        parents[name] = {"name": name, "aa_seq": parent, "DMS_group": "G0", "WT_cluster": "HHH"}
        for site in range(40):
            for aa in [a for a in alphabet if a != residue][:6]:
                identity = f"{name}_{site}_{aa}"
                records.append(
                    {
                        "id": identity,
                        "sequence": parent[:site] + aa + parent[site + 1 :],
                        "group": name,
                        "topology": "HHH",
                        "endpoints": {"dG_ML": float(site % 7)},
                    }
                )
                assignments[identity] = {"group": name, "phase": phase}
    return records, {"parent_scaffolds": parents}, assignments


def test_single_actual_substitution_not_name_heuristic():
    assert mutation_site("A" * 19 + "C", "A" * 20) == 20
    assert mutation_site("A" * 20, "A" * 20) is None
    assert mutation_site("A" * 18 + "CC", "A" * 20) is None
    assert mutation_site("A" * 21, "A" * 20) is None
    with pytest.raises(ValueError):
        mutation_site("A" * 19 + "X", "A" * 20)


def test_old_evaluation_parents_and_used_rows_are_excluded(source):
    records, curation, assignments = source
    chosen, excluded = select_records(records, curation, assignments, {records[0]["id"]}, {records[1]["sequence"]})
    assert set(chosen) == {"fixture_training"}
    assert len(chosen["fixture_training"]) == 238
    assert excluded["original_validation_or_future_parent"] == 480
    assert excluded["previously_used_observation"] == 2


@pytest.mark.parametrize(
    "attack",
    [
        "duplicate_id",
        "duplicate_sequence",
        "missing_assignment",
        "cross_parent_phase",
        "wrong_topology",
        "wrong_assignment_group",
    ],
)
def test_source_inconsistency_is_rejected_before_task_creation(source, attack):
    records, curation, assignments = copy.deepcopy(source)
    if attack == "duplicate_id":
        records[1]["id"] = records[0]["id"]
    elif attack == "duplicate_sequence":
        records[1]["sequence"] = records[0]["sequence"]
    elif attack == "missing_assignment":
        del assignments[records[0]["id"]]
    elif attack == "cross_parent_phase":
        assignments[records[1]["id"]]["phase"] = "future"
    elif attack == "wrong_topology":
        records[0]["topology"] = "hall"
    else:
        assignments[records[0]["id"]]["group"] = "wrong_parent"
    with pytest.raises(ValueError):
        select_records(records, curation, assignments, set(), set())


def test_mutation_site_partitions_are_target_independent(source):
    records, curation, assignments = source
    selected, _ = select_records(records, curation, assignments, set(), set())
    rows = selected["fixture_training"]
    first = site_assignments(rows)
    for row in rows:
        row["endpoints"]["dG_ML"] = 1000000
    assert site_assignments(list(reversed(rows))) == first
    for site in range(1, 41):
        assert len({first[r["id"]]["phase"] for r in rows if r["mutation_site"] == site}) == 1


def source_manifest(tmp_path, source):
    records, curation, assignments = source
    manifest = {"schema": VERSION}
    for name, value in [
        ("observations", records),
        ("curation", curation),
        ("assignments", assignments),
        ("licenses", {"license": "synthetic fixture"}),
    ]:
        path = tmp_path / (name + ".json")
        atomic_json(path, value)
        manifest[name] = bound(path)
    # An older task contains 14 observations from this parent in any old phase.
    previous_rows, assets = records[:14], {}
    for phase, selected in [
        ("training", previous_rows[:10]),
        ("validation", previous_rows[10:12]),
        ("future", previous_rows[12:]),
    ]:
        path = tmp_path / ("prior_" + phase + ".json")
        atomic_json(
            path, [{"id": r["id"], "sequence": r["sequence"], "target": r["endpoints"]["dG_ML"]} for r in selected]
        )
        assets[phase] = bound(path)
    task = {
        "schema": 1,
        "id": "prior_synthetic",
        "stage": "stability",
        "plugin_id": "generated_sequence.v1",
        "scope": {
            "protein": "synthetic fixture",
            "assay": "synthetic assay",
            "target": "dG_ML",
            "unit": "test",
            "minimum_sequence_length": 20,
            "maximum_sequence_length": 2048,
        },
        "paper": {"title": "Synthetic fixture", "doi": "test:fixture"},
        "data": assets,
        "references": [manifest["licenses"]],
        "minimum_skill": 0.0,
    }
    atomic_json(tmp_path / "prior_task.json", task)
    atomic_json(tmp_path / "prior_catalog.json", {"entries": [{"task": bound(tmp_path / "prior_task.json")}]})
    manifest["prior_catalog"] = bound(tmp_path / "prior_catalog.json")
    return manifest


def test_end_to_end_produces_callable_task_contract_and_no_model_claim(source, tmp_path):
    manifest = source_manifest(tmp_path, source)
    output = tmp_path / "prepared"
    report = prepare(manifest, output)
    assert report["eligible_training_parents"] == 1
    assert report["catalog"]["statuses"] == {"admitted": 1}
    assert report["new_autonomous_models"] == 0
    assert report["protected_prior_observation_ids"] == 14
    catalog = json.loads((output / "project_catalog.json").read_text())
    task = json.loads(Path(catalog["entries"][0]["task"]["path"]).read_text())
    data = load_task(task)
    ids = {r["id"] for rows in data.values() for r in rows}
    assert not ids & {r["id"] for r in source[0][:14]}
    assert all("fixture_training" in identity for identity in ids)
    assert catalog["entries"][0]["protein_scope"]["kind"] == "protein"
    with pytest.raises(FileExistsError):
        prepare(manifest, output)


def test_changed_bound_data_rejected_before_output(source, tmp_path):
    manifest = source_manifest(tmp_path, source)
    Path(manifest["observations"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        prepare(manifest, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
