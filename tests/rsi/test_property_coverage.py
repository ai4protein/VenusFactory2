"""Property diversity must survive skewed source inventories and aliases."""

import json
from pathlib import Path

import pytest
import test_shared_acquisition_plan as fixtures

from agent.rsi.campaigns.property_coverage import select_by_property, study_identity, validate_policy
from agent.rsi.campaigns.shared_acquisition_plan import plan_acquisition

manifest = fixtures.manifest


def policy():
    return {
        "schema": "property_coverage_policy.v1",
        "max_per_protein": 2,
        "families": {
            "abundance": {"target": 0, "max_per_study": 0, "min_studies": 0},
            "activity": {"target": 4, "max_per_study": 2, "min_studies": 2},
            "stability": {"target": 4, "max_per_study": 2, "min_studies": 2},
        },
        "rules": [
            {"stage": "expression", "targets": ["protein_abundance"], "family": "abundance"},
            {"stage": "activity", "family": "activity"},
            {"stage": "stability", "family": "stability"},
        ],
    }


def record(name, stage, study, protein=None, target=None):
    return {
        "id": name,
        "stage": stage,
        "target": target or stage,
        "study_doi": study,
        "protein": protein or name,
        "scenario": {"assay": "measured"},
        "interface": "generated_sequence.v1",
    }


def test_large_abundance_pool_cannot_fill_missing_properties():
    rows = [record(f"a{i}", "expression", "study_a", target="protein_abundance") for i in range(2000)]
    rows += [record("e1", "activity", "study_e"), record("e2", "activity", "study_e")]
    result = select_by_property(rows, policy())
    assert result["selected_ids"] == ["e1", "e2"]
    assert result["families"]["stability"]["shortfall"] == 4
    assert result["families"]["activity"]["shortfall"] == 2
    assert result["skipped"]["deferred_property_family"] == 2000
    assert not result["coverage_satisfied"]


def test_project_and_doi_aliases_cannot_bypass_source_cap():
    rows = [
        record(str(i), "activity", doi)
        for i, doi in enumerate(
            ["10.1234/EXAMPLE", "https://doi.org/10.1234/example", "doi:10.1234/example", "10.1234/other"]
        )
    ]
    result = select_by_property(rows, policy())
    assert result["families"]["activity"]["selected"] == 3
    assert len(result["families"]["activity"]["studies"]) == 2


def test_protein_cap_applies_across_families():
    rows = [
        record(f"{s}{i}", s, f"study_{i}", protein="Same Protein") for s in ["activity", "stability"] for i in range(4)
    ]
    result = select_by_property(rows, policy())
    assert result["selected_tasks"] == 2 and result["distinct_protein_labels"] == 1


def test_balanced_complete_selection_and_order_invariance():
    rows = [
        record(f"{s}{paper}{i}", s, f"study_{paper}")
        for s in ["activity", "stability"]
        for paper in range(2)
        for i in range(8)
    ]
    first = select_by_property(rows, policy())
    second = select_by_property(list(reversed(rows)), policy())
    assert first == second
    assert first["coverage_satisfied"] and first["selected_tasks"] == 8
    assert all(first["families"][s]["selected"] == 4 for s in ["activity", "stability"])


def test_renamed_unknown_endpoint_is_visible_and_not_guessed():
    result = select_by_property([record("a", "expression", "paper", target="unknown label")], policy())
    assert result["selected_tasks"] == 0
    assert result["skipped"] == {"unclassified_property": 1}


def test_scores_and_future_values_cannot_change_selection():
    rows = [record(str(i), "activity", f"study_{i % 2}") for i in range(12)]
    expected = select_by_property(rows, policy())
    for i, row in enumerate(rows):
        row.update(future_rmse=-i, model_accepted=i % 2 == 0)
    assert select_by_property(rows, policy()) == expected


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["families"]["activity"].update(target=True),
        lambda p: p["families"]["activity"].update(max_per_study=0),
        lambda p: p["families"]["abundance"].update(max_per_study=1),
        lambda p: p["rules"].append({"stage": "activity", "targets": ["a"], "family": "stability"}),
        lambda p: p["rules"].append({"stage": "other", "family": "unknown"}),
    ],
)
def test_invalid_policy_rejected(change):
    p = policy()
    change(p)
    with pytest.raises(ValueError):
        validate_policy(p)


def classified_manifest(manifest, tmp_path):
    m = fixtures.make_manifest(manifest, tmp_path, counts=(8, 2, 2))
    for c in m["candidates"]:
        task = json.loads(Path(c["task"]["path"]).read_text())
        n = int(c["project_id"].split("_")[-1])
        stage, target = [("expression", "protein_abundance"), ("activity", "Activity"), ("stability", "dG_ML")][n]
        task["stage"], task["scope"]["target"] = stage, target
        c["property"] = target
        c["task"] = fixtures.fixtures.save(Path(c["task"]["path"]), task)
    return m


def test_source_bound_planner_leaves_shortfall_instead_of_backfilling(manifest, tmp_path):
    m = classified_manifest(manifest, tmp_path)
    a, b = fixtures.prepared(m, tmp_path)
    p = fixtures.fixtures.save(tmp_path / "policy.json", policy())
    result = plan_acquisition(a, b, projects=[x["id"] for x in m["projects"]], target_count=8, coverage_policy=p)
    assert result["schema"] == "shared_acquisition_plan.v2"
    assert result["status"] == "prepared_partial_allocation"
    assert result["selected_count"] == 4 and result["target_count"] == 8
    assert {t["stage"] for t in fixtures.tasks_of(result)} == {"activity", "stability"}
    assert result["available_scopes"] == 12
    assert sum(v["available_scopes"] for v in result["capacity"]) == 12
    assert {x["property_family"] for x in result["source_bindings"]} == {"activity", "stability"}


def test_source_bound_policy_cannot_be_changed_after_binding(manifest, tmp_path):
    m = classified_manifest(manifest, tmp_path)
    a, b = fixtures.prepared(m, tmp_path)
    p = fixtures.fixtures.save(tmp_path / "policy.json", policy())
    Path(p["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        plan_acquisition(a, b, projects=[x["id"] for x in m["projects"]], target_count=8, coverage_policy=p)


def test_repository_policy_declares_six_non_abundance_families():
    root = Path(__file__).resolve().parents[2]
    p = json.loads((root / "configs/rsi/property_coverage_v1.json").read_text())
    assert validate_policy(p) == 48
    assert sum(v["target"] > 0 for v in p["families"].values()) == 6
    assert p["families"]["protein_abundance"]["target"] == 0


@pytest.mark.parametrize(
    "prefix", ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"]
)
def test_canonical_study_identity(prefix):
    assert study_identity(prefix + "10.1234/Example") == "10.1234/example"
