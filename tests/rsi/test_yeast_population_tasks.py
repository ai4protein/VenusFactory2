import copy
import json

import pytest

from agent.rsi import yeast_population_tasks as source
from agent.rsi.context_baselines import run_controls
from agent.rsi.model_tasks import read_task
from agent.rsi.research_project_catalog import build_catalog
from agent.rsi.yeast_population import DOI, target_protocol


@pytest.fixture
def quantified(tmp_path):
    root = tmp_path / "quantified"
    root.mkdir()
    samples = []
    for group in range(6):
        phase = "training" if group < 4 else "validation" if group == 4 else "future"
        for i in range(10):
            samples.append(
                {
                    "id": f"S{group}_{i}",
                    "population": f"population_{group}",
                    "phase": phase,
                    "od_rna": 0.5,
                    "od_protein": 1.0,
                    "ploidy": 2.0,
                }
            )
    plan_ref = source.save(root / "plan.json", {"study_doi": DOI, "samples": samples})
    protocol_ref = source.save(root / "protocol.json", target_protocol())
    source.save(
        root / "source_evidence.json",
        {
            "schema": "yeast_food_population_quantification.v1",
            "study_doi": DOI,
            "plan": plan_ref,
            "protocol": protocol_ref,
        },
    )
    genes = {}
    for gene in ["YAL001C", "YAL002W", "YAL003W"]:
        rows = samples[:30] if gene == "YAL002W" else samples[:55] if gene == "YAL003W" else samples
        genes[gene] = source.save(
            root / (gene + ".json"),
            {
                "gene": gene,
                "observations": {s["id"]: {"target": i / 10, "runs": []} for i, s in enumerate(rows)},
                "rna": {s["id"]: i / 10 + 1 for i, s in enumerate(rows)},
            },
        )
    source.save(root / "gene_index.json", genes)
    source.save(root / "status.json", {"status": "completed"})
    return root


def test_export_preserves_population_split_and_reports_missing_coverage(quantified, tmp_path):
    out = tmp_path / "tasks"
    result = source.export_tasks(quantified, out)
    assert result["source_genes"] == 3 and result["exported_tasks"] == 2
    assert result["catalog_statuses"] == {"admitted": 1, "deferred": 1}
    assert result["excluded"][0]["gene"] == "YAL002W"
    task, data = read_task(out / "tasks/yeast_expression_yal001c/task.json")
    assert {p: len(v) for p, v in data.items()} == {"training": 40, "validation": 10, "future": 10}
    assert task["scope"]["grouping"]["level"] == "genetic_population"
    assert data["future"][0]["group"].endswith(":population_5")
    controls = run_controls(task)
    assert controls["fit_partition"] == "training" and controls["future_scored"] is False
    assert len(controls["outcomes"]) == 4
    assert all(c["status"] == "completed" for c in controls["outcomes"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert build_catalog(manifest)["summary"]["new_trained_models"] == 0
    evidence = json.loads((out / "tasks/yeast_expression_yal001c/quantification_evidence.json").read_text())
    assert "observations" not in evidence and "rna" not in evidence


def test_population_cannot_be_relabelled_as_replicates(quantified, tmp_path):
    out = tmp_path / "tasks"
    source.export_tasks(quantified, out)
    manifest = json.loads((out / "manifest.json").read_text())
    modified = copy.deepcopy(manifest)
    modified["candidates"][0]["group_basis"] = "replicate_group"
    catalog = build_catalog(modified)
    assert catalog["entries"][0]["status"] == "invalid"
    assert "grouping basis" in catalog["entries"][0]["reasons"][0].lower()


def test_source_mutation_and_unfinished_quantification_fail(quantified, tmp_path):
    source.save(quantified / "status.json", {"status": "running"})
    with pytest.raises(ValueError, match="Completed quantification"):
        source.export_tasks(quantified, tmp_path / "not_finished")
    source.save(quantified / "status.json", {"status": "completed"})
    path = quantified / "YAL001C.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        source.export_tasks(quantified, tmp_path / "changed")
