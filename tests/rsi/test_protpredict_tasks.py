"""Source-bound task materialization on synthetic matrices, with no model fits."""

import hashlib
import json

import numpy as np
import pytest

from agent.rsi import model_tasks
from agent.rsi import protpredict_tasks as api
from agent.rsi.protpredict_source import FEATURES, PAPER_DOI
from agent.rsi.protpredict_source import SCHEMA as SOURCE_SCHEMA
from agent.rsi.storage import atomic_json, sha256_file


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    protein_raw, metabolite_raw, license_path = [
        root / n for n in ["raw_proteomics.zip", "raw_metabolites.xls", "LICENSE"]
    ]
    for p in (protein_raw, metabolite_raw):
        p.write_text("Synthetic raw-source fixture; not an acquired scientific dataset")
    license_path.write_text("MIT License - synthetic fixture")
    original = {
        "schema": SOURCE_SCHEMA,
        "repository_commit": "a" * 40,
        "paper": {
            "title": "Interpreting protein abundance in Saccharomyces cerevisiae through relational learning",
            "doi": PAPER_DOI,
        },
        "proteomics": binding(protein_raw),
        "metabolites": binding(metabolite_raw),
        "references": [binding(license_path)],
    }
    genes, proteins = [f"synthetic_deletion_{i}" for i in range(100)], ["synthetic_P1", "synthetic_P2"]
    x = np.random.default_rng(12).uniform(0.1, 5, (100, 19))
    y = np.column_stack([x.sum(axis=1), x[:, 0] * 3 + 5])
    y[0, 0] = np.nan
    for name, array in [("features.npy", x), ("targets.npy", y)]:
        np.save(root / name, array, allow_pickle=False)
    for name, value in [
        ("genotypes.json", genes),
        ("proteins.json", proteins),
        ("covariates.json", [{"name": n, "unit": "mM"} for n in FEATURES]),
        (
            "row_provenance.json",
            {g: {"protein_csv_rows": [i + 2], "metabolite_xls_row": i + 2} for i, g in enumerate(genes)},
        ),
        ("source.json", original),
    ]:
        atomic_json(root / name, value)
    manifest = {
        "schema": SOURCE_SCHEMA,
        "status": "source_reconstructed",
        "source": original,
        "files_sha256": {n: sha256_file(root / n) for n in api.FILES},
        "acquisition_admitted": False,
        "target": "unchanged synthetic abundance",
    }
    atomic_json(root / "manifest.json", manifest)
    order = sorted(genes, key=lambda g: hashlib.sha256((api.SEED + "\0" + g).encode()).hexdigest())
    plan = {
        "schema": "protpredict_partition_plan.v1",
        "seed": api.SEED,
        "matrix_manifest_sha256": sha256_file(root / "manifest.json"),
        "partitions": {"training": order[:60], "validation": order[60:80], "future": order[80:]},
        "per_task_caps": api.CAPS,
        "minimum_rows": api.MINIMUM,
        "declared_proteins": 2,
        "proteins_with_sufficient_counts": 2,
        "acquisition_admitted": False,
        "models_trained": 0,
        "external_model_calls": 0,
    }
    atomic_json(root / "plan.json", plan)
    declaration = {
        "schema": api.SCHEMA,
        "matrix": binding(root / "manifest.json"),
        "partition_plan": binding(root / "plan.json"),
        "license": binding(license_path),
    }
    return root, declaration, x, y, genes


def test_exact_measurements_common_genotype_holdout_and_catalog(source, tmp_path):
    root, declaration, x, y, genes = source
    output = tmp_path / "tasks"
    report = api.prepare(declaration, output)
    assert report["catalog_statuses"] == {"admitted": 2} and report["exported_tasks"] == 2
    assert report["external_model_calls"] == report["new_trained_models"] == report["new_published_models"] == 0
    manifest, plan = read(output / "manifest.json"), read(root / "plan.json")
    all_phases = {}
    for column, candidate in enumerate(manifest["candidates"]):
        definition, data = model_tasks.read_task(model_tasks.bound_file(candidate["task"]))
        assert definition["plugin_id"] == "generated_context.v2"
        assert [c["name"] for c in definition["scope"]["covariates"]] == list(FEATURES)
        for phase, rows in data.items():
            expected = [genes.index(g) for g in plan["partitions"][phase] if np.isfinite(y[genes.index(g), column])]
            assert len(rows) == len(expected)
            for row, index in zip(rows, expected, strict=True):
                np.testing.assert_array_equal(row["values"], x[index])
                assert row["target"] == y[index, column]
                assert row["group"] == PAPER_DOI + ":" + genes[index]
                assert all_phases.setdefault(row["group"], phase) == phase
    with pytest.raises(ValueError, match="New absolute"):
        api.prepare(declaration, output)


@pytest.mark.parametrize("attack", ["partitions", "caps", "seed", "minimum", "feasibility", "matrix_binding"])
def test_rehashed_plan_cannot_change_predeclared_protocol(source, tmp_path, attack):
    root, declaration, *_ = source
    plan = read(root / "plan.json")
    if attack == "partitions":
        a, b = plan["partitions"]["training"], plan["partitions"]["future"]
        a[0], b[0] = b[0], a[0]
    elif attack == "caps":
        plan["per_task_caps"]["training"] = 1000
    elif attack == "seed":
        plan["seed"] = "selected_after_evaluation"
    elif attack == "minimum":
        plan["minimum_rows"]["training"] = 1
    elif attack == "feasibility":
        plan["proteins_with_sufficient_counts"] = 99
    else:
        plan["matrix_manifest_sha256"] = "0" * 64
    atomic_json(root / "plan.json", plan)
    declaration["partition_plan"] = binding(root / "plan.json")
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["features.npy", "targets.npy", "row_provenance.json", "raw_proteomics.zip"])
def test_source_asset_drift_is_rejected_before_output(source, tmp_path, name):
    root, declaration, *_ = source
    with (root / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("attack", ["nineteenth_feature", "target_inf", "provenance"])
def test_invalid_reconstructed_values_or_provenance_rejected_even_with_fresh_hashes(source, tmp_path, attack):
    root, declaration, x, y, genes = source
    if attack == "nineteenth_feature":
        x[0, -1] = float("nan")
        np.save(root / "features.npy", x, allow_pickle=False)
    elif attack == "target_inf":
        y[0, 0] = float("inf")
        np.save(root / "targets.npy", y, allow_pickle=False)
    else:
        provenance = read(root / "row_provenance.json")
        del provenance[genes[0]]
        atomic_json(root / "row_provenance.json", provenance)
    matrix = read(root / "manifest.json")
    matrix["files_sha256"] = {n: sha256_file(root / n) for n in api.FILES}
    atomic_json(root / "manifest.json", matrix)
    declaration["matrix"] = binding(root / "manifest.json")
    plan = read(root / "plan.json")
    plan["matrix_manifest_sha256"] = declaration["matrix"]["sha256"]
    atomic_json(root / "plan.json", plan)
    declaration["partition_plan"] = binding(root / "plan.json")
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()
