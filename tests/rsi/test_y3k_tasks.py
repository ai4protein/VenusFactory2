"""Task/lifecycle boundary checks on synthetic source-bound matrices; no model fits."""

import hashlib
import json

import numpy as np
import pytest

from agent.rsi import model_tasks
from agent.rsi import y3k_tasks as api
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.y3k_source import FEATURES, PAPER, SHEETS, UNIT
from agent.rsi.y3k_source import SCHEMA as SOURCE_SCHEMA


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def refresh(root, declaration):
    matrix = read(root / "manifest.json")
    matrix["files_sha256"] = {name: sha256_file(root / name) for name in api.FILES}
    atomic_json(root / "manifest.json", matrix)
    declaration["matrix"] = binding(root / "manifest.json")
    plan = read(root / "plan.json")
    plan["matrix_manifest"] = declaration["matrix"]
    atomic_json(root / "plan.json", plan)
    declaration["partition_plan"] = binding(root / "plan.json")
    audit = read(root / "audit.json")
    audit["matrix_manifest_sha256"] = declaration["matrix"]["sha256"]
    atomic_json(root / "audit.json", audit)
    declaration["audit"] = binding(root / "audit.json")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "matrix"
    root.mkdir()
    terms = root / "source_terms.txt"
    terms.write_text("Synthetic scientific-data fixture; not a source study")
    original = {"schema": SOURCE_SCHEMA, "paper": PAPER, "references": [binding(terms)]}
    for name in ("strains", "biomolecules", "quantitative"):
        path = root / (name + ".xlsx")
        path.write_text("Synthetic source binding only; not a workbook")
        original[name] = binding(path)
    genes = [f"synthetic_deletion_{i}" for i in range(100)]
    proteins = ["synthetic_P1", "synthetic_P2"]
    atomic_json(root / "source.json", original)
    atomic_json(root / "strain_metadata.json", {gene: {"systematic": gene} for gene in genes})
    atomic_json(root / "protein_metadata.json", {})
    atomic_json(root / "excluded_proteins.json", {})
    order = sorted(genes, key=lambda g: hashlib.sha256((api.SEED + "\0" + g).encode()).hexdigest())
    plan = {
        "schema": "y3k_partition_plan.v1",
        "seed": api.SEED,
        "partitions": {"training": order[:60], "validation": order[60:80], "future": order[80:]},
        "conditions": {},
        "acquisition_admitted": False,
        "models_trained": 0,
        "external_model_calls": 0,
    }
    originals = {}
    for number, condition in enumerate(SHEETS):
        folder = root / condition
        folder.mkdir()
        local_genes = list(reversed(genes)) if number else list(genes)
        x = np.random.default_rng(number + 15).normal(0, 1, (100, 19))
        y = np.column_stack([x.sum(axis=1), x[:, 0] * 3 + number])
        x[0, -1] = np.nan
        y[1, 0] = np.nan
        np.save(folder / "features.npy", x, allow_pickle=False)
        np.save(folder / "targets.npy", y, allow_pickle=False)
        atomic_json(folder / "genotypes.json", local_genes)
        atomic_json(folder / "proteins.json", proteins)
        witness = {
            "columns": [
                {"column": i + 3, "source_label": "∆" + gene, "genotype": gene} for i, gene in enumerate(local_genes)
            ],
            "features": {name: {"row": i + 2, "unit": UNIT} for i, name in enumerate(FEATURES)},
            "proteins": {
                protein: {"row": i + 21, "name": protein, "locus": "synthetic_locus_" + str(i)}
                for i, protein in enumerate(proteins)
            },
            "excluded_protein_rows": [],
        }
        atomic_json(folder / "row_sources.json", witness)
        positions = {g: i for i, g in enumerate(local_genes)}
        counts = {
            protein: {
                phase: sum(np.isfinite(x[positions[g]]).all() and np.isfinite(y[positions[g], j]) for g in group)
                for phase, group in plan["partitions"].items()
            }
            for j, protein in enumerate(proteins)
        }
        counts = {p: {phase: int(v) for phase, v in c.items()} for p, c in counts.items()}
        plan["conditions"][condition] = {
            "resolved_proteins": 2,
            "proteins_with_minimum_partition_pairs": 2,
            "minimum_rows": api.MINIMUM,
            "per_protein_counts": counts,
        }
        originals[condition] = (local_genes, proteins, x, y)
    matrix = {
        "schema": SOURCE_SCHEMA,
        "status": "source_reconstructed",
        "source": original,
        "files_sha256": {name: sha256_file(root / name) for name in api.FILES},
        "acquisition_admitted": False,
        "covariates": [{"name": n, "unit": UNIT} for n in FEATURES],
        "target": {"name": "protein_abundance_log2_fold_change", "unit": UNIT},
        "observation_unit": "Synthetic genotype summaries",
        "preprocessing": "Synthetic fixture only",
        "limitations": ["Not scientific evidence"],
    }
    atomic_json(root / "manifest.json", matrix)
    plan["matrix_manifest"] = binding(root / "manifest.json")
    atomic_json(root / "plan.json", plan)
    atomic_json(
        root / "audit.json",
        {"passed": True, "raw_hashes_rechecked": True, "matrix_manifest_sha256": sha256_file(root / "manifest.json")},
    )
    declaration = {
        "schema": api.SCHEMA,
        "matrix": binding(root / "manifest.json"),
        "partition_plan": binding(root / "plan.json"),
        "audit": binding(root / "audit.json"),
        "source_terms": binding(terms),
    }
    return root, declaration, originals


def test_catalog_exact_measurements_missingness_and_cross_condition_groups(source, tmp_path):
    root, declaration, originals = source
    output = tmp_path / "export"
    report = api.prepare(declaration, output)
    assert report["catalog_statuses"] == {"admitted": 4} and report["exported_tasks"] == 4
    assert report["external_model_calls"] == report["new_trained_models"] == report["new_published_models"] == 0
    plan = read(root / "plan.json")
    all_groups = {}
    for candidate in read(output / "manifest.json")["candidates"]:
        definition, data = model_tasks.read_task(model_tasks.bound_file(candidate["task"]))
        assert definition["plugin_id"] == "generated_context.v2" and definition["scope"]["unit"] == UNIT
        condition = definition["scope"]["context"]["growth_condition"]
        genes, proteins, x, y = originals[condition]
        j = proteins.index(definition["scope"]["protein"])
        for phase, rows in data.items():
            expected = [
                genes.index(g)
                for g in plan["partitions"][phase]
                if np.isfinite(x[genes.index(g)]).all() and np.isfinite(y[genes.index(g), j])
            ]
            assert len(rows) == len(expected)
            for row, i in zip(rows, expected, strict=True):
                np.testing.assert_array_equal(row["values"], x[i])
                assert row["target"] == y[i, j]
                assert row["group"] == PAPER["doi"] + ":" + genes[i]
                assert all_groups.setdefault(row["group"], phase) == phase
    with pytest.raises(ValueError, match="New absolute"):
        api.prepare(declaration, output)


@pytest.mark.parametrize("attack", ["split", "seed", "minimum", "feasibility", "condition", "matrix_binding"])
def test_rehashed_plan_cannot_change_frozen_policy(source, tmp_path, attack):
    root, declaration, _ = source
    plan = read(root / "plan.json")
    if attack == "split":
        a, b = plan["partitions"]["training"], plan["partitions"]["future"]
        a[0], b[0] = b[0], a[0]
    elif attack == "seed":
        plan["seed"] = "new_seed"
    elif attack == "minimum":
        plan["conditions"]["respiration"]["minimum_rows"]["training"] = 1
    elif attack == "feasibility":
        plan["conditions"]["respiration"]["proteins_with_minimum_partition_pairs"] = 999
    elif attack == "condition":
        plan["conditions"]["adjusted_respiration"] = plan["conditions"]["respiration"]
    else:
        plan["matrix_manifest"]["sha256"] = "0" * 64
    atomic_json(root / "plan.json", plan)
    declaration["partition_plan"] = binding(root / "plan.json")
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "name",
    ["respiration/features.npy", "fermentation/targets.npy", "respiration/row_sources.json", "quantitative.xlsx"],
)
def test_source_drift_is_rejected_before_materialization(source, tmp_path, name):
    root, declaration, _ = source
    with (root / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("attack", ["feature_inf", "feature_domain", "target_inf", "witness", "target_units", "audit"])
def test_invalid_data_or_audit_rejected_with_fresh_bindings(source, tmp_path, attack):
    root, declaration, _ = source
    if attack in {"feature_inf", "feature_domain", "target_inf"}:
        name = "targets.npy" if attack == "target_inf" else "features.npy"
        path = root / "respiration" / name
        a = np.load(path, allow_pickle=False)
        a[1, 0] = 65.0 if attack == "feature_domain" else np.inf
        np.save(path, a, allow_pickle=False)
    elif attack == "witness":
        path = root / "respiration/row_sources.json"
        d = read(path)
        d["columns"][0]["column"] = 999
        atomic_json(path, d)
    elif attack == "target_units":
        path = root / "manifest.json"
        d = read(path)
        d["target"]["unit"] = "mM"
        atomic_json(path, d)
    else:
        path = root / "audit.json"
        d = read(path)
        d["passed"] = False
        atomic_json(path, d)
    refresh(root, declaration)
    with pytest.raises(ValueError):
        api.prepare(declaration, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
