import copy
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from agent.rsi import protpredict_source as source


def frames():
    proteins = pd.DataFrame(
        {"Unnamed: 0": ["g1", "g2", "g1", "g3"], "p1": [2.0, np.nan, 4.0, 9.0], "p2": [np.nan, 5.0, 8.0, 7.0]}
    )
    metabolites = pd.DataFrame(
        {
            "ORF": ["g2", "g1", "other"],
            "gene": ["a", "b", "c"],
            **{name: [i + 1.0, i + 2.0, i + 3.0] for i, name in enumerate(source.FEATURES)},
        }
    )
    return proteins, metabolites


def test_exact_join_mean_missing_targets_and_column_order():
    genotypes, proteins, x, y, provenance, summary = source.combine_frames(*frames())
    assert genotypes == ["g1", "g2"] and proteins == ["p1", "p2"]
    np.testing.assert_equal(y, [[3.0, 8.0], [np.nan, 5.0]])
    np.testing.assert_equal(x, [np.arange(2, 21), np.arange(1, 20)])
    assert provenance["g1"] == {"protein_csv_rows": [2, 4], "metabolite_xls_row": 3}
    assert summary["duplicate_rows_averaged"] == 1 and summary["observed_targets"] == 3
    assert summary["unmatched_proteome_genotypes"] == 1


def test_no_imputation_and_no_target_based_predictor_changes():
    proteins, metabolites = frames()
    original = source.combine_frames(proteins, metabolites)
    proteins["p1"] = [999.0, 123.0, 777.0, 456.0]
    changed = source.combine_frames(proteins, metabolites)
    np.testing.assert_array_equal(original[2], changed[2])
    metabolites.loc[0, "alanine"] = np.nan
    filtered = source.combine_frames(proteins, metabolites)
    assert filtered[0] == ["g1"]
    assert filtered[-1]["incomplete_covariate_genotypes_excluded"] == 1


@pytest.mark.parametrize("bad", [np.inf, -1.0, "invalid"])
def test_invalid_measurements_fail(bad):
    proteins, metabolites = frames()
    proteins["p1"] = proteins["p1"].astype(object)
    proteins.loc[0, "p1"] = bad
    with pytest.raises((ValueError, TypeError)):
        source.combine_frames(proteins, metabolites)


@pytest.mark.parametrize(
    "change", ["duplicate_gene", "whitespace", "missing_feature", "reordered_feature", "duplicate_protein"]
)
def test_ambiguous_identities_and_features_rejected(change):
    proteins, metabolites = frames()
    if change == "duplicate_gene":
        metabolites.loc[1, "ORF"] = "g2"
    elif change == "whitespace":
        proteins.loc[0, "Unnamed: 0"] = " g1"
    elif change == "missing_feature":
        metabolites = metabolites.drop(columns="alanine")
    elif change == "reordered_feature":
        metabolites = metabolites[list(metabolites.columns[:2]) + list(reversed(source.FEATURES))]
    else:
        proteins.columns = ["Unnamed: 0", "p1", "p1"]
    with pytest.raises(ValueError):
        source.combine_frames(proteins, metabolites)


@pytest.fixture
def bound(tmp_path, monkeypatch):
    proteins, metabolites = frames()
    p = tmp_path / "proteins.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("proteomics_messner.csv", proteins.to_csv(index=False))
        z.writestr("__MACOSX/._proteomics_messner.csv", b"resource fork")
    m = tmp_path / "metabolites.xls"
    m.write_bytes(b"synthetic workbook placeholder")
    readme = tmp_path / "readme.md"
    readme.write_text("synthetic test source")

    def bind(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    declaration = {
        "schema": source.SCHEMA,
        "repository_commit": "a" * 40,
        "paper": {
            "title": "Interpreting protein abundance in Saccharomyces cerevisiae through relational learning",
            "doi": source.PAPER_DOI,
        },
        "proteomics": bind(p),
        "metabolites": bind(m),
        "references": [bind(readme)],
    }
    monkeypatch.setattr(source.pd, "read_excel", lambda *a, **k: metabolites.copy())
    return declaration


def test_immutable_source_bound_matrix_export(tmp_path, bound):
    out = tmp_path / "output"
    report = source.prepare(bound, out)
    assert report["new_models"] == report["external_model_calls"] == 0
    assert report["acquisition_admitted"] is False and report["partitions_created"] is False
    for name, sha in report["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    np.testing.assert_equal(np.load(out / "targets.npy", allow_pickle=False), [[3.0, 8.0], [np.nan, 5.0]])
    assert json.loads((out / "covariates.json").read_text()) == [{"name": n, "unit": "mM"} for n in source.FEATURES]
    with pytest.raises(ValueError, match="already exists"):
        source.prepare(bound, out)


def test_source_drift_prevents_output(tmp_path, bound):
    altered = copy.deepcopy(bound)
    altered["proteomics"]["sha256"] = "0" * 64
    out = tmp_path / "absent"
    with pytest.raises(ValueError, match="SHA256"):
        source.prepare(altered, out)
    assert not out.exists()


def test_unknown_source_and_csv_headers_fail(tmp_path, bound):
    bad = copy.deepcopy(bound)
    bad["schema"] = "unknown"
    with pytest.raises(ValueError):
        source.prepare(bad, tmp_path / "bad")
    p = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("proteomics_messner.csv", "Unnamed: 0,p1,p1\ng1,2,3\n")
    bad = copy.deepcopy(bound)
    bad["proteomics"] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="Duplicate CSV"):
        source.prepare(bad, tmp_path / "duplicate")
