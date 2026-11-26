"""Source integrity, biological labels and partition invariants for localization."""

import copy
import csv
import hashlib
import io
import json

import pytest

from agent.rsi.datasets.deeploc_source import (
    FILES,
    HPA_LABELS,
    LABELS,
    SOURCE,
    export_source,
    normalize,
    summarize,
    table,
)


def raw_rows():
    swissprot = []
    for fold in range(5):
        row = {
            "": str(fold),
            "ACC": f"P{fold}",
            "Kingdom": "Metazoa",
            "Partition": str(fold),
            "Membrane": "0.0",
            "Sequence": "ACDEFGHIK" + "A" * fold,
        }
        row.update({label: "1.0" if label in {"Cytoplasm", "Nucleus"} else "0.0" for label in LABELS})
        swissprot.append(row)
    hpa = [
        {
            "sid": "ENSP1",
            "fasta": "WWWWAC",
            "Lengths": "6",
            **{label: "1" if label == "Nucleus" else "0" for label in HPA_LABELS},
        }
    ]
    return {"swissprot": swissprot, "hpa": hpa}


def snapshot(tmp_path, mutate=None):
    source = tmp_path / "raw"
    source.mkdir()
    cohorts = raw_rows()
    if mutate:
        mutate(cohorts)
    files = []
    for cohort, rows in cohorts.items():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        content = buf.getvalue().encode()
        filename = FILES[cohort]
        (source / filename).write_bytes(content)
        files.append(
            {
                "source": "deeploc2",
                "file": filename,
                "url": SOURCE + "data/" + filename,
                "status": "downloaded_not_admitted",
                "bytes": len(content),
                "rows": len(rows),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    path = source / "manifest.json"
    path.write_text(json.dumps({"schema": "property_source_snapshot.v1", "files": files}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_preserves_multilabel_unknowns_folds_and_source_coordinates(tmp_path):
    manifest, identity = snapshot(tmp_path)
    raw_before = {p.name: p.read_bytes() for p in manifest.parent.iterdir()}
    report = export_source(manifest, identity, tmp_path / "out")
    swissprot = [json.loads(line) for line in (tmp_path / "out/swissprot.jsonl").read_text().splitlines()]
    hpa = json.loads((tmp_path / "out/hpa.jsonl").read_text())
    assert [row["published_partition"] for row in swissprot] == list(range(5))
    assert swissprot[0]["labels"]["Nucleus"] == swissprot[0]["labels"]["Cytoplasm"] == 1
    assert swissprot[0]["csv_line"] == 2
    assert swissprot[0]["metadata"] == {"source_index": "0", "kingdom": "Metazoa", "membrane": 0}
    assert hpa["published_partition"] is None
    assert hpa["labels"]["Extracellular"] is None and hpa["labels"]["Plastid"] is None
    assert hpa["labels"]["Cytoplasm"] == 0
    assert hpa["protein_namespace"] == "ensembl_protein"
    assert set(hpa["observed_labels"]) == set(HPA_LABELS)
    summary = report["summary"]
    assert summary["source_dataset_count"] == 1 and summary["cohort_count"] == 2
    assert summary["admitted_tasks"] == summary["trained_models"] == summary["external_model_requests"] == 0
    assert summary["cohorts"]["swissprot"]["multilabel_rows"] == 5
    assert summary["cohorts"]["hpa"]["labels"]["Plastid"] == {
        "observed": 0,
        "positive": 0,
        "source_zero": 0,
        "unobserved": 1,
    }
    assert {p.name: p.read_bytes() for p in manifest.parent.iterdir()} == raw_before
    second = export_source(manifest, identity, tmp_path / "again")
    assert second == report
    for asset in report["observations"].values():
        assert asset["sha256"] == hashlib.sha256((tmp_path / "out" / asset["file"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="new directory"):
        export_source(manifest, identity, tmp_path / "out")


@pytest.mark.parametrize("asset", ["manifest.json", FILES["swissprot"], FILES["hpa"]])
def test_changed_source_is_rejected_without_partial_output(tmp_path, asset):
    manifest, identity = snapshot(tmp_path)
    path = manifest.parent / asset
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        export_source(manifest, identity, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "field,value",
    [("file", "../hpa_testset.csv"), ("url", "https://example.org/data.csv"), ("rows", 1), ("rows", True)],
)
def test_manifest_must_bind_the_actual_assets(tmp_path, field, value):
    manifest, _ = snapshot(tmp_path)
    obj = json.loads(manifest.read_text())
    obj["files"][0][field] = value
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        export_source(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), tmp_path / "out")


def test_source_symlink_rejected(tmp_path):
    manifest, identity = snapshot(tmp_path)
    target = manifest.parent / FILES["hpa"]
    alias = manifest.parent / "untracked.csv"
    target.rename(alias)
    target.symlink_to(alias)
    with pytest.raises(ValueError, match="symlink"):
        export_source(manifest, identity, tmp_path / "out")


@pytest.mark.parametrize(
    "field,value",
    [
        ("Partition", "5"),
        ("Cytoplasm", ""),
        ("Cytoplasm", "nan"),
        ("Nucleus", "2"),
        ("Membrane", "unknown"),
        ("Sequence", "ACD*"),
        ("Sequence", "acde"),
        ("Sequence", "AC D"),
        ("Kingdom", ""),
    ],
)
def test_invalid_source_values_are_not_silently_imputed(field, value):
    rows = raw_rows()["swissprot"]
    rows[0][field] = value
    with pytest.raises(ValueError):
        normalize("swissprot", rows)


def test_incomplete_annotated_hpa_column_is_not_treated_as_globally_unobserved():
    rows = raw_rows()["hpa"]
    rows[0]["Nucleus"] = ""
    with pytest.raises(ValueError, match="Invalid Nucleus"):
        normalize("hpa", rows)


def test_hpa_sequence_length_and_duplicate_ids_are_checked():
    rows = raw_rows()["hpa"]
    rows[0]["Lengths"] = "7"
    with pytest.raises(ValueError, match="length"):
        normalize("hpa", rows)
    with pytest.raises(ValueError, match="duplicate"):
        normalize("hpa", raw_rows()["hpa"] * 2)


def test_multilabel_protein_is_not_converted_to_a_single_class():
    row = normalize("swissprot", raw_rows()["swissprot"])[0]
    assert sum(value == 1 for value in row["labels"].values()) == 2
    assert "Membrane" not in row["labels"]


@pytest.mark.parametrize("kind", ["cross_fold", "external", "missing_fold"])
def test_partition_leakage_or_incomplete_folds_fail(kind):
    cohorts = {name: normalize(name, rows) for name, rows in raw_rows().items()}
    if kind == "cross_fold":
        cohorts["swissprot"][1]["sequence_sha256"] = cohorts["swissprot"][0]["sequence_sha256"]
    elif kind == "external":
        cohorts["hpa"][0]["sequence_sha256"] = cohorts["swissprot"][0]["sequence_sha256"]
    else:
        cohorts["swissprot"].pop()
    with pytest.raises(ValueError):
        summarize(cohorts)


def test_same_fold_sequence_duplicates_are_retained_and_counted():
    cohorts = {name: normalize(name, rows) for name, rows in raw_rows().items()}
    row = copy.deepcopy(cohorts["swissprot"][0])
    row.update(id="deeploc2:swissprot:Piso", protein_id="Piso")
    cohorts["swissprot"].append(row)
    summary = summarize(cohorts)
    assert summary["swissprot_duplicate_sequence_rows_retained"] == 1
    row["labels"]["Nucleus"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        summarize(cohorts)


@pytest.mark.parametrize("content", [b"ACC,ACC\nx,y\n", b"sid,Nucleus\nx,1,2\n"])
def test_malformed_table_columns_do_not_pass(content):
    with pytest.raises(ValueError, match="columns"):
        table(content, "hpa")
