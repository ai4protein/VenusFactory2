"""Frozen biological partitions and protected Scientist ingestion, not model results."""

import copy
import csv
import hashlib
import io
import json

import pytest
from test_deeploc_source import snapshot

from agent.harness.research_data_store import ResearchDataStore
from agent.harness.research_dataset import prepare_dataset
from agent.rsi.datasets.deeploc_protocol import export_protocol, selected_splits
from agent.rsi.datasets.deeploc_source import export_source


def curate(tmp_path):
    def expand(cohorts):
        cohorts["swissprot"] = [
            {**row, "ACC": row["ACC"] + "-" + str(i), "Sequence": row["Sequence"] + "C" * (i + 1)}
            for row in cohorts["swissprot"]
            for i in range(8)
        ]

    source, identity = snapshot(tmp_path, expand)
    export_source(source, identity, tmp_path / "curated")
    path = tmp_path / "curated/source.json"
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_folds_sequence_only_inputs_and_protected_target_handoff(tmp_path):
    path, identity = curate(tmp_path)
    report = export_protocol(path, identity, tmp_path / "protocol")
    assert report["cohorts"]["train"]["selected"] == 24
    assert report["cohorts"]["validation"]["selected"] == report["cohorts"]["test"]["selected"] == 8
    assert report["selection"]["all_training_folds_retained"]
    assert report["selection"]["uses_targets"] is False
    assert report["model_task_count"] == 1 and report["trained_models"] == 0
    assert not report["external_hpa"]["used_for_fit_or_selection"]
    data = (tmp_path / "protocol/dataset.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    for row in rows:
        assert row["split"] == (
            "train"
            if int(row["published_partition"]) < 3
            else "validation"
            if row["published_partition"] == "3"
            else "test"
        )
        assert not {"Kingdom", "Membrane", "metadata"} & row.keys()
    work = tmp_path / "workspace"
    work.mkdir()
    store = ResearchDataStore(tmp_path / "private", work)
    store.ingest("inputs/localization.csv", data, report["artifacts"]["dataset.csv"]["sha256"])
    prepared = prepare_dataset(work, data_store=store, **report["native_prepare"])
    visible = list(csv.DictReader(io.StringIO((work / "inputs/localization.csv").read_text())))
    assert len(visible) == 24
    assert {row["id"] for row in visible} == {row["id"] for row in rows if row["split"] == "train"}
    frozen = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    assert set(frozen["splits"]["train"][0]["inputs"]) == {"sequence"}
    assert prepared["evaluation_access"] == "host_reference"
    handle = json.loads((work / prepared["evaluation_file"]).read_text())
    assert set(handle) == {"schema", "id"}
    assert handle["schema"] == "venusfactory.research-evaluation.v1"
    assert len(handle["id"]) == 64
    for name, asset in report["artifacts"].items():
        assert hashlib.sha256((tmp_path / "protocol" / name).read_bytes()).hexdigest() == asset["sha256"]
    again = export_protocol(path, identity, tmp_path / "again")
    assert again == report
    assert (tmp_path / "protocol/dataset.csv").read_bytes() == (tmp_path / "again/dataset.csv").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        export_protocol(path, identity, tmp_path / "protocol")


def test_selection_is_stable_label_blind_and_training_is_not_capped():
    rows = [
        {
            "id": f"{fold}-{i}",
            "sequence_sha256": hashlib.sha256(f"sequence{fold}-{i}".encode()).hexdigest(),
            "published_partition": fold,
            "labels": {"location": i % 2},
        }
        for fold in range(5)
        for i in range(600)
    ]
    expected = selected_splits(rows)
    modified = copy.deepcopy(list(reversed(rows)))
    for row in modified:
        row["labels"] = {"different": None}
        row["metadata"] = {"irrelevant": "annotation"}
    actual = selected_splits(modified)
    assert {name: [row["id"] for row in values] for name, values in actual.items()} == {
        name: [row["id"] for row in values] for name, values in expected.items()
    }
    assert {name: len(values) for name, values in expected.items()} == {"train": 1800, "validation": 512, "test": 512}


@pytest.mark.parametrize("filename", ["source.json", "swissprot.jsonl", "hpa.jsonl"])
def test_changed_curated_source_fails_before_export(tmp_path, filename):
    path, identity = curate(tmp_path)
    asset = path.parent / filename
    asset.write_bytes(asset.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        export_protocol(path, identity, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_source_symlink_is_rejected(tmp_path):
    path, identity = curate(tmp_path)
    asset = path.parent / "swissprot.jsonl"
    saved = path.parent / "swissprot-copy.jsonl"
    asset.rename(saved)
    asset.symlink_to(saved)
    with pytest.raises(ValueError, match="symlink"):
        export_protocol(path, identity, tmp_path / "out")
