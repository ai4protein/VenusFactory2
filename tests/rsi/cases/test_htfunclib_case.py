import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from agent.rsi import htfunclib_case as case


@pytest.mark.parametrize(
    "name,kind,extra",
    [
        ("../escape.csv", tarfile.REGTYPE, False),
        ("/absolute.csv", tarfile.REGTYPE, False),
        ("data.csv", tarfile.SYMTYPE, False),
        ("data.csv", tarfile.REGTYPE, True),
    ],
)
def test_archive_rejects_wrong_path_link_and_extra_members(tmp_path, name, kind, extra):
    path = tmp_path / "input.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.size = 4 if kind == tarfile.REGTYPE else 0
        archive.addfile(item, io.BytesIO(b"a\n1\n") if item.size else None)
        if extra:
            second = tarfile.TarInfo("extra")
            archive.addfile(second, io.BytesIO())
    with pytest.raises(ValueError, match="archive member"):
        case.read_csv_archive(path, "data.csv")


def test_notebook_identity_fails_closed(tmp_path):
    path = tmp_path / "RF_train.ipynb"
    path.write_text('{"cells":[]}')
    with pytest.raises(ValueError, match="notebook identity"):
        case.notebook_contract(path)


def test_legacy_filter_and_chain_are_distinct():
    binary = np.array([False, True, False, True])
    keep, _ = case.legacy_second_stage_rows(binary)
    assert keep.tolist() == [0, 1, 2, 3]
    assert case.chained_predictions(binary, ["G", "C", "G/C", "U"]).tolist() == ["U", "C", "U", "U"]
    with pytest.raises(ValueError, match="binary predictions"):
        case.legacy_second_stage_rows(["U", "F"])
    with pytest.raises(ValueError, match="binary type"):
        case.chained_predictions(np.array([0, 1]), ["U", "C"])


@pytest.mark.parametrize("mutation", ["feature", "column_order", "row_order", "labels"])
def test_partition_validation_detects_scientific_mismatch(mutation):
    expected = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    actual = expected.copy()
    truth = pd.Series(["U", "G"])
    labels = truth.to_frame("assign")
    if mutation == "feature":
        actual.loc[0, "a"] += 1e-5
    elif mutation == "column_order":
        actual = actual[["b", "a"]]
    elif mutation == "row_order":
        actual = actual.iloc[::-1]
    else:
        labels.loc[0, "assign"] = "C"
    with pytest.raises(ValueError, match="Released"):
        case.compare_partition(actual, labels, expected, truth)


def test_metrics_preserve_minority_class_and_boolean_labels():
    result = case.classification_metrics([False, False, False, True], [False, False, False, False], [False, True])
    assert result["accuracy"] == 0.75
    assert result["balanced_accuracy"] == 0.5
    assert result["support"] == [3, 1]
    with pytest.raises(ValueError, match="absent"):
        case.classification_metrics(["U"], ["U"], ["U", "C"])


def test_plugin_pins_full_cpu_allocation_and_isolated_environment(tmp_path):
    from agent.rsi.campaigns.case_plugins import invocation, list_case_plugins, resource_for

    def call(params):
        return invocation(
            case.CASE_ID,
            params,
            root=tmp_path,
            output=tmp_path / "research_runs/rsi/run",
            controller_python=tmp_path / "controller_python",
        )

    argv = call({})
    assert argv[0].endswith("research_runs/rsi/envs/htfunclib/bin/python")
    assert argv[2].endswith("scripts/rsi/cases/run_htfunclib_case.py")
    assert resource_for(case.CASE_ID, {}) == "cpu"
    catalog = next(x for x in list_case_plugins() if x["id"] == case.CASE_ID)
    assert catalog["parameters"]["additionalProperties"] is False
    for params in [{"n_estimators": 1}, {"seed": 43}, {"source": "/tmp/unbound"}, {"admit_only": True}]:
        with pytest.raises(ValueError, match="complete pinned"):
            call(params)


@pytest.mark.parametrize("status", ["running", "admitted", "failed"])
def test_incomplete_training_never_supports_claims(status):
    from agent.rsi.engine.case_evidence import canonical_claims

    with pytest.raises(ValueError, match="completed domain"):
        canonical_claims(case.CASE_ID, {"status": status})
