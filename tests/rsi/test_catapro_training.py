from pathlib import Path

import numpy as np
import pytest

from agent.rsi.cases.catapro_case import file_sha
from agent.rsi.cases.catapro_training import STAGES, make_table, training_arguments, validate_fold_batches, verify_files


def inputs():
    rows = [
        {
            "id": f"row_{i}",
            "fold": i % 10,
            "targets": {
                "kcat": np.log10(2 + i),
                "km": np.log10(0.05 + i / 100),
                "activity": np.log10((2 + i) / (0.05 + i / 100)),
            },
        }
        for i in range(40)
    ]
    features = np.arange(40 * 1959, dtype=np.float32).reshape(40, 1959) / 1000
    raw = np.asarray([[2 + i, 0.05 + i / 100] for i in range(40)])
    return rows, features, raw


@pytest.mark.parametrize("target", ["kcat", "km", "activity"])
def test_training_table_preserves_features_fold_order_and_label_units(target):
    rows, features, raw = inputs()
    table = make_table(rows, features, target, raw)
    assert table.index.tolist() == [r["id"] for r in rows]
    assert np.array_equal(table.iloc[:, :1959].to_numpy(), features)
    assert np.array_equal(table.iloc[:, -1].to_numpy(), [r["fold"] for r in rows])
    if target == "activity":
        assert table.shape == (40, 1962)
        assert np.array_equal(table.iloc[:, -3:-1].to_numpy(), raw.astype(np.float32))
    else:
        assert table.shape == (40, 1961)
        assert np.array_equal(
            table.iloc[:, -2].to_numpy(), np.asarray([r["targets"][target] for r in rows], dtype=np.float32)
        )
    assert sum(r["validation_rows"] for r in validate_fold_batches(table, 8)) == 40


@pytest.mark.parametrize(
    "problem", ["duplicate_id", "missing_fold", "wrong_width", "nonfinite", "wrong_activity_units"]
)
def test_training_table_rejects_alignment_or_conversion_problems(problem):
    rows, features, raw = inputs()
    if problem == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "missing_fold":
        for row in rows:
            if row["fold"] == 9:
                row["fold"] = 8
    elif problem == "wrong_width":
        features = features[:, :-1]
    elif problem == "nonfinite":
        features[1, 2] = np.nan
    else:
        raw[:, 1] /= 1000
    with pytest.raises(ValueError):
        make_table(rows, features, "activity", raw)


def test_training_does_not_silently_drop_singleton_batches():
    rows, features, raw = inputs()
    table = make_table(rows[:11], features[:11], "activity", raw[:11])
    with pytest.raises(ValueError, match="singleton"):
        validate_fold_batches(table, 8)


@pytest.mark.parametrize("stage", list(STAGES))
def test_author_arguments_use_explicit_published_parameters_and_new_dependencies(tmp_path, stage):
    args = training_arguments(tmp_path, stage, 150)
    values = dict(zip(args[::2], args[1::2], strict=True))
    target = STAGES[stage][1]
    expected = {"kcat": (1e-5, 8, 0), "km": (1e-5, 16, 0.1), "activity": (0.01, 16, 0.1)}[target]
    assert (
        float(values["-lr"]),
        int(values["-batch_size"]),
        float(values.get("-rate", values.get("-drop_rate"))),
    ) == expected
    assert values["-epochs"] == "150"
    if target == "activity":
        assert values["-alpha"] == "0.4"
        assert values["-kcat_model_dpath"] == str(tmp_path / "stages/activity_kcat")
        assert values["-Km_model_dpath"] == str(tmp_path / "stages/activity_km")


@pytest.mark.parametrize("epochs", [0, 151, True])
def test_invalid_epoch_budget_rejected(epochs):
    with pytest.raises(ValueError):
        training_arguments(Path("/tmp/fixture"), "kcat", epochs)


def test_source_binding_rejects_content_change_or_symlink(tmp_path):
    path = tmp_path / "model.py"
    path.write_text("SOURCE")
    bindings = {"model.py": file_sha(path)}
    verify_files(tmp_path, bindings)
    path.write_text("CHANGED")
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, bindings)
    path.unlink()
    other = tmp_path / "original.py"
    other.write_text("SOURCE")
    path.symlink_to(other)
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, bindings)


def test_training_snapshot_runs_without_original_checkout(tmp_path):
    import os
    import subprocess
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    from scripts.rsi.cases.run_catapro_training import IMPLEMENTATION, ROOT

    for name in IMPLEMENTATION:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/rsi/cases/run_catapro_training.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--pdf-review" in result.stdout


@pytest.mark.parametrize("external", [False, True])
def test_full_row_audit_serializes_float32_labels(tmp_path, monkeypatch, external):
    import csv
    import json

    import torch

    from agent.rsi import catapro_training as training

    rows, features, raw = inputs()
    labels = np.asarray([r["targets"]["kcat"] for r in rows], dtype=np.float32)
    predictions = labels + np.float32(0.1)
    features[:, 0] = predictions
    table = make_table(rows, features, "kcat")
    table_path = tmp_path / "table.pkl"
    table.to_pickle(table_path)
    folder = tmp_path / "stages/kcat"
    folder.mkdir(parents=True)
    for fold in range(10):
        torch.save({}, folder / f"{fold}_bestmodel.pth")
    with (folder / "final_pred_label.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred", "label"])
        w.writerows(
            (row["id"], f"{pred:.4f}", f"{label:.4f}")
            for row, pred, label in zip(rows, predictions, labels, strict=True)
        )
    expected = training.metrics(labels, predictions)
    with (folder / "final_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(expected))
        w.writeheader()
        w.writerow(expected)
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            {"tables": {"kcat": {"path": "table.pkl", "sha256": file_sha(table_path)}}, "epochs": 1, "scope": "fixture"}
        )
    )
    monkeypatch.setattr(training, "numpy_forward", lambda feats, state, **kwargs: feats[:, 0])
    before = {str(p.relative_to(tmp_path)): file_sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    destination = tmp_path.parent / (tmp_path.name + "_external_audit") if external else folder
    result = training.audit_stage(tmp_path, "kcat", output=destination if external else None)
    if external:
        after = {str(p.relative_to(tmp_path)): file_sha(p) for p in tmp_path.rglob("*") if p.is_file()}
        assert before == after
    assert result["rows"] == 40 and result["folds"] == 10
    assert json.loads((destination / "audit.json").read_text()) == result
    assert result["max_label_difference_from_rounded_author_csv"] <= 6e-5
