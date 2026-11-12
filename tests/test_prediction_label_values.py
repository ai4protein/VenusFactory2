"""Display paths must not invent a scaler or convert a value twice."""

import pandas as pd
import pytest

from mcp_tools.common.constants import DATASET_TO_TASK_MAP, LABEL_MAPPING_FUNCTION
from mcp_tools.common.data_processors import map_prediction_labels
from mcp_tools.common.label_mappers import map_labels, map_labels_individual


@pytest.mark.parametrize("value", [0.5, 53.55, -0.9709106683731079])
@pytest.mark.parametrize("dataset,task", [("Thermostability", "Stability"), ("DeepET_Topt", "Optimal Temperature")])
def test_regression_display_preserves_the_inference_scale(value, dataset, task):
    row = pd.Series({"Dataset": dataset, "prediction": value})
    assert map_labels(row, task) == round(value, 2)
    assert map_prediction_labels(row, task) == round(value, 2)
    text = map_labels_individual(row, task, {"Stability": [40.1995166, 66.8968874]})
    assert float(text) == round(value, 3)
    assert "°C" not in text


@pytest.mark.parametrize("dataset", ["DeepLocBinary", "DeepLocMulti"])
def test_classification_labels_still_use_the_dataset_mapping(dataset):
    row = pd.Series({"Dataset": dataset, "prediction": 1, "predicted_class": 1})
    expected = LABEL_MAPPING_FUNCTION[DATASET_TO_TASK_MAP[dataset]][1]
    assert map_labels(row, "Subcellular Localization") == expected
    assert map_prediction_labels(row, "Subcellular Localization") == expected
    assert map_labels_individual(row, "Subcellular Localization") == expected


def test_sorting_signals_keep_multiple_labels():
    row = pd.Series({"Dataset": "SortingSignal", "prediction": "[1,0,0,0,0,0,1,0,0]", "predicted_class": "[1,0,0,0,0,0,1,0,0]"})
    assert map_labels(row, "Sorting Signal") == "CH_SP"
    assert map_prediction_labels(row, "Sorting Signal") == "CH_SP"
    assert map_labels_individual(row, "Sorting Signal") == "CH_SP"


def test_report_table_preserves_actual_csv_regression_value(tmp_path, monkeypatch):
    from pathlib import Path

    from mcp_tools.tools.predict import report_workflows as report

    # Exercise the report's CSV ingestion and actual label formatting. Only
    # the neural CLI invocation is replaced with a known prediction file.
    script = tmp_path / "mcp_tools/models/factory/prediction/esm2.py"
    script.parent.mkdir(parents=True)
    script.touch()
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report, "get_save_path", lambda *args: tmp_path)
    monkeypatch.setattr(report, "ensure_project_ckpt", lambda path: tmp_path)
    monkeypatch.setattr(report, "DATASET_MAPPING_FUNCTION", {"Stability": ["Thermostability"]})

    def neural_cli(cmd):
        destination = Path(cmd[cmd.index("--output_csv") + 1])
        pd.DataFrame([{"header": "query", "sequence": "MKTAYI", "prediction": 0.5}]).to_csv(destination, index=False)

    monkeypatch.setattr(report, "run_finetuned_cli", neural_cli)
    text = report.handle_individual_function_prediction("MKTAYI", "A", None, None, None)
    assert "**Thermal Stability**" in text
    assert "`0.5` (model output; scale and unit unspecified)" in text
    assert "53.55" not in text and "°C" not in text
