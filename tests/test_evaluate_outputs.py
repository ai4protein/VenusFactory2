"""Real CPU tensors verify Factory evaluation without loading any PLM or data service."""
import json

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAUROC, MultilabelAUROC
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef

import evaluate as evaluation


class TinyModel(nn.Module):
    def __init__(self, inputs, outputs, weights=None):
        super().__init__()
        self.layer = nn.Linear(inputs, outputs, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(outputs, inputs) if weights is None else torch.tensor(weights))

    def forward(self, plm_model, batch):
        assert not torch.is_grad_enabled()
        assert not self.training
        return self.layer(batch["features"])


class CountingMSE(nn.MSELoss):
    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)
        self.calls = []

    def forward(self, prediction, target):
        self.calls.append((tuple(prediction.shape), tuple(target.shape)))
        return super().forward(prediction, target)


def loader(features, labels, batch_size=2, label_column="label"):
    return DataLoader([{ "features": torch.tensor(x, dtype=torch.float32), label_column: torch.tensor(y)}
                       for x, y in zip(features, labels)], batch_size=batch_size, shuffle=False)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_regression_continuous_predictions_metrics_and_loss_match_csv(tmp_path, monkeypatch, reduction):
    model = TinyModel(1, 1, [[2.]])
    metrics = {"custom_mse": MeanSquaredError(), "rank": SpearmanCorrCoef()}
    original_metric = metrics["custom_mse"]
    monkeypatch.setattr(evaluation, "metrics_dict", {"wrong_global": object()}, raising=False)
    criterion = CountingMSE(reduction)
    result, predicted = evaluation.evaluate(model, None, metrics,
        loader([[1.], [2.], [3.]], [1., 5., 9.], label_column="measurement"), criterion,
        problem_type="regression", label_column_name="measurement")
    assert predicted == [2., 4., 6.]
    assert criterion.calls == [((2,), (2,)), ((1,), (1,))]
    assert result["loss"] == pytest.approx([11 / 3])
    assert result["custom_mse"] == pytest.approx([11 / 3])
    assert result["rank"] == pytest.approx([1.], abs=1e-5)
    assert metrics["custom_mse"] is original_metric
    assert "loss" not in metrics
    frame = pd.DataFrame({"sequence": ["A", "C", "D"], "measurement": [1., 5., 9.]})
    evaluation.save_evaluation_results(result, predicted, frame, tmp_path)
    saved = pd.read_csv(tmp_path / "evaluation_result.csv")
    saved_metrics = pd.read_csv(tmp_path / "evaluation_metrics.csv")
    assert saved["pred_label"].tolist() == predicted
    assert saved["sequence"].tolist() == frame["sequence"].tolist()
    assert saved_metrics["loss"].iloc[0] == pytest.approx(11 / 3)
    assert "pred_label" not in frame


def test_empty_metrics_still_compute_one_loss_per_batch_and_preserve_singleton():
    criterion = CountingMSE()
    result, predicted = evaluation.evaluate(TinyModel(1, 1), None, {},
        loader([[3.]], [2.], batch_size=1), criterion, problem_type="regression")
    assert predicted == [3.]
    assert result == {"loss": [1.]}
    assert criterion.calls == [((1,), (1,))]


def test_binary_auroc_uses_positive_softmax_scores_not_labels_or_second_logit():
    # All argmax predictions are negative, but positive-class ranks are perfect.
    logits = [[3., 2.], [100., 98.], [3., 2.5], [2., -1.]]
    labels = [1, 0, 1, 0]
    metrics = {"renamed_score": BinaryAUROC(), "accuracy": BinaryAccuracy()}
    result, predicted = evaluation.evaluate(TinyModel(2, 2), None, metrics,
        loader(logits, labels), nn.CrossEntropyLoss(), problem_type="single_label_classification")
    assert predicted == [0, 0, 0, 0]
    assert result["renamed_score"] == [1.]
    assert result["accuracy"] == [.5]
    expected_loss = nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels)).item()
    assert result["loss"] == pytest.approx([expected_loss])


def test_multiclass_auroc_receives_scores_and_singleton_batch_is_kept():
    logits = [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]]
    metric = MulticlassAUROC(num_classes=3)
    result, predicted = evaluation.evaluate(TinyModel(3, 3), None, {"renamed_auc": metric},
        loader(logits, [0, 1, 2]), nn.CrossEntropyLoss(), problem_type="single_label_classification")
    assert predicted == [0, 1, 2]
    assert result["renamed_auc"] == [1.]
    assert result["loss"] == pytest.approx([torch.log(torch.tensor(1 + 2 * torch.exp(torch.tensor(-3.)).item())).item()])


def test_multilabel_uses_one_sigmoid_and_independent_thresholds(tmp_path):
    logits = [[.2, -.2], [-.1, .1], [0., .8]]
    targets = [[1, 0], [0, 1], [0, 1]]
    metric = MultilabelAUROC(num_labels=2)
    metrics = {"renamed_auc": metric, "f1_max": evaluation.MultilabelF1Max(num_labels=2)}
    result, predicted = evaluation.evaluate(TinyModel(2, 2), None, metrics,
        loader(logits, targets), nn.BCEWithLogitsLoss(), problem_type="multi_label_classification")
    assert predicted == [[1, 0], [0, 1], [0, 1]]
    assert result["renamed_auc"] == [1.]
    assert result["f1_max"] == pytest.approx([1.])
    assert torch.allclose(torch.cat(metric.preds), torch.tensor(logits).sigmoid())
    assert result["loss"] == pytest.approx([nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits), torch.tensor(targets, dtype=torch.float32)).item()])
    evaluation.save_evaluation_results(result, predicted, pd.DataFrame({"id": ["a", "b", "c"]}), tmp_path)
    assert [json.loads(x) for x in pd.read_csv(tmp_path / "evaluation_result.csv")["pred_label"]] == predicted


def test_binary_one_logit_uses_bce_and_retains_singleton():
    result, predicted = evaluation.evaluate(TinyModel(1, 1), None, {"auc": BinaryAUROC()},
        loader([[-1.], [1.], [-2.]], [0, 1, 0]), evaluation.evaluation_loss("single_label_classification", 1),
        problem_type="single_label_classification")
    assert predicted == [0, 1, 0]
    assert result["auc"] == [1.]


def test_metric_construction_matches_task_and_accepts_loss_only():
    assert evaluation.build_metrics([], "regression", 1) == {}
    metrics = evaluation.build_metrics(["accuracy", "auroc", "f1_max"], "multi_label_classification", 2)
    assert isinstance(metrics["auroc"], MultilabelAUROC)
    assert metrics["accuracy"].num_labels == 2
    assert isinstance(evaluation.evaluation_loss("regression", 1), nn.MSELoss)
    assert isinstance(evaluation.evaluation_loss("multi_label_classification", 2), nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError, match="Invalid regression metric"):
        evaluation.build_metrics(["accuracy"], "regression", 1)


def test_empty_dataset_and_dropped_rows_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="no samples"):
        evaluation.evaluate(TinyModel(1, 1), None, {}, loader([], []), nn.MSELoss(), problem_type="regression")
    with pytest.raises(ValueError, match="must not drop samples"):
        evaluation.save_evaluation_results({"loss": [0.]}, [1.], pd.DataFrame({"id": [1, 2]}), tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["none", "standard", "min_max", "robust", "log", "quantile"])
def test_checkpoint_transform_restores_raw_scale_for_metrics_loss_and_csv(tmp_path, method):
    from mcp_tools.models.factory.target_transform import fit_target_transform, transform_targets
    # Fitted statistics come only from these synthetic training observations.
    transform = fit_target_transform([10., 20., 30., 40.], method, label_column="label")
    raw_predictions = [10., 20., 30.]
    model_outputs = transform_targets(raw_predictions, transform).reshape(-1, 1).tolist()
    result, predicted = evaluation.evaluate(TinyModel(1, 1), None, {"mse": MeanSquaredError()},
        loader(model_outputs, [12., 19., 33.]), nn.MSELoss(), problem_type="regression", target_transform=transform)
    assert predicted == pytest.approx(raw_predictions, abs=1e-5)
    assert result["loss"] == pytest.approx([14 / 3], abs=1e-5)
    assert result["mse"] == pytest.approx([14 / 3], abs=1e-5)
    evaluation.save_evaluation_results(result, predicted, pd.DataFrame({"label": [12., 19., 33.]}), tmp_path)
    assert pd.read_csv(tmp_path / "evaluation_result.csv")["pred_label"].tolist() == pytest.approx(raw_predictions, abs=1e-5)


@pytest.mark.parametrize("normalization,space", [("standard", "standardized_target"),
    ("standard", "original_target"), (None, "unknown"), (None, None)])
def test_cli_refuses_unknown_or_unrestored_regression_target_space(normalization, space):
    from types import SimpleNamespace
    config = SimpleNamespace(problem_type="regression", normalize=normalization, target_transform=None,
                             prediction_output_scale={"value_space": space} if space else None)
    with pytest.raises(ValueError, match="requires verified target-transform"):
        evaluation.evaluation_target_transform(config)


def test_cli_accepts_explicit_original_targets_and_checkpoint_verified_transform():
    from types import SimpleNamespace

    from mcp_tools.models.factory.target_transform import fit_target_transform
    config = SimpleNamespace(problem_type="regression", normalize=None, target_transform=None,
                             prediction_output_scale={"value_space": "original_target"})
    assert evaluation.evaluation_target_transform(config) is None
    config.normalize = "standard"
    config.target_transform = fit_target_transform([10., 20.], "standard")
    assert evaluation.evaluation_target_transform(config) == config.target_transform


def test_legacy_evaluate_entrypoint_does_not_silently_compare_normalized_outputs(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(evaluation, "args", SimpleNamespace(problem_type="regression", normalize="standard"), raising=False)
    with pytest.raises(ValueError, match="requires verified target-transform"):
        evaluation.evaluate(TinyModel(1, 1), None, {}, loader([[1.]], [10.]), nn.MSELoss())


@pytest.mark.parametrize("file_format", ["json", "csv"])
@pytest.mark.parametrize("normalization", [None, "standard"])
def test_cli_runs_local_rows_through_torch_model_and_writes_predictions(tmp_path, monkeypatch, file_format, normalization):
    import runpy
    import sys
    from types import ModuleType, SimpleNamespace

    rows = [{"sequence": "A", "measurement": 1.}, {"sequence": "AA", "measurement": 5.},
            {"sequence": "AAA", "measurement": 9.}]
    data = tmp_path / ("input." + file_format)
    if file_format == "json":
        data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        pd.DataFrame(rows).to_csv(data, index=False)

    class SequenceModel(TinyModel):
        def forward(self, plm_model, batch):
            return super().forward(plm_model, {"features": batch["aa_seq_input_ids"].float()})

    def tokenizer(sequences, **kwargs):
        ids = torch.tensor([[len(sequence)] for sequence in sequences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def load_fixture(config):
        from mcp_tools.models.factory.target_transform import fit_target_transform
        config.target_transform = fit_target_transform([-2., 2.], "standard", label_column="measurement") if normalization else None
        config.normalize = normalization
        config.prediction_output_scale = {"value_space": "standardized_target" if normalization else "original_target"}
        return SequenceModel(1, 1, [[1. if normalization else 2.]]), None, tokenizer, torch.device("cpu")

    # Replace only external model/data loaders; collation, model arithmetic,
    # loss, torchmetrics, input CSV parsing and result CSV writes are real.
    module_name = "mcp_tools.models.factory.training.models.inference_loader"
    model_loader = ModuleType(module_name)
    model_loader.load_inference_model = load_fixture
    monkeypatch.setitem(sys.modules, module_name, model_loader)
    dataset_loader = ModuleType("datasets")
    dataset_loader.load_dataset = lambda kind, data_files: {"train": pd.read_csv(data_files).to_dict("records")}
    monkeypatch.setitem(sys.modules, "datasets", dataset_loader)
    transformers = ModuleType("transformers")
    transformers.logging = SimpleNamespace(set_verbosity_error=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    destination = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [evaluation.__file__, "--test_file", str(data),
        "--test_result_dir", str(destination), "--problem_type", "regression", "--num_labels", "1",
        "--sequence_column_name", "sequence", "--label_column_name", "measurement", "--metrics", "mse",
        "--plm_model", "local-fixture", "--batch_size", "2", "--num_workers", "0"])
    runpy.run_path(evaluation.__file__, run_name="__main__")
    saved = pd.read_csv(destination / "evaluation_result.csv")
    metrics = pd.read_csv(destination / "evaluation_metrics.csv")
    assert saved["pred_label"].tolist() == [2., 4., 6.]
    assert saved["measurement"].tolist() == [1., 5., 9.]
    assert metrics["loss"].iloc[0] == pytest.approx(11 / 3)
    assert metrics["mse"].iloc[0] == pytest.approx(11 / 3)
