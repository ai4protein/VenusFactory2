"""Real CPU dataloaders/checkpoint writes/prediction, with no downloaded model."""
import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mcp_tools.models.factory.target_transform import (
    SCHEMA,
    checkpoint_transform_path,
    fit_target_transform,
    load_checkpoint_transform,
    save_checkpoint_transform,
)


def training_args(**overrides):
    values = {"sequence_column_name": "aa_seq", "label_column_name": "label", "quick_test": False,
              "normalize": "standard", "problem_type": "regression", "num_labels": 1,
              "target_unit": "nM", "seed": 42, "dataset": "synthetic", "max_seq_len": 32,
              "structure_seq": "", "plm_model": "synthetic", "num_workers": 0,
              "batch_token": None, "batch_size": 2}
    values.update(overrides)
    return SimpleNamespace(**values)


def prepare(monkeypatch, args):
    from mcp_tools.models.factory.training.data import dataloader
    data = {"train": [{"aa_seq": "AA", "label": 10.}, {"aa_seq": "CC", "label": 14.}],
            "validation": [{"aa_seq": "DD", "label": 1000.}],
            "test": [{"aa_seq": "EE", "label": -1000.}]}
    original = copy.deepcopy(data)
    monkeypatch.setattr(dataloader, "_load_dataset_dict", lambda _: data)
    monkeypatch.setattr(dataloader, "ensure_structure_columns", lambda data, *_: data)
    monkeypatch.setattr(dataloader, "Collator", lambda **_: lambda batch: batch)
    loaders = dataloader.prepare_dataloaders(args, None, SimpleNamespace(info=lambda _: None))
    assert data == original
    return loaders


def test_real_dataloaders_fit_once_before_holdout_transform(monkeypatch):
    args = training_args()
    train, val, test = prepare(monkeypatch, args)
    assert args.target_transform["parameters"] == {"mean": 12., "scale": 2., "ddof": 0}
    assert args.target_transform_schema == SCHEMA
    assert [r["label"] for r in train.dataset] == [-1., 1.]
    assert val.dataset[0]["label"] == 494.
    assert test.dataset[0]["label"] == -506.


def test_identity_training_records_known_raw_target_units(monkeypatch):
    args = training_args(normalize=None)
    train, _, _ = prepare(monkeypatch, args)
    assert args.target_transform["method"] == "none"
    assert args.target_transform["target_unit"] == "nM"
    assert [r["label"] for r in train.dataset] == [10., 14.]


@pytest.mark.parametrize("problem", ["single_label_classification", "multi_label_classification"])
def test_classification_does_not_normalize_even_with_legacy_option(monkeypatch, problem):
    args = training_args(problem_type=problem, num_labels=2,
                         target_transform={"stale": True}, target_transform_schema="stale")
    train, val, _ = prepare(monkeypatch, args)
    assert [r["label"] for r in train.dataset] == [10., 14.]
    assert val.dataset[0]["label"] == 1000.
    assert args.target_transform is None and args.target_transform_schema is None


class TinyHead(torch.nn.Module):
    def __init__(self, value=0.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))

    def forward(self, _plm, _data):
        return self.weight.reshape(1, 1)


class TinyPLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("constant", torch.tensor([1.]))
        self.config = SimpleNamespace(hidden_size=1, vocab_size=4)

    def save_pretrained(self, path):
        from pathlib import Path
        Path(path).mkdir()
        torch.save(self.state_dict(), Path(path) / "adapter_model.bin")


def state_config():
    state = fit_target_transform([10., 14.], "standard", target_unit="nM")
    config = {"problem_type": "regression", "num_labels": 1, "normalize": "standard", "target_unit": "nM",
              "target_transform_schema": SCHEMA, "target_transform": state, "label_column_name": "label"}
    return state, config


@pytest.mark.parametrize("method", ["freeze", "full", "plm-lora", "plm-qlora", "plm-dora", "plm-adalora", "plm-ia3"])
def test_trainer_real_checkpoint_save_publishes_verified_state(tmp_path, method):
    from mcp_tools.models.factory.training.loop.trainer import Trainer
    state, config = state_config()
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(**config, training_method=method)
    trainer.model, trainer.plm_model = TinyHead(2.), TinyPLM()
    path = tmp_path / "model.pt"
    trainer._save_model(str(path))
    assert load_checkpoint_transform(path, config) == state
    assert json.loads(path.with_suffix(".json").read_text())["target_transform"] == state
    weights = torch.load(path, weights_only=True)
    actual = weights["model_state_dict"] if method == "full" else weights
    assert actual["weight"].item() == 2.
    # Changing only checkpoint weights invalidates even a perfectly valid state.
    torch.save({"weight": torch.tensor([8.])}, path)
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint_transform(path, config)


def test_legacy_save_does_not_invent_transform(tmp_path):
    from mcp_tools.models.factory.training.loop.trainer import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(training_method="freeze", problem_type="regression", normalize="standard")
    trainer.model = TinyHead(2.)
    path = tmp_path / "legacy.pt"
    trainer._save_model(str(path))
    assert path.exists() and not checkpoint_transform_path(path).exists()


def loader_fixture(tmp_path, monkeypatch, *, metadata=True, fallback=False):
    from mcp_tools.models.factory.training.models import inference_loader as loader
    state, config = state_config()
    path = tmp_path / "model.pt"
    torch.save(TinyHead(2.).state_dict(), path)
    if metadata:
        config_path = tmp_path / ("training_config.json" if fallback else "model.json")
        config_path.write_text(json.dumps(config))
        save_checkpoint_transform(path, state, config=config)
    args = SimpleNamespace(model_path=str(path), plm_model="synthetic", eval_method="freeze",
                           problem_type="regression", num_labels=1, structure_seq="",
                           _checkpoint_config={"normalize": "standard", "stale": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(loader, "create_plm_and_tokenizer", lambda _: (None, TinyPLM()))
    monkeypatch.setattr(loader, "get_hidden_size", lambda *_: 1)
    monkeypatch.setattr(loader, "AdapterModel", lambda _: TinyHead())
    return loader, path, args, state, config


@pytest.mark.parametrize("fallback", [False, True])
def test_loader_verifies_state_before_predictions_and_accepts_registry_config(tmp_path, monkeypatch, fallback):
    loader, _, args, state, config = loader_fixture(tmp_path, monkeypatch, fallback=fallback)
    head, _, _, device = loader.load_inference_model(args, verbose=False)
    assert str(device) == "cpu" and head.weight.item() == 2.
    assert args._checkpoint_config == config
    assert args.target_transform == state
    assert args.prediction_output_scale["inverse_transform_available"] is True


def test_loader_corruption_rejects_before_plm_allocation(tmp_path, monkeypatch):
    loader, path, args, _, _ = loader_fixture(tmp_path, monkeypatch)
    path.write_bytes(b"corrupted checkpoint")
    def forbidden(_):
        pytest.fail("A PLM must not be allocated before verifying target state")
    monkeypatch.setattr(loader, "create_plm_and_tokenizer", forbidden)
    with pytest.raises(ValueError, match="hash"):
        loader.load_inference_model(args, verbose=False)


def test_loader_does_not_use_stale_cli_metadata_for_missing_config(tmp_path, monkeypatch):
    loader, _, args, _, _ = loader_fixture(tmp_path, monkeypatch, metadata=False)
    loader.load_inference_model(args, verbose=False)
    assert args._checkpoint_config == {}
    assert args.target_transform is None
    assert args.prediction_output_scale["value_space"] == "unknown"


def test_single_and_batch_prediction_restore_identical_units(tmp_path, monkeypatch):
    from src import predict
    loader, _, args, _, _ = loader_fixture(tmp_path, monkeypatch)
    head, plm, tokenizer, device = loader.load_inference_model(args, verbose=False)
    single = predict.predict(head, {}, device, args, plm, verbose=False)
    assert single["predictions"] == 16.
    assert single["raw_predictions"] == 2.
    assert single["output_scale"]["unit"] == "nM"
    assert single["output_scale"]["inverse_transform_applied"] is True
    inputs = tmp_path / "inputs.csv"
    inputs.write_text("id,aa_seq\na,AAAA\nb,CCCC\n")
    args.input_file, args.output_dir, args.output_file = str(inputs), str(tmp_path), "results.csv"
    args.use_foldseek = args.use_ss8 = False
    monkeypatch.setattr(predict, "process_sequence", lambda *_: {})
    assert predict._run_batch(args, head, plm, tokenizer, device) == 0
    result = pd.read_csv(tmp_path / "results.csv")
    assert result["prediction"].tolist() == [16., 16.]
    assert result["prediction_model_space"].tolist() == [2., 2.]
    assert result["prediction_value_space"].tolist() == ["original_target"] * 2
    assert result["prediction_unit"].tolist() == ["nM"] * 2
    assert json.loads((tmp_path / "results.csv.output-scale.json").read_text()) == single["output_scale"]


def test_legacy_prediction_retains_unconverted_score_and_unavailable_inverse():
    from src.predict import predict
    args = SimpleNamespace(plm_model="synthetic", problem_type="regression", target_transform=None,
                           _checkpoint_config={"problem_type": "regression", "normalize": "standard"})
    result = predict(TinyHead(.5), {}, "cpu", args, None, verbose=False)
    assert result["predictions"] == .5 and "raw_predictions" not in result
    assert result["output_scale"]["value_space"] == "standardized_target"
    assert result["output_scale"]["inverse_transform_available"] is False


def test_classification_prediction_unchanged():
    from src.predict import predict
    logits = torch.tensor([[.2, .8]])
    args = SimpleNamespace(plm_model="synthetic", problem_type="single_label_classification")
    result = predict(lambda *_: logits, {}, "cpu", args, None, verbose=False)
    assert result["predicted_classes"] == [1]
    np.testing.assert_array_equal(result["probabilities"], torch.softmax(logits, dim=1).tolist())
    assert set(result) == {"predicted_classes", "probabilities"}


def test_explicit_target_unit_is_cli_and_dataset_configurable(tmp_path):
    from mcp_tools.models.factory.training.cli_utils.args import add_dataset_args, process_dataset_config
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    config = tmp_path / "data.json"
    config.write_text(json.dumps({"target_unit": "nM", "normalize": "standard"}))
    args = parser.parse_args(["--dataset_config", str(config)])
    process_dataset_config(args)
    assert args.target_unit == "nM" and args.normalize == "standard"
    args = parser.parse_args(["--dataset_config", str(config), "--target_unit", "pKd"])
    process_dataset_config(args)
    assert args.target_unit == "pKd"


def test_train_split_to_alternate_checkpoint_to_restored_prediction(tmp_path, monkeypatch):
    from src.predict import predict

    from mcp_tools.models.factory.training.loop.trainer import Trainer
    from mcp_tools.models.factory.training.models import inference_loader as loader
    args = training_args(training_method="freeze", output_model_name="best.pt")
    prepare(monkeypatch, args)
    trainer = Trainer.__new__(Trainer)
    trainer.args, trainer.model, trainer.plm_model = args, TinyHead(0.), TinyPLM()
    alternate = tmp_path / "last_model.pt"
    trainer._save_model(str(alternate))
    inference = SimpleNamespace(model_path=str(alternate), plm_model="synthetic", eval_method="freeze",
                                problem_type="single_label_classification", num_labels=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(loader, "create_plm_and_tokenizer", lambda _: (None, TinyPLM()))
    monkeypatch.setattr(loader, "get_hidden_size", lambda *_: 1)
    monkeypatch.setattr(loader, "AdapterModel", lambda _: TinyHead())
    head, plm, _, device = loader.load_inference_model(inference, verbose=False)
    result = predict(head, {}, device, inference, plm, verbose=False)
    assert result["predictions"] == 12. and result["raw_predictions"] == 0.
    assert result["output_scale"]["unit"] == "nM"
    assert inference.target_transform["training_count"] == 2


def test_new_classification_overwrite_removes_only_its_stale_transform(tmp_path):
    from mcp_tools.models.factory.target_transform import target_output_scale
    from mcp_tools.models.factory.training.loop.trainer import Trainer
    state, config = state_config()
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(**config, training_method="freeze")
    trainer.model = TinyHead(2.)
    path = tmp_path / "model.pt"
    untouched = tmp_path / "other.pt"
    trainer._save_model(str(path))
    trainer._save_model(str(untouched))
    other_sidecar = checkpoint_transform_path(untouched).read_bytes()
    trainer.args = SimpleNamespace(training_method="freeze", problem_type="single_label_classification",
                                   num_labels=2, normalize="standard", target_transform=None,
                                   target_transform_schema=None)
    trainer.model = TinyHead(7.)
    trainer._save_model(str(path))
    assert torch.load(path, weights_only=True)["weight"].item() == 7.
    assert not checkpoint_transform_path(path).exists()
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["normalize"] is None and saved["target_transform_schema"] is None
    assert load_checkpoint_transform(path, saved) is None
    assert target_output_scale(saved)["normalization"] == "not_applicable"
    assert checkpoint_transform_path(untouched).read_bytes() == other_sidecar
    assert load_checkpoint_transform(untouched, config) == state


def test_failed_classification_weight_write_preserves_previous_transform(tmp_path, monkeypatch):
    from mcp_tools.models.factory.training.loop.trainer import Trainer
    state, config = state_config()
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(**config, training_method="freeze")
    trainer.model = TinyHead(2.)
    path = tmp_path / "model.pt"
    trainer._save_model(str(path))
    before = {file: file.read_bytes() for file in tmp_path.iterdir()}
    trainer.args = SimpleNamespace(training_method="freeze", problem_type="single_label_classification",
                                   num_labels=2, normalize="standard", target_transform=None,
                                   target_transform_schema=None)
    def failed_write(*_):
        raise OSError("disk write interrupted")
    monkeypatch.setattr(torch, "save", failed_write)
    with pytest.raises(OSError, match="interrupted"):
        trainer._save_model(str(path))
    assert {file: file.read_bytes() for file in tmp_path.iterdir()} == before
    assert load_checkpoint_transform(path, config) == state
