"""Dataset presets and CLI overrides must produce usable training arguments."""

import json
import sys
import warnings
from pathlib import Path

import pytest

from mcp_tools.models.factory.training.cli_utils import args as training_args


def parse(monkeypatch, tmp_path, config=None, options=()):
    argv = ["train", "--batch_size", "2", "--output_dir", str(tmp_path / "output")]
    if config is not None:
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(config))
        argv.extend(["--dataset_config", str(path)])
    monkeypatch.setattr(sys, "argv", [*argv, *options])
    return training_args.parse_args()


@pytest.mark.parametrize("metrics", ["accuracy,mcc", ["accuracy", "mcc"]])
@pytest.mark.parametrize("override", [False, True])
def test_dataset_metrics_and_explicit_cli_override(monkeypatch, tmp_path, metrics, override):
    options = ["--metrics", "mse,pearson"] if override else []
    args = parse(monkeypatch, tmp_path, {"dataset": "public/example", "metrics": metrics}, options)
    assert args.metrics == (["mse", "pearson"] if override else ["accuracy", "mcc"])
    assert args.dataset == "public/example"
    assert Path(args.output_dir).is_dir()


@pytest.mark.parametrize("structure", ["foldseek_seq,ss8_seq", ["foldseek_seq", "ss8_seq"]])
def test_ses_structure_config_reaches_validation_once(monkeypatch, tmp_path, structure):
    args = parse(
        monkeypatch, tmp_path, {"structure_seq": structure, "metrics": "mcc"}, ["--training_method", "ses-adapter"]
    )
    assert args.structure_seq == ["foldseek_seq", "ss8_seq"]
    assert args.metrics == ["mcc"]


def test_preserve_dataset_column_and_structure_precedence(monkeypatch, tmp_path):
    config = {
        "dataset": "public/preset",
        "metrics": "mcc",
        "structure_seq": ["ss8_seq"],
        "sequence_column_name": "sequence",
        "label_column_name": "activity",
        "pdb_dir": "preset-structures",
        "structure_merge_key": "uid",
        "structure_merge_key_format": "{interpro_id}-{uid}",
    }
    options = [
        "--dataset",
        "public/explicit",
        "--pdb_dir",
        "explicit-structures",
        "--metrics",
        "mse",
        "--training_method",
        "ses-adapter",
        "--structure_seq",
        "foldseek_seq",
        "--sequence_column_name",
        "aa_seq",
    ]
    args = parse(monkeypatch, tmp_path, config, options)
    assert args.dataset == "public/explicit"
    assert args.pdb_dir == "explicit-structures"
    assert args.metrics == ["mse"]
    assert args.structure_seq == ["ss8_seq"]
    assert (args.sequence_column_name, args.label_column_name) == ("sequence", "activity")
    assert args._column_override_info["sequence_column_name"] == {"old": "aa_seq", "new": "sequence"}
    assert args.structure_merge_key == "uid"
    assert args.structure_merge_key_format == "{interpro_id}-{uid}"


def test_direct_cli_remains_supported(monkeypatch, tmp_path):
    args = parse(
        monkeypatch,
        tmp_path,
        options=["--metrics", "mcc,mse", "--training_method", "ses-adapter", "--structure_seq", "foldseek_seq"],
    )
    assert args.metrics == ["mcc", "mse"]
    assert args.structure_seq == ["foldseek_seq"]


@pytest.mark.parametrize("metrics", ["None", ["None"]])
@pytest.mark.parametrize("override", [False, True])
def test_loss_only_sentinel_emits_one_warning(monkeypatch, tmp_path, metrics, override):
    options = ["--metrics", "None"] if override else []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = parse(monkeypatch, tmp_path, {"metrics": metrics}, options)
    assert args.metrics == ["loss"]
    notices = [item for item in caught if "using default metrics: loss" in str(item.message)]
    assert len(notices) == 1


@pytest.mark.parametrize("structure", [None, "", " , ", []])
def test_ses_requires_nonempty_structure_channels(monkeypatch, tmp_path, structure):
    with pytest.raises(ValueError, match="structure_seq"):
        parse(monkeypatch, tmp_path, {"structure_seq": structure}, ["--training_method", "ses-adapter"])


@pytest.mark.parametrize(
    "field,value",
    [("metrics", {"mcc": 1}), ("metrics", [1]), ("structure_seq", {"ss8_seq": True}), ("structure_seq", [1])],
)
def test_invalid_list_fields_have_a_configuration_error(monkeypatch, tmp_path, field, value):
    options = ["--training_method", "ses-adapter"] if field == "structure_seq" else []
    with pytest.raises(ValueError, match=field):
        parse(monkeypatch, tmp_path, {field: value}, options)


def test_list_fields_trim_whitespace_without_changing_order(monkeypatch, tmp_path):
    args = parse(
        monkeypatch,
        tmp_path,
        {"metrics": [" mse ", "", "pearson"], "structure_seq": " ss8_seq , foldseek_seq , "},
        ["--training_method", "ses-adapter"],
    )
    assert args.metrics == ["mse", "pearson"]
    assert args.structure_seq == ["ss8_seq", "foldseek_seq"]


def test_non_ses_training_ignores_structure_channels(monkeypatch, tmp_path):
    args = parse(monkeypatch, tmp_path, {"structure_seq": ["ss8_seq"], "metrics": "mcc"})
    assert args.structure_seq == []
