"""Checkpoint settings cannot redirect a host-selected factory prediction."""

import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mcp_tools.models.factory.prediction.checkpoint_config import apply_checkpoint_config
from mcp_tools.models.factory.prediction.output_scale import output_scale


def settings():
    return {
        "hidden_size": 1280, "num_attention_head": 8,
        "attention_probs_dropout": 0.0, "num_labels": 2,
        "plm_model": "facebook/esm2_t33_650M_UR50D",
        "pooling_dropout": 0.1, "pooling_method": "mean",
        "problem_type": "single_label_classification", "structure_seq": "",
        "training_method": "freeze", "vocab_size": 33, "dataset": "DeepSol",
        "normalize": "standard", "label_names": ["insoluble", "soluble"],
        "target_unit": "unknown", "future_metadata": {"version": 1},
    }


def test_model_settings_and_full_semantics_snapshot_do_not_replace_host_fields():
    config = settings()
    config.update({
        "adapter_path": "/other-session/adapter", "output_csv": "/other/output.csv",
        "fasta_file": "/other/input.fasta", "output_file": "/other/result.csv",
        "out_dir": "/other", "ckpt_base": "/other/checkpoints",
        "task": "wrong-task", "model_name": "wrong-backbone", "device": "cuda:99",
        "_inference_checkpoint_config": {"dataset": "forged"},
        "__dict__": {"adapter_path": "/other"}, "__class__": "not-a-class",
    })
    host = {
        "adapter_path": "/selected/adapter", "output_csv": "/session/result.csv",
        "fasta_file": "/session/input.fasta", "output_file": "/session/result.csv",
        "out_dir": "/session", "ckpt_base": "/installed/checkpoints",
        "task": "Solubility", "model_name": "ESM2-650M", "device": "cpu",
    }
    args = Namespace(**host)
    apply_checkpoint_config(args, config)
    assert all(getattr(args, key) == value for key, value in host.items())
    assert args.hidden_size == 1280 and args.dataset == "DeepSol"
    assert args.problem_type == "single_label_classification"
    assert args._inference_checkpoint_config == config
    assert "future_metadata" not in vars(args)
    assert "label_names" not in vars(args)
    assert output_scale(args._inference_checkpoint_config) == output_scale(config)
    config["label_names"].append("changed later")
    assert args._inference_checkpoint_config["label_names"] == ["insoluble", "soluble"]


def test_allowed_mutable_field_is_copied_only_as_namespace_isolation():
    # Synthetic transport test, not a claim that this shape is a valid model.
    config = {"structure_seq": ["sentinel"]}
    args = Namespace(adapter_path="selected")
    apply_checkpoint_config(args, config)
    args.structure_seq.append("runtime edit")
    assert config == {"structure_seq": ["sentinel"]}
    assert args._inference_checkpoint_config == config
    config["structure_seq"].append("source edit")
    assert args._inference_checkpoint_config == {"structure_seq": ["sentinel"]}


@pytest.mark.parametrize("stale", [False, True])
def test_preliminary_projection_is_not_actual_loader_evidence(stale):
    args = Namespace(adapter_path="selected", output_csv="result.csv")
    if stale:
        args._inference_checkpoint_config = {"dataset": "previous model"}
    preliminary = settings() | {"adapter_path": "other", "dataset": "DeepSol"}
    apply_checkpoint_config(args, preliminary, record_snapshot=False)
    assert args.adapter_path == "selected" and args.output_csv == "result.csv"
    assert args.dataset == "DeepSol" and args.hidden_size == 1280
    assert not hasattr(args, "_inference_checkpoint_config")
    actual = settings() | {"dataset": "ProtSolM", "normalize": None}
    apply_checkpoint_config(args, actual)
    assert args._inference_checkpoint_config == actual
    assert args._inference_checkpoint_config != preliminary


@pytest.mark.parametrize("invalid", [None, [], "config", {1: "invalid key"}])
def test_invalid_config_cannot_partially_mutate_namespace(invalid):
    args = Namespace(adapter_path="selected", hidden_size=7)
    before = vars(args).copy()
    with pytest.raises(ValueError, match="JSON object"):
        apply_checkpoint_config(args, invalid)
    assert vars(args) == before


def load_actual_loader(name, monkeypatch, observed):
    """Import production loader with only expensive dependencies substituted."""
    torch = ModuleType("torch")
    torch.device = lambda value: value
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    def weights(path, **kwargs):
        observed["weight_path"] = str(path)
        return {"loaded": True}

    torch.load = weights
    monkeypatch.setitem(sys.modules, "torch", torch)
    transformers = ModuleType("transformers")

    class Backbone:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            observed.setdefault("backbones", []).append(name)
            return cls()

        def to(self, device):
            return self

    for exported in ("EsmModel", "EsmTokenizer", "T5EncoderModel", "AutoTokenizer",
                     "BertModel", "BertTokenizer", "T5Tokenizer"):
        setattr(transformers, exported, Backbone)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model_module = ModuleType("mcp_tools.models.factory.training.models.adapter_model")

    class Adapter:
        def __init__(self, args):
            observed["args_at_construct"] = copy.deepcopy(vars(args))

        def load_state_dict(self, state):
            assert state == {"loaded": True}

        def to(self, device):
            return self

        def eval(self):
            return self

    model_module.AdapterModel = Adapter
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    root = Path(__file__).resolve().parents[1]
    path = root / "mcp_tools/models/factory/prediction" / (name + ".py")
    spec = importlib.util.spec_from_file_location("factory_checkpoint_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("loader", ["esm2", "ankh", "protbert", "prott5"])
def test_actual_loader_keeps_selected_weights_input_and_output(tmp_path, monkeypatch, loader):
    adapter = tmp_path / "selected"
    adapter.mkdir()
    foreign = tmp_path / "other-owner" / "adapter"
    config = settings()
    config.update(adapter_path=str(foreign), output_csv=str(tmp_path / "escaped.csv"),
                  fasta_file=str(tmp_path / "escaped.fasta"),
                  _inference_checkpoint_config={"dataset": "forged"})
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
    fasta = tmp_path / "input.fasta"
    output = tmp_path / "result.csv"
    args = Namespace(adapter_path=str(adapter), fasta_file=str(fasta), output_csv=str(output))
    observed = {}
    module = load_actual_loader(loader, monkeypatch, observed)
    module.load_model_and_tokenizer(args)
    assert observed["weight_path"] == str(adapter / "lr5e-4_bt12k_ga8.pt")
    assert args.adapter_path == str(adapter)
    assert args.fasta_file == str(fasta) and args.output_csv == str(output)
    assert observed["args_at_construct"]["adapter_path"] == str(adapter)
    assert args._inference_checkpoint_config == config
    assert args.dataset == "DeepSol" and args.num_labels == 2
    assert output_scale(args._inference_checkpoint_config) == output_scale(config)
    assert not foreign.exists() and not output.exists()


@pytest.mark.parametrize("loader", ["esm2", "ankh", "protbert", "prott5"])
def test_actual_loader_missing_config_does_not_invent_snapshot(tmp_path, monkeypatch, loader):
    adapter = tmp_path / "selected"
    adapter.mkdir()
    args = Namespace(adapter_path=str(adapter), output_csv=str(tmp_path / "output.csv"),
                     _inference_checkpoint_config={"dataset": "old"})
    observed = {}
    module = load_actual_loader(loader, monkeypatch, observed)
    module.load_model_and_tokenizer(args)
    assert args._inference_checkpoint_config == {}
    assert observed["weight_path"] == str(adapter / "lr5e-4_bt12k_ga8.pt")
    assert not hasattr(args, "dataset")
