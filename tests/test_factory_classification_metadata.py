"""Actual loader snapshots and bounded MCP semantics; no neural inference or network."""
import csv
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_tools.models.factory.prediction import fintuned_operations as operations
from mcp_tools.models.factory.prediction.output_scale import output_scale


@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def binary():
    return {"dataset": "tyang816/DeepLocBinary_ESMFold", "problem_type": "single_label_classification", "num_labels": 2}


def fixture(tmp_path, monkeypatch, *, actual=None, first=None, count=1, prediction=None, snapshot=True):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    initial = binary() if first is None else first
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(initial))
    fasta = tmp_path / "input.fasta"
    sequences = [("sequence-" + str(i), "MKTAYIAKQRQISFVK") for i in range(count)]
    fasta.write_text("".join(">" + name + "\n" + sequence + "\n" for name, sequence in sequences))
    inferred = binary() if actual is None else actual
    def load(args):
        if snapshot:
            # A trusted loader records the actual second read, not the caller's first read.
            (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(inferred))
            args._inference_checkpoint_config = json.loads((adapter / "lr5e-4_bt12k_ga8.json").read_text())
        return None, None, None, "cpu"
    predicted = prediction or {"predicted_class": 1, "probabilities": [0.1, 0.9]}
    module = SimpleNamespace(parse_fasta=lambda _: sequences, load_model_and_tokenizer=load, predict=lambda *args: predicted)
    monkeypatch.setattr(operations, "_get_module", lambda _: (module, lambda *args: {"input_ids": [], "attention_mask": []}))
    output = tmp_path / "output.csv"
    result = operations.predict_protein_function.runtime_implementation(
        str(fasta), "Membrane Protein", "ESM2-650M", str(adapter), output_file=str(output))
    return json.loads(result), output, sequences, predicted


@pytest.mark.parametrize("snapshot", [True, False])
def test_tool_returns_actual_config_semantics_or_unknown_not_first_read(tmp_path, monkeypatch, snapshot):
    first = binary() | {"dataset": "unknown/first-read", "num_labels": 9}
    result, output, _, _ = fixture(tmp_path, monkeypatch, first=first, snapshot=snapshot)
    assert result["status"] == "success"
    meta = result["biological_metadata"]
    if snapshot:
        assert meta["output_scale"] == output_scale(binary())
        assert meta["classification"]["checkpoint_config_sha256"] == output_scale(binary())["checkpoint_config_sha256"]
        assert meta["classification"]["decoded"][0]["predicted_labels"] == ["Soluble"]
    else:
        assert meta["output_scale"]["value_space"] == "unknown"
        assert "checkpoint_config_sha256" not in meta["output_scale"]
        assert meta["classification"]["mapping_status"] == "unavailable"
        assert meta["classification"]["reason"] == "inference_config_not_recorded"
    assert result["file_info"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_changed_actual_label_contract_does_not_reuse_valid_first_read(tmp_path, monkeypatch):
    actual = binary() | {"num_labels": 10}
    result, _, _, _ = fixture(tmp_path, monkeypatch, actual=actual)
    meta = result["biological_metadata"]
    assert meta["output_scale"] == output_scale(actual)
    assert meta["classification"]["mapping_status"] == "unavailable"
    assert "decoded" not in meta["classification"]


def test_missing_actual_file_is_not_replaced_by_first_read_config(tmp_path, monkeypatch):
    result, _, _, _ = fixture(tmp_path, monkeypatch, actual={})
    meta = result["biological_metadata"]
    assert meta["output_scale"] == output_scale({})
    assert meta["classification"]["mapping_status"] == "unavailable"


def test_multisequence_preview_is_bounded_and_row_bound_without_changing_csv(tmp_path, monkeypatch):
    result, path, sequences, prediction = fixture(tmp_path, monkeypatch, count=200)
    classification = result["biological_metadata"]["classification"]
    assert classification["preview"] == {"total_rows": 200, "decoded_rows": 8, "limit": 8, "truncated": True, "row_index_base": 0}
    for index, row in enumerate(classification["decoded"]):
        assert row["row_index"] == index
        assert row["header_sha256"] == hashlib.sha256(sequences[index][0].encode()).hexdigest()
        assert "header" not in row and "sequence" not in row
    assert len(json.dumps(classification).encode()) < 8_000
    assert len(result["content_preview"]) <= 500
    expected = io.StringIO(newline="")
    writer = csv.DictWriter(expected, fieldnames=["header", "sequence", *prediction])
    writer.writeheader()
    for header, sequence in sequences:
        writer.writerow({"header": header, "sequence": sequence,
                         **{key: json.dumps(value) if isinstance(value, list) else value for key, value in prediction.items()}})
    assert path.read_bytes() == expected.getvalue().encode()
    assert result["file_info"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sorting_index_six_and_regression_keep_their_distinct_spaces(tmp_path, monkeypatch):
    cfg = {"dataset": "tyang816/SortingSignal", "problem_type": "multi_label_classification", "num_labels": 9}
    prediction = {"predicted_class": [0, 0, 0, 0, 0, 0, 1, 0, 0], "probabilities": [0.1] * 6 + [0.9, 0.1, 0.1]}
    result, _, _, _ = fixture(tmp_path, monkeypatch, actual=cfg, prediction=prediction)
    classification = result["biological_metadata"]["classification"]
    assert classification["decoded"][0]["predicted_meanings"] == ["Signal peptide"]
    args = SimpleNamespace(_inference_checkpoint_config={"problem_type": "regression", "normalize": "standard"})
    scale, semantics = operations._prediction_interpretation(args, [{"prediction": -0.97091}])
    assert scale["value_space"] == "standardized_target" and scale["unit"] == "dimensionless"
    assert semantics["mapping_status"] == "not_applicable" and "decoded" not in semantics


@pytest.mark.parametrize("name,tokenizer,plm", [
    ("esm2", "EsmTokenizer", "EsmModel"), ("ankh", "AutoTokenizer", "T5EncoderModel"),
    ("protbert", "BertTokenizer", "BertModel"), ("prott5", "T5Tokenizer", "T5EncoderModel"),
])
@pytest.mark.parametrize("missing", [False, True])
def test_four_real_loaders_record_independent_config_at_actual_read(tmp_path, monkeypatch, name, tokenizer, plm, missing):
    module = importlib.import_module("mcp_tools.models.factory.prediction." + name)
    actual = binary() | {"nested": {"items": [1]}}
    if not missing:
        (tmp_path / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(actual))
    class FakeModel:
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, state): pass
    monkeypatch.setattr(module, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda device: device, load=lambda *args, **kwargs: {}))
    monkeypatch.setattr(module, tokenizer, SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr(module, plm, SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()))
    monkeypatch.setattr(module, "AdapterModel", lambda args: FakeModel())
    args = SimpleNamespace(adapter_path=str(tmp_path), _inference_checkpoint_config={"stale": True})
    module.load_model_and_tokenizer(args)
    assert args._inference_checkpoint_config == ({} if missing else actual)
    if not missing:
        assert not hasattr(args, "nested")
        args.problem_type = "runtime-only-change"
        assert args._inference_checkpoint_config == actual


def test_finetuned_subprocess_roundtrip_retains_bounded_metadata(tmp_path):
    """Actual JSON request/result subprocess path, with an in-child neural double."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(binary()))
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">query\nMKTAYIAKQRQISFVK\n")
    output = tmp_path / "output.csv"
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({"operation": "predict_protein_function", "arguments": {
        "fasta_file": str(fasta), "task": "Membrane Protein", "model_name": "ESM2-650M",
        "adapter_path": str(adapter), "output_file": str(output)}}))
    code = r'''import json,sys
from pathlib import Path
from types import SimpleNamespace
from mcp_tools.models.factory.prediction import fintuned_operations as operations
from mcp_tools.models.factory.runners.finetuned import main
def load(args):
    args._inference_checkpoint_config=json.loads((Path(args.adapter_path)/"lr5e-4_bt12k_ga8.json").read_text())
    return None,None,None,"cpu"
module=SimpleNamespace(parse_fasta=lambda path:[("query","MKTAYIAKQRQISFVK")],load_model_and_tokenizer=load,predict=lambda *args:{"predicted_class":1,"probabilities":[0.1,0.9]})
operations._get_module=lambda key:(module,lambda *args:{"input_ids":[],"attention_mask":[]})
sys.argv=["finetuned",sys.argv[1],sys.argv[2]]
main()
'''
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
           "PYTHONPATH": os.pathsep.join(str(Path(p).absolute()) for p in sys.path if p),
           "VENUS_RUNTIME_ROOT": str(tmp_path / "child-runtime"), "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(request), str(result)], env=env,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    value = json.loads(json.loads(result.read_text()))
    assert value["status"] == "success"
    assert value["biological_metadata"]["classification"]["decoded"][0]["predicted_labels"] == ["Soluble"]
    assert value["biological_metadata"]["classification"]["preview"]["total_rows"] == 1
    assert value["file_info"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result.stat().st_size < 12_000


def test_largest_verified_multilabel_preview_does_not_expand_headers_or_full_batch():
    cfg = {"dataset": "tyang816/SortingSignal", "problem_type": "multi_label_classification", "num_labels": 9}
    row = {"header": "long-header" * 10000, "predicted_class": [1] * 9, "probabilities": [0.9123456789012345] * 9}
    _, metadata = operations._prediction_interpretation(SimpleNamespace(_inference_checkpoint_config=cfg), [row] * 200)
    assert len(metadata["decoded"]) == 8
    assert len(json.dumps(metadata).encode()) < 12_000
    assert "long-header" not in json.dumps(metadata)
    assert metadata["preview"]["total_rows"] == 200 and metadata["preview"]["truncated"]
