"""Label semantics use audited dataset conventions, never guessed class meanings.

Small output doubles exercise real CSV/report/receipt code; no biological inference.
"""
import copy
import csv
import hashlib
import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from mcp_tools.models.factory.prediction.label_schema import (
    checkpoint_label_schema,
    decode_values,
    schema_for_config,
)
from mcp_tools.models.factory.prediction.output_scale import output_scale


@pytest.fixture(autouse=True)
def private_roots(tmp_path, monkeypatch):
    for name, part in [("VENUS_RUNTIME_ROOT", "runtime"), ("VENUS_SANDBOX_CONTROL_ROOT", "sandboxes"),
                       ("VENUS_JOB_QUEUE_ROOT", "queue"), ("VENUS_ACCESS_DB", "access.sqlite3")]:
        monkeypatch.setenv(name, str(tmp_path / part))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def config(dataset="DeepLocBinary_ESMFold", labels=2, problem="single_label_classification"):
    return {"dataset": "tyang816/" + dataset, "num_labels": labels, "problem_type": problem}


def decoded(config_value, predictions, probabilities):
    values = [{"predicted_class": json.dumps(predictions), "probabilities": json.dumps(probabilities)}]
    before = copy.deepcopy(values)
    result = decode_values(values, schema_for_config(config_value), output_scale(config_value))
    assert values == before  # Interpretation never edits the original model outputs.
    return result


@pytest.mark.parametrize("namespace", ["tyang816", "AI4Protein"])
def test_binary_soluble_is_class_one_with_audited_not_checkpoint_revision(namespace):
    cfg = config() | {"dataset": namespace + "/DeepLocBinary_ESMFold"}
    result = decoded(cfg, 1, [0.002, 0.998])
    assert result["class_order"] == ["Membrane", "Soluble"]
    assert result["decoded"][0]["predicted_labels"] == ["Soluble"]
    assert result["provenance"]["revision"] == "e300ff0b3b09b2ce023d5d35efaeb187f657db7e"
    assert result["provenance"]["checkpoint_revision_recorded"] is False
    assert result["mapping_status"] == "legacy_dataset_convention"
    assert result["probability_calibration"] == "not_established"


@pytest.mark.parametrize("index,label", list(enumerate([
    "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane",
    "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome",
])))
def test_full_deeploc_order_matches_author_annotations_in_both_consumers(index, label):
    from mcp_tools.common.data_processors import map_prediction_labels
    from mcp_tools.common.label_mappers import map_labels, map_labels_individual

    cfg = config("DeepLocMulti_ESMFold", 10)
    probabilities = [0.01] * 10
    probabilities[index] = 0.91
    result = decoded(cfg, index, probabilities)
    assert result["decoded"][0]["predicted_labels"] == [label]
    row = pd.Series({"Dataset": "DeepLocMulti", "predicted_class": str(index)})
    for mapper in [map_prediction_labels, map_labels, map_labels_individual]:
        assert mapper(row, "Subcellular Localization") == label


def test_sorting_signal_nine_positions_and_threshold_match_the_actual_predictor():
    from mcp_tools.common.data_processors import map_prediction_labels
    from mcp_tools.common.label_mappers import map_labels, map_labels_individual

    cfg = config("SortingSignal", 9, "multi_label_classification")
    probabilities = [0.1] * 9
    probabilities[5], probabilities[6] = 0.5, 0.9
    prediction = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    result = decoded(cfg, prediction, probabilities)
    assert result["class_order"] == ["CH", "GPI", "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]
    assert result["decoded"][0]["predicted_labels"] == ["SP"]
    assert result["decoded"][0]["predicted_meanings"] == ["Signal peptide"]
    assert result["decoded"][0]["decision_threshold"] == {"operator": ">", "value": 0.5}
    row = pd.Series({"Dataset": "SortingSignal", "predicted_class": json.dumps(prediction)})
    for mapper in [map_prediction_labels, map_labels, map_labels_individual]:
        assert mapper(row, "Sortingsignal") == "SP"
    empty = decoded(cfg, [0] * 9, [0.5] * 9)["decoded"][0]
    assert empty["predicted_labels"] == [] and "No listed signal" in empty["interpretation"]
    assert "No signal" not in result["class_order"]


@pytest.mark.parametrize("change", [
    {"num_labels": 10}, {"num_labels": "2"}, {"num_labels": True},
    {"problem_type": "multi_label_classification"},
    {"dataset": "other/DeepLocBinary_ESMFold"}, {"dataset": "DeepLocBinary"}, {"dataset": None},
])
def test_unverified_or_conflicting_config_never_infers_labels_from_task_or_name(change):
    result = schema_for_config(config() | change | {"task": "Membrane Protein"})
    assert result["mapping_status"] == "unavailable"
    assert "class_order" not in result and "decoded" not in result


def test_missing_config_and_regression_never_get_class_labels(tmp_path):
    assert checkpoint_label_schema(tmp_path)["mapping_status"] == "unavailable"
    cfg = {"dataset": "DeepET_Topt", "problem_type": "regression", "normalize": "standard"}
    values = [{"prediction": -0.97091}]
    scale = output_scale(cfg)
    result = decode_values(values, schema_for_config(cfg), scale)
    assert result["mapping_status"] == "not_applicable"
    assert values == [{"prediction": -0.97091}] and scale == output_scale(cfg)
    assert scale["unit"] == "dimensionless" and not scale["inverse_transform_available"]


@pytest.mark.parametrize("prediction,probabilities", [
    (2, [0.1, 0.9]), (True, [0.1, 0.9]), (1, [0.9, 0.1]), (1, [0.1]),
    (1, [0.1, float("nan")]), (1, [0.1, True]), (1, [0.1, 1.9]), (1, [0.1, 0.2]),
])
def test_bad_output_shape_or_semantics_retains_raw_but_does_not_decode(prediction, probabilities):
    result = decoded(config(), prediction, probabilities)
    assert result["mapping_status"] == "output_mismatch" and "decoded" not in result


@pytest.mark.parametrize("prediction", [[0] * 10, [0] * 9, [False] * 9])
def test_multilabel_shape_and_threshold_disagreement_are_not_reinterpreted(prediction):
    cfg = config("SortingSignal", 9, "multi_label_classification")
    result = decoded(cfg, prediction, [0.1] * 6 + [0.9, 0.1, 0.1])
    assert result["mapping_status"] == "output_mismatch" and "decoded" not in result


def test_same_dataset_with_changed_checkpoint_config_cannot_reuse_label_binding():
    cfg = config()
    result = decode_values([{"predicted_class": 1, "probabilities": [0.1, 0.9]}],
                           schema_for_config(cfg), output_scale(cfg | {"num_labels": 10}))
    assert result["mapping_status"] == "unavailable"
    assert result["reason"] == "prediction_config_binding_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["known", "unknown", "shape_mismatch"])
async def test_actual_factory_csv_semantics_reach_writer_report_and_zip_without_editing_raw_csv(tmp_path, monkeypatch, case):
    from backend.web_v2 import research_predictions as predictions
    from backend.web_v2 import research_report as report
    from backend.web_v2.research_chapter_context import source_context
    from backend.web_v2.research_dossier import build_dossier
    from mcp_tools.models.factory.prediction import fintuned_operations as operations

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    cfg = config() if case != "unknown" else config() | {"dataset": "unverified/DeepLocBinary_ESMFold"}
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(cfg))
    predicted = {"predicted_class": 1, "probabilities": [0.1, 0.9] if case != "shape_mismatch" else [0.9]}
    sequence = "MKTAYIAKQRQISFVK"
    module = SimpleNamespace(parse_fasta=lambda _: [("query", sequence)],
                             load_model_and_tokenizer=lambda args: (None, None, None, "cpu"),
                             predict=lambda *args: predicted)
    monkeypatch.setattr(operations, "_get_module", lambda _: (module, lambda *args: {"input_ids": [], "attention_mask": []}))
    scale = output_scale(cfg)
    spec = {"id": "membrane", "task": "Membrane Protein", "model": "ESM2-650M", "dataset": "DeepLocBinary",
            "tool": "predict_protein_function", "adapter_path": str(adapter), "output_scale": scale,
            "label_schema": checkpoint_label_schema(adapter)}
    monkeypatch.setattr(predictions, "candidates", lambda context: [spec])
    captured = []

    async def execute(spec, fasta, output, progress):
        result = json.loads(operations.predict_protein_function.runtime_implementation(
            str(fasta), spec["task"], spec["model"], spec["adapter_path"], output_file=str(output)))
        assert result["status"] == "success"
        captured.append((output.read_bytes(), result["file_info"]["sha256"]))
        return result

    real_run = predictions.run_predictions
    async def predict(context, update, folder):
        return await real_run(context, update, folder, execute=execute)

    monkeypatch.setattr(predictions, "run_predictions", predict)
    monkeypatch.setattr(report, "is_online_mode", lambda: False)
    state = await report.run_research({"analyses": ["ai_predictions"], "model_id": "fixture", "sequence": sequence},
                                     lambda state: None, synthesize=False, service_cache=tmp_path / "remote")
    row = state["predictions"][0]
    assert row["status"] == "completed"
    expected = {"known": "legacy_dataset_convention", "unknown": "unavailable", "shape_mismatch": "output_mismatch"}[case]
    assert row["classification"]["mapping_status"] == expected
    assert json.loads(row["values"][0]["predicted_class"]) == 1
    assert state["sections"][0]["data"]["classification"] == row["classification"]
    assert json.loads(state["sources"][0]["text"])["classification"] == row["classification"]
    raw, digest = captured[0]
    assert hashlib.sha256(raw).hexdigest() == digest
    assert (tmp_path / "predictions/membrane.csv").read_bytes() == raw
    if case == "known":
        assert "Soluble" in json.dumps(source_context(state["sources"]))
        assert "e300ff0b3b09b2ce023d5d35efaeb187f657db7e" in json.dumps(source_context(state["sources"]))
        assert "Soluble" in predictions.report_appendix([row])
    else:
        assert "Class labels unavailable" in predictions.report_appendix([row])
        assert "decoded" not in row["classification"]
    state["report"] = "# Engineering fixture"
    artifact = {"id": row["id"], "status": "included", "content": raw, "sha256": digest,
                "bytes": len(raw), "validation": "worker_sha256", "output_scale": scale}
    with build_dossier(state, prediction_files=[artifact]) as handle, zipfile.ZipFile(handle) as archive:
        assert archive.read("predictions/001.csv") == raw
        index = json.loads(archive.read("predictions/index.json"))[0]
        assert index["classification"] == row["classification"]
        assert index["artifact"]["sha256"] == digest
        assert json.loads(archive.read(index["analysis_file"]))["data"]["classification"] == row["classification"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_config", [False, True])
async def test_receipt_recovery_uses_bound_saved_config_not_a_forged_row_label(tmp_path, monkeypatch, changed_config):
    from backend.web_v2 import research_receipts as receipts
    from backend.web_v2.job_queue import JobQueue
    from backend.web_v2.research_input_snapshot import save_inputs
    from backend.web_v2.research_recovery import recover_evidence
    from mcp_tools.scheduler import client, jobs
    from mcp_tools.scheduler.receipt_binding import value_digest

    queue = JobQueue(tmp_path / "jobs")
    parent, _ = queue.enqueue("alice", "protein-research", "research", {"fixture": True}, "label-recovery")
    assert queue.claim_next("label-recovery", "research")["id"] == parent
    folder = tmp_path / "predictions"
    folder.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    cfg = config()
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(cfg | ({"num_labels": 10} if changed_config else {})))
    scale = output_scale(cfg)
    sequence = "MKTAYIAKQRQISFVK"
    (folder / "query.fasta").write_text(">query\n" + sequence + "\n")
    output = folder / "model.csv"
    sample = {"header": "query", "sequence": sequence, "predicted_class": 1, "probabilities": [0.1, 0.9]}
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample)
        writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in sample.items()})
    args = {"fasta_file": str(folder / "query.fasta"), "output_file": str(output), "task": "Membrane Protein",
            "model_name": "ESM2-650M", "adapter_path": str(adapter)}
    value = {"status": "success", "content_preview": json.dumps([sample]),
             "file_info": {"file_name": output.name, "format": "csv", "file_size": output.stat().st_size},
             "biological_metadata": {"task": args["task"], "dataset_name": "DeepLocBinary", "model_name": args["model_name"],
                                     "fasta_file": args["fasta_file"], "adapter_path": str(adapter), "sequence_count": 1,
                                     "output_scale": scale}}
    task, _ = queue.enqueue("alice", "mcp:cpu_analysis", "cpu", {"fixture": True}, "label-recovery",
                            metadata={"tool": "predict_protein_function", "category": "cpu_analysis", "parent_job": parent,
                                      "arguments_sha256": value_digest(args)})
    claimed = queue.claim_next("label-recovery", "cpu")
    jobs.save_json(queue.files(task) / "mcp-result.json", {"value": value, "is_error": False})
    queue.finish(task, "completed", lease=claimed["lease"])
    calls = []
    async def rpc(name, arguments):
        calls.append(name)
        assert name == "tool_job"
        return jobs.ToolJobs(queue).get(arguments["identity"], arguments["owner"])
    monkeypatch.setattr(client, "rpc", rpc)
    monkeypatch.setenv("VENUS_QUEUE_OWNER", "alice")
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID", "resumed-parent")
    row = {"id": "model", "task": args["task"], "model": args["model_name"], "dataset": "DeepLocBinary",
           "tool": "predict_protein_function", "status": "running", "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
           "output_scale": scale, "classification": {"decoded": [{"predicted_labels": ["FABRICATED"]}]}}
    state = {"predictions": [row], "sources": [], "sections": [],
             "input_snapshot": save_inputs(tmp_path / "inputs", {"sequence": sequence}),
             "tool_receipts": {"prediction:model": {"version": 1, "tool": row["tool"], "arguments": args,
                               "arguments_sha256": value_digest(args), "owner_sha256": receipts.owner_digest("alice"),
                               "parent_job": parent, "task_id": task}}}
    def add(sources):
        state["sources"].extend(sources)
        return sources
    before = output.read_bytes()
    await recover_evidence(state, lambda: None, add, service_cache=tmp_path / "remote", input_store=tmp_path / "inputs")
    assert row["status"] == "completed" and calls == ["tool_job"]
    assert row["recovery"]["validation"] == "complete_saved_values"
    assert "FABRICATED" not in json.dumps(state)
    if changed_config:
        assert row["classification"]["mapping_status"] == "unavailable"
        assert "decoded" not in row["classification"]
    else:
        assert row["classification"]["decoded"][0]["predicted_labels"] == ["Soluble"]
    assert output.read_bytes() == before


@pytest.mark.parametrize("declared", [
    {"class_order": ["Soluble", "Membrane"]},
    {"id2label": {"0": "Soluble", "1": "Membrane"}},
    {"label2id": {"Soluble": 0, "Membrane": 1}},
])
def test_saved_explicit_mapping_conflicts_are_not_overridden_by_legacy_convention(declared):
    result = schema_for_config(config() | declared)
    assert result["mapping_status"] == "unavailable"
    assert result["reason"] == "checkpoint_explicit_labels_conflict"


def test_matching_explicit_class_order_preserves_legacy_source_qualification():
    result = schema_for_config(config() | {"class_order": ["Membrane", "Soluble"]})
    assert result["mapping_status"] == "legacy_dataset_convention"
    assert result["provenance"]["checkpoint_revision_recorded"] is False
