"""Prediction delivery tests: receipts, isolation, precision and recovery."""
import copy
import csv
import io
import json

import pytest

from agent.harness.contracts import digest
from mcp_tools.models.evolved.result_delivery import PREVIEW_ROWS, deliver_prediction


@pytest.fixture
def prediction():
    scope = {
        "protein": "fixture", "target": "relative_response",
        "assay": "synthetic_assay", "unit": "ratio_to_reference",
        "context": {"organism": "synthetic"},
        "covariates": [
            {"name": "temperature", "unit": "C", "minimum": 21.8, "maximum": 65.5},
            {"name": "reference", "unit": "C", "minimum": 21.8, "maximum": 24.8},
        ],
        "grouping": {"level": "biological_replicate", "evidence_sha256": "d" * 64},
    }
    inputs = [{"id": f"sample-{i}", "protein": "fixture", "group": "g1", "values": [24.8, 24.8]}
              for i in range(25)]
    predictions = [{"id": row["id"], "prediction": 0.9942129558371191 + i} for i, row in enumerate(inputs)]
    receipt = {"schema": 1, "kind": "model_library_inference", "call_id": "b" * 32,
               "entry_id": "a" * 64, "session_id": "owner", "mode": "science_agent",
               "scope_sha256": digest(scope), "input_sha256": digest(inputs),
               "prediction_sha256": digest(predictions), "rows": len(inputs)}
    result = {"success": True, "interface": "generated_context.v1",
              "predictions": predictions, "scope": scope, "receipt": receipt,
              "receipt_id": digest(receipt), "evaluation_scope": "Synthetic frozen validation."}
    return result, {"entry_id": "a" * 64, "inputs": inputs}


def test_exports_exact_values_scope_and_label_free_inputs_with_bounded_preview(prediction, tmp_path):
    result, args = prediction
    before = copy.deepcopy(result)
    for row in args["inputs"]:
        row["target"] = "must not enter a prediction export"
    card = deliver_prediction(tmp_path, json.dumps(result), args, session_id="owner")
    assert result == before
    payload = json.loads((tmp_path / card["files"]["json"]).read_text())
    assert payload["predictions"] == result["predictions"]
    assert payload["scope"] == result["scope"]
    assert all("target" not in row for row in payload["inputs"])
    assert payload["receipt_id"] == digest(payload["receipt"])
    assert str(result["predictions"][0]["prediction"]) in card["content"]
    assert f"Showing {PREVIEW_ROWS} of 25" in card["content"]
    assert "sample-24" not in card["content"]
    assert all(bound in card["scope_markdown"] for bound in ("21.8", "65.5", "24.8"))
    assert "temperature: 21.8 to 65.5 C" in card["applicability_summary"]
    assert "reference: 21.8 to 24.8 C" in card["applicability_summary"]
    assert "Synthetic frozen validation." in card["applicability_summary"]
    assert "d" * 64 not in card["applicability_summary"]
    assert "d" * 64 in card["scope_markdown"]
    table = list(csv.DictReader(io.StringIO((tmp_path / card["files"]["csv"]).read_text())))
    assert len(table) == 25 and table[0]["unit"] == "ratio_to_reference"
    assert float(table[0]["prediction"]) == result["predictions"][0]["prediction"]
    report = (tmp_path / card["files"]["report"]).read_text()
    assert "sample-24" in report and "65.5" in report
    assert str(tmp_path) not in json.dumps(card)
    assert not any(path.stat().st_mode & 0o077 for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize("damage", ["scope", "predictions", "receipt", "identity", "input", "session"])
def test_changed_receipt_or_cross_session_result_cannot_create_files(prediction, tmp_path, damage):
    result, args = prediction
    sid = "owner"
    if damage == "scope":
        result["scope"]["protein"] = "another"
    if damage == "predictions":
        result["predictions"][0]["prediction"] = 900
    if damage == "receipt":
        result["receipt"]["rows"] = 1
    if damage == "identity":
        result["receipt_id"] = "../outside"
    if damage == "input":
        args["inputs"][0]["values"][0] = 33.8
    if damage == "session":
        sid = "other-owner"
    with pytest.raises(ValueError):
        deliver_prediction(tmp_path, result, args, session_id=sid)
    assert not list(tmp_path.iterdir())


def test_ordered_sample_identity_is_checked_even_with_matching_output_hash(prediction, tmp_path):
    result, args = prediction
    result["predictions"].reverse()
    result["receipt"]["prediction_sha256"] = digest(result["predictions"])
    result["receipt_id"] = digest(result["receipt"])
    with pytest.raises(ValueError, match="records"):
        deliver_prediction(tmp_path, result, args, session_id="owner")
    assert not list(tmp_path.iterdir())


def test_repeated_delivery_is_stable_and_download_is_scoped(prediction, tmp_path):
    from backend.web_v2.research_files import open_owned_file

    result, args = prediction
    owner = tmp_path / "owner"
    owner.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    first = deliver_prediction(owner, result, args, session_id="owner")
    files = {p: (owner/p).read_bytes() for p in first["files"].values()}
    assert first == deliver_prediction(owner, result, args, session_id="owner")
    assert files == {p: (owner/p).read_bytes() for p in first["files"].values()}
    key = "agent/" + first["files"]["json"]
    with open_owned_file({"agent": owner}, key) as stream:
        assert json.load(stream)["receipt_id"] == result["receipt_id"]
    with pytest.raises((OSError, ValueError)):
        open_owned_file({"agent": other}, key)


def test_linked_export_directory_does_not_escape_or_repeat_inference(prediction, tmp_path):
    result, args = prediction
    workspace = tmp_path/"work"
    workspace.mkdir()
    outside = tmp_path/"outside"
    outside.mkdir()
    (workspace/"results").symlink_to(outside, target_is_directory=True)
    card = deliver_prediction(workspace, result, args, session_id="owner")
    assert card["file_error"] and not card["files"]
    assert "temperature: 21.8 to 65.5 C" in card["applicability_summary"]
    assert "0.9942129558371191" in card["content"]
    assert not list(outside.iterdir())


def test_metadata_is_text_and_csv_formula_ids_are_escaped(prediction, tmp_path):
    result, args = prediction
    malicious = "=HYPERLINK(1)|<script>alert(1)</script>"
    args["inputs"][0]["id"] = result["predictions"][0]["id"] = malicious
    result["scope"]["context"]["note"] = "[click](javascript:alert(1))"
    result["receipt"].update(input_sha256=digest(args["inputs"]),
                             prediction_sha256=digest(result["predictions"]),
                             scope_sha256=digest(result["scope"]))
    result["receipt_id"] = digest(result["receipt"])
    card = deliver_prediction(tmp_path, result, args, session_id="owner")
    assert "<script>" not in card["content"]
    assert "[click](javascript:" not in card["scope_markdown"]
    assert "[click](javascript:" not in card["applicability_summary"]
    assert "context / note:" in card["applicability_summary"]
    rows = list(csv.DictReader(io.StringIO((tmp_path/card["files"]["csv"]).read_text())))
    assert rows[0]["id"] == "'" + malicious
    payload = json.loads((tmp_path/card["files"]["json"]).read_text())
    assert payload["inputs"][0]["id"] == malicious


def test_result_checkpoint_survives_error_reload_and_does_not_rewrite_answer(prediction, tmp_path):
    from agent.session_store import persistable_state
    from backend.web_v2.agent_conversation import completed_messages, record_model_result
    from backend.web_v2.agent_recovery import canonical_recovered_history

    result, args = prediction
    event = {"kind": "model_result", "result": deliver_prediction(tmp_path, result, args, session_id="owner")}
    original = copy.deepcopy(event)
    state = {"session_id": "owner", "history": [{"role": "user", "content": "Predict", "turn_id": "t"}]}
    assert record_model_result(state, event, "t")
    assert event == original
    state["status"] = "error"
    restored = json.loads(json.dumps(persistable_state(state)))
    assert not record_model_result(restored, event, "t")
    card = restored["history"][1]
    assert card["kind"] == "model_result" and "65.5" in card["model_result"]["scope_markdown"]
    recovered = canonical_recovered_history(restored["history"], "t", "Done.")
    assert recovered[1] == card and recovered[-1]["content"] == "Done."
    assert completed_messages([], "Predict", [], "Done.", "kimi-code")[-1]["content"] == "Done."


def test_applicability_keeps_zero_negative_bounds_and_condition_units(prediction, tmp_path):
    result, args = prediction
    result["scope"]["covariates"] = [
        {"name": "temperature", "unit": "C", "minimum": -5, "maximum": 0},
        {"name": "treatment", "unit": "0=control; 1=treated", "minimum": 0, "maximum": 1},
    ]
    for row in args["inputs"]:
        row["values"] = [-1, 0]
    result["receipt"].update(scope_sha256=digest(result["scope"]), input_sha256=digest(args["inputs"]))
    result["receipt_id"] = digest(result["receipt"])
    summary = deliver_prediction(tmp_path, result, args, session_id="owner")["applicability_summary"]
    assert "temperature: -5 to 0 C" in summary
    assert "treatment: 0 to 1 0=control; 1=treated" in summary
    assert "context / organism: synthetic" in summary


@pytest.mark.parametrize("interface, limits", [
    ("generated_python.v1", {"sequence_length": 3}),
    ("generated_sequence.v1", {"minimum_sequence_length": 3, "maximum_sequence_length": 100}),
    ("generated_classifier.v1", {"sequence_length": 3, "class_labels": ["nonbinder", "binder"]}),
])
def test_sequence_applicability_preserves_length_and_class_order(prediction, tmp_path, interface, limits):
    result, args = prediction
    result["interface"] = interface
    result["scope"] = {key: result["scope"][key] for key in ("protein", "assay", "target", "unit")} | limits
    args["inputs"] = [{"id": "sample", "sequence": "ACD"}]
    result["predictions"] = [{"id": "sample", "prediction": 0}]
    result["receipt"].update(rows=1, scope_sha256=digest(result["scope"]),
                             input_sha256=digest(args["inputs"]), prediction_sha256=digest(result["predictions"]))
    result["receipt_id"] = digest(result["receipt"])
    card = deliver_prediction(tmp_path, result, args, session_id="owner")
    summary = card["applicability_summary"]
    if "minimum_sequence_length" in limits:
        assert "minimum sequence length: 3" in summary and "maximum sequence length: 100" in summary
    else:
        assert "sequence length: 3" in summary
    if "class_labels" in limits:
        assert "class 0: nonbinder" in summary and "class 1: binder" in summary
    saved = json.loads((tmp_path / card["files"]["json"]).read_text())
    assert saved["scope"] == result["scope"] and saved["receipt"] == result["receipt"]
