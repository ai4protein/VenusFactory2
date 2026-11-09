"""Session-owned exports of an already executed Scientist model prediction.

This module checks the inference receipt and formats its values. It never fits,
replays or publishes a model, and never modifies the originating research goal.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
import re

from agent.harness.contracts import digest
from utils.scoped_files import write_bytes

PREVIEW_ROWS = 12


def _checked_payload(result, arguments, session_id):
    from agent.rsi.datasets.model_tasks import inference_inputs

    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict) or result.get("success") is not True:
        raise ValueError("A successful prediction is required")
    receipt = result["receipt"]
    scope, predictions = result["scope"], result["predictions"]
    rows = inference_inputs(arguments["inputs"], result["interface"])
    expected = {
        "schema": 1, "kind": "model_library_inference",
        "entry_id": arguments["entry_id"], "session_id": session_id,
        "scope_sha256": digest(scope), "input_sha256": digest(rows),
        "prediction_sha256": digest(predictions), "rows": len(rows),
    }
    if (not re.fullmatch(r"[0-9a-f]{64}", str(arguments["entry_id"]))
            or not session_id
            or not re.fullmatch(r"[0-9a-f]{32}", str(receipt.get("call_id", "")))
            or any(receipt.get(key) != value for key, value in expected.items())
            or result["receipt_id"] != digest(receipt)
            or not 1 <= len(rows) <= 256 or len(predictions) != len(rows)):
        raise ValueError("Prediction receipt does not match this session and input")
    for row, prediction in zip(rows, predictions):
        value = prediction.get("prediction")
        if (prediction.get("id") != row["id"] or type(value) not in (float, int)
                or not math.isfinite(value)):
            raise ValueError("Prediction records do not match the inference receipt")
    return {
        "schema": "scientist_prediction_result.v1",
        "interface": result["interface"], "inputs": rows,
        "predictions": predictions, "scope": scope,
        "evaluation_scope": result["evaluation_scope"],
        "receipt": receipt, "receipt_id": result["receipt_id"],
    }


def _cell(value):
    # All model metadata is data, including Markdown-looking IDs and scope text.
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, allow_nan=False)
    value = html.escape(value, quote=False)
    for character in ("\\", "|", "*", "_", "[", "]", chr(96)):
        value = value.replace(character, "\\" + character)
    return value.replace("\r", " ").replace("\n", " ")


def _scope_rows(value, path=""):
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _scope_rows(child, (path + " / " if path else "") + key.replace("_", " "))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value, 1):
            yield from _scope_rows(child, f"{path} / {index}")
    else:
        yield path, value


def _scope_markdown(payload):
    lines = ["| Property | Scope |", "| --- | --- |"]
    lines += [f"| {_cell(key)} | {_cell(value)} |" for key, value in _scope_rows(payload["scope"])]
    return "\n".join(lines) + "\n\n" + _cell(payload["evaluation_scope"])


def _prediction_markdown(payload, limit=None):
    scope = payload["scope"]
    title = " · ".join(str(scope[k]) for k in ("protein", "target") if scope.get(k))
    unit = scope.get("unit") or "Not specified"
    predictions = payload["predictions"]
    lines = [f"**{_cell(title or 'Model prediction')}**", "",
             "| Sample | Prediction | Unit |", "| --- | ---: | --- |"]
    lines += [f"| {_cell(row['id'])} | {_cell(row['prediction'])} | {_cell(unit)} |"
              for row in predictions[:limit]]
    if limit is not None and len(predictions) > limit:
        lines += ["", f"Showing {limit} of {len(predictions)} predictions. All rows are in the result files."]
    return "\n".join(lines)


def _csv_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def deliver_prediction(workspace, result, arguments, *, session_id):
    """Return a bounded display card and exports under the host-owned workspace.

    Only the trusted executor calls this after prediction. The model cannot choose
    a session, output root or filename. Link-safe writes reuse the shared file API.
    """
    payload = _checked_payload(result, arguments, session_id)
    scope_text = _scope_markdown(payload)
    folder = "results/predictions/" + payload["receipt_id"]
    files = {"report": folder + "/README.md", "csv": folder + "/predictions.csv",
             "json": folder + "/result.json"}
    card = {"receipt_id": payload["receipt_id"],
            "content": _prediction_markdown(payload, PREVIEW_ROWS),
            "scope_markdown": scope_text, "files": files}
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(["id", "prediction", "unit"])
    for row in payload["predictions"]:
        writer.writerow([_csv_cell(row["id"]), row["prediction"],
                         _csv_cell(payload["scope"].get("unit", ""))])
    report = ("# Prediction result\n\n" + _prediction_markdown(payload)
              + "\n\n## Applicability\n\n" + scope_text
              + "\n\n[Prediction table](predictions.csv) · [Full result and receipt](result.json)\n")
    try:
        write_bytes(workspace, files["csv"], table.getvalue().encode("utf-8"))
        write_bytes(workspace, files["report"], report.encode("utf-8"))
        # Complete machine-readable result is written last.
        write_bytes(workspace, files["json"],
                    (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    except (OSError, ValueError):
        # The inference already happened. Keep its values and receipt visible;
        # an export failure must not ask the model to repeat the computation.
        card["files"] = {}
        card["file_error"] = True
    return card
