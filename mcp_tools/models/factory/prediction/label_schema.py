"""Checkpoint-bound class semantics from audited legacy dataset conventions.

Class order lives in src/constant.json. The small source manifest binds that
order to public dataset evidence; it does not invent a checkpoint revision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from .output_scale import output_scale

SCHEMA = "venusfactory.classification-labels.v1"
ROOT = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _catalog():
    labels = json.loads((ROOT / "src/constant.json").read_text())["web_ui"]["label_mapping_function"]
    records = json.loads(Path(__file__).with_name("label_schema_sources.json").read_text())["datasets"]
    return labels, records


def _unavailable(reason):
    return {"schema": SCHEMA, "mapping_status": "unavailable", "reason": reason}


def schema_for_config(config):
    """No task-name or directory-name inference; require the saved dataset ABI."""
    if not isinstance(config, dict):
        return _unavailable("missing_checkpoint_config")
    if config.get("problem_type") == "regression":
        return {"schema": SCHEMA, "mapping_status": "not_applicable"}
    labels, records = _catalog()
    record = next((r for r in records.values() if config.get("dataset") in r["dataset_ids"]), None)
    if record is None:
        return _unavailable("dataset_mapping_not_verified")
    if (config.get("problem_type") != record["problem_type"]
            or type(config.get("num_labels")) is not int
            or config["num_labels"] != record["num_labels"]):
        return _unavailable("checkpoint_label_contract_mismatch")
    order = labels[record["task"]]
    digest = hashlib.sha256(json.dumps(order, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if len(order) != record["num_labels"] or digest != record["label_order_sha256"]:
        return _unavailable("canonical_label_order_changed")
    # A saved explicit mapping cannot be silently overridden by a legacy convention.
    declared = {"class_order": list(order), "id2label": {str(i): name for i, name in enumerate(order)},
                "label2id": {name: i for i, name in enumerate(order)}}
    if any(key in config and config[key] != expected for key, expected in declared.items()):
        return _unavailable("checkpoint_explicit_labels_conflict")
    provenance = {k: copy.deepcopy(record[k]) for k in (
        "provenance_kind", "repo", "revision", "source_file", "source_url", "sha256",
        "rows", "label_order_sha256", "checkpoint_revision_recorded", "audited_on")}
    if record.get("meaning_source"):
        provenance["meaning_source"] = record["meaning_source"]
    meanings = record.get("meanings", {})
    return {"schema": SCHEMA, "mapping_status": "legacy_dataset_convention",
            "dataset": config["dataset"], "problem_type": config["problem_type"],
            "num_labels": len(order), "class_order": list(order),
            "class_meanings": [meanings.get(name, name) for name in order],
            "checkpoint_config_sha256": output_scale(config)["checkpoint_config_sha256"],
            "provenance": provenance}


def checkpoint_label_schema(adapter):
    try:
        path = Path(adapter) / "lr5e-4_bt12k_ga8.json"
        if path.stat().st_size > 1_000_000:
            return _unavailable("checkpoint_config_too_large")
        return schema_for_config(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return _unavailable("missing_checkpoint_config")


def _parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def decode_values(values, schema, scale):
    """Derive labels separately; never replace CSV bytes, raw values or scores."""
    schema = copy.deepcopy(schema) if isinstance(schema, dict) else _unavailable("missing_label_schema")
    if schema.get("mapping_status") != "legacy_dataset_convention":
        return schema
    if schema.get("checkpoint_config_sha256") != (scale or {}).get("checkpoint_config_sha256"):
        return _unavailable("prediction_config_binding_mismatch")
    decoded = []
    try:
        order = schema["class_order"]
        if not isinstance(values, list) or not values:
            raise ValueError("missing_outputs")
        for row in values:
            probabilities = _parsed(row.get("probabilities"))
            prediction = _parsed(row.get("predicted_class"))
            if (not isinstance(probabilities, list) or len(probabilities) != len(order)
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
                raise ValueError("probability_shape_or_range_mismatch")
            if schema["problem_type"] == "single_label_classification":
                if type(prediction) is not int or not 0 <= prediction < len(order):
                    raise ValueError("class_index_out_of_range")
                if abs(sum(probabilities) - 1) > 1e-4 or probabilities[prediction] != max(probabilities):
                    raise ValueError("class_probability_mismatch")
                active = [prediction]
            else:
                if (not isinstance(prediction, list) or len(prediction) != len(order)
                        or any(type(x) is not int or x not in (0, 1) for x in prediction)):
                    raise ValueError("multilabel_shape_mismatch")
                if prediction != [int(p > 0.5) for p in probabilities]:
                    raise ValueError("multilabel_threshold_mismatch")
                active = [i for i, flag in enumerate(prediction) if flag]
            result = {"predicted_labels": [order[i] for i in active],
                      "predicted_meanings": [schema["class_meanings"][i] for i in active],
                      "probabilities_by_label": dict(zip(order, probabilities, strict=True))}
            if schema["problem_type"] == "multi_label_classification":
                result["decision_threshold"] = {"operator": ">", "value": 0.5}
                if not active:
                    result["interpretation"] = "No listed signal exceeds the model decision threshold."
            decoded.append(result)
    except (KeyError, TypeError, ValueError, OverflowError):
        schema.update(mapping_status="output_mismatch", reason="raw_output_does_not_match_label_contract")
        return schema
    schema["decoded"] = decoded
    schema["probability_calibration"] = "not_established"
    return schema
