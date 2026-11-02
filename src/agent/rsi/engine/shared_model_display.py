"""Versioned compact native returns backed by complete local scientific records."""

from __future__ import annotations

import copy
import json

if __package__:
    from .protein_model import digest
    from .training_feedback_view import compact_feedback
else:
    from protein_model import digest
    from training_feedback_view import compact_feedback

FIELD = "tool_result_presentation"
LEGACY_POLICY = "compact_shared_tool_results.v1"
VECTOR_POLICY = "compact_shared_tool_results.v2"
POLICY = "compact_shared_tool_results.v3"
POLICIES = frozenset({LEGACY_POLICY, VECTOR_POLICY, POLICY})
MAX_DISPLAY_CHARACTERS = 90000
OPERATIONS = frozenset({"inspect_model_code_task", "evaluate_model_code", "discover_generated_models"})


def _common(records):
    result = {}
    for key in records[0]:
        if key == "task_id" or not all(key in row for row in records):
            continue
        values = [row[key] for row in records]
        if all(digest(value) == digest(values[0]) for value in values):
            result[key] = copy.deepcopy(values[0])
        elif all(isinstance(value, dict) for value in values):
            shared = _common(values)
            if shared:
                result[key] = shared
    return result


def _difference(value, common):
    result = {}
    for key, item in value.items():
        if key not in common:
            result[key] = copy.deepcopy(item)
        elif digest(item) != digest(common[key]):
            # Only equal values and dictionary intersections enter the defaults.
            result[key] = _difference(item, common[key])
    return result


def _factored(rows, name):
    if not rows:
        return {"shared_defaults": {}, name: []}
    common = _common(rows)
    return {"shared_defaults": common, name: [_difference(row, common) for row in rows]}


def _inspection_examples(rows):
    """Share exact input vectors, retaining every example identity and target."""
    rows = copy.deepcopy(rows)
    vectors, indices = [], {}
    for row in rows:
        training = row.get("contract", {}).get("training", {})
        examples = training.get("examples")
        if (
            not isinstance(examples, list)
            or not examples
            or "example_rows" in training
            or not all(
                isinstance(example, dict) and isinstance(example.get("values"), list) and "values_ref" not in example
                for example in examples
            )
        ):
            continue
        converted = []
        for example in examples:
            vector = example["values"]
            identity = digest(vector)
            if identity not in indices:
                indices[identity] = len(vectors)
                vectors.append(copy.deepcopy(vector))
            specific = {key: copy.deepcopy(value) for key, value in example.items() if key != "values"}
            specific["values_ref"] = indices[identity]
            converted.append(specific)
        del training["examples"]
        training["example_rows"] = converted
    return rows, vectors


def presentation(name, full, *, policy=POLICY):
    if policy not in POLICIES:
        raise ValueError("Unsupported shared tool presentation policy")
    if name not in OPERATIONS:
        raise ValueError("Unsupported shared tool presentation operation")
    if name == "inspect_model_code_task":
        content = {key: copy.deepcopy(value) for key, value in full.items() if key != "targets"}
        rows = full["targets"]
        if policy in {VECTOR_POLICY, POLICY}:
            rows, vectors = _inspection_examples(rows)
            content["training_example_values"] = vectors
        content.update(_factored(rows, "targets"))
    elif name == "evaluate_model_code":
        content = {key: copy.deepcopy(value) for key, value in full.items() if key != "results"}
        rows = [
            {
                "task_id": row["task_id"],
                "ledger_sha256": row["ledger_sha256"],
                "feedback_sha256": digest(row["result"]),
                "result": compact_feedback(row["result"]),
            }
            for row in full["results"]
        ]
        content.update(_factored(rows, "results"))
        content["feedback_format"] = (
            "Training-only candidate metrics and control tables retain their numeric values. Columns specify each table. Repeated group IDs and implementation/prediction hashes are omitted from this display; the complete feedback is bound by feedback_sha256."
        )
    else:
        content = {key: copy.deepcopy(value) for key, value in full.items() if key != "targets"}
        if all(len(row["result"]["models"]) == 1 for row in full["targets"]):
            rows = [{"task_id": row["task_id"], "model": row["result"]["models"][0]} for row in full["targets"]]
            content.update(_factored(rows, "targets"))
        else:
            content.update(_factored(full["targets"], "targets"))
    return content


def envelope(name, full, *, policy=POLICY):
    identity = digest(full)
    value = {
        "schema": policy,
        "operation": name,
        "artifact": "shared_tool_response_" + identity + ".json",
        "result_sha256": identity,
        "reading": "For each target, recursively merge shared_defaults with its target-specific fields. Lists are complete values, not patches. task_id remains explicit. This compact display has a complete local receipt verified independently by the host.",
        "content": presentation(name, full, policy=policy),
    }
    if policy in {VECTOR_POLICY, POLICY} and name == "inspect_model_code_task":
        value["reading"] += (
            " For training.example_rows, replace values_ref with values=training_example_values[values_ref], then rename example_rows to examples. Indices are zero-based; all original identities, groups, targets and vector values are retained without rounding."
        )
    # Preserve the source handle and attempt budget used by the next tool call.
    for key in ["source_sha256", "attempts_remaining", "status"]:
        if key in full:
            value[key] = copy.deepcopy(full[key])
    # v3 explicitly emits compact JSON as the MCP text block. Earlier policies
    # retain their original pretty-JSON limit and byte-for-byte envelopes.
    if len(wire_text(value)) > MAX_DISPLAY_CHARACTERS:
        raise ValueError("Shared tool display exceeds its 90000-character budget; use a smaller admitted batch")
    return value


def wire_text(value):
    """Serialize exactly the text emitted by the versioned native tool."""
    if value["schema"] == POLICY:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(value, ensure_ascii=True, indent=2)
