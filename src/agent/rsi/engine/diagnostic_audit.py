"""External trace/artifact audit for deterministic engineering fixtures only.

Kept outside the agent sandbox. Unlike the legacy pilot check, successful tool
results must match their input calls. No biological generalization is measured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from agent.rsi.campaigns.pilot import matches
from .storage import safe_child


def schema_matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and schema_matches(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and (not expected or all(schema_matches(x, expected[0]) for x in actual))
    if isinstance(expected, bool):
        return type(actual) is bool
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and math.isfinite(actual)
    return isinstance(actual, str) if isinstance(expected, str) else actual is None


def expected_output(task):
    expected = json.loads(json.dumps(task["expected"]))
    if task["kind"] == "sequence":
        # Same public mass constants as Biopython; independent sum formula,
        # not an independent scientific dataset or oracle implementation.
        from Bio.Data.IUPACData import protein_weights

        sequence = "ACDEFGHIKLMNPQRSTVWY" * (task["variant"] + 1)
        expected["molecular_weight"] = round(
            sum(protein_weights[a] for a in sequence) - (len(sequence) - 1) * 18.0153, 2
        )
    return expected


def _path(actual, name):
    return actual in (name, "./" + name, "/workspace/" + name)


def _arguments(task, name, args, slot):
    kind = task["kind"]
    if kind in ("sequence", "invalid"):
        return _path(args.get("fasta_file"), "input.fasta")
    if kind == "temperature":
        return (
            matches(args.get("value"), 37.0 + task["variant"] * 7.5 + 273.15)
            and str(args.get("source_unit", "")).lower() in ("k", "kelvin")
            and str(args.get("target_unit", "")).lower() in ("c", "celsius")
        )
    if kind == "mutation":
        return args.get("sequence") == "ACDEFGHIKLMNPQRSTVWY" * (task["variant"] + 1) and args.get("mutation") == (
            "C2A" if slot == "valid_mutation" else "A2V"
        )
    if kind == "ids":
        return _path(args.get("input_csv"), "input.csv") and _path(args.get("prediction_csv"), "prediction.csv")
    if args.get("model_id") != "fixture":
        return False
    if name == "fit_linear_fixture":
        return _path(args.get("dataset_csv"), "train.csv")
    if name == "validate_linear_fixture":
        return _path(args.get("validation_csv"), "validation.csv")
    wanted = "unrelated_protein_property" if slot == "reject" else "synthetic_linear_fixture"
    return _path(args.get("input_csv"), "test_inputs.csv") and args.get("task", "synthetic_linear_fixture") == wanted


def _requirements(task):
    kind, expected = task["kind"], expected_output(task)
    if kind == "sequence":
        return [("properties", "sequence_properties", {"valid": True, "properties": expected})]
    if kind == "invalid":
        return [("invalid", "sequence_properties", expected)]
    if kind == "temperature":
        return [("temperature", "convert_temperature", expected)]
    if kind == "mutation":
        return [(slot, "check_mutation", expected[slot]) for slot in ("valid_mutation", "invalid_mutation")]
    if kind == "ids":
        return [("ids", "audit_prediction_table", expected)]
    return [
        ("fit", "fit_linear_fixture", {"model_id": "fixture", "status": "trained", "registered": False}),
        ("validate", "validate_linear_fixture", {"model_id": "fixture", "validated": True, "registered": True}),
        ("predict", "predict_linear_fixture", {"valid": True, "predictions": expected["predictions"]}),
        ("reject", "predict_linear_fixture", {"valid": False, "error": "incompatible_or_unvalidated_model"}),
    ]


def audit(task: dict, workspace: Path, run: dict, events_path: Path) -> dict:
    events, trace_errors = [], []
    if events_path.exists():
        for line_number, line in enumerate(events_path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("event must be an object")
                events.append(event)
            except (ValueError, TypeError):
                trace_errors.append(f"invalid_event:{line_number}")
    starts, records, ended = {}, [], set()
    for index, event in enumerate(events):
        cid = event.get("tool_call_id")
        if event.get("kind") not in {"tool_call_start", "tool_result"}:
            continue
        if not isinstance(cid, str) or not cid:
            trace_errors.append(f"missing_call_id:{index}")
            continue
        if event.get("kind") == "tool_call_start":
            if cid in starts:
                trace_errors.append(f"duplicate_call_start:{index}")
                continue
            starts[cid] = (index, event)
        else:
            if cid not in starts or cid in ended:
                trace_errors.append(f"unpaired_or_duplicate_result:{index}")
                continue
            ended.add(cid)
            if event.get("is_error"):
                continue
            begin, call = starts[cid]
            try:
                payload = json.loads(
                    event.get("output", ""), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
                )
                args = call.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be an object")
            except (ValueError, TypeError):
                continue
            records.append(
                {
                    "call_id": cid,
                    "start": begin,
                    "end": index,
                    "name": str(call.get("tool_name", "")).split("__")[-1],
                    "args": args,
                    "payload": payload,
                }
            )
    evidence, mismatches = {}, []
    for slot, name, expected in _requirements(task):
        correct = [r for r in records if r["name"] == name and matches(r["payload"], expected)]
        for record in correct:
            if _arguments(task, name, record["args"], slot):
                if slot == "validate":
                    rmse = record["payload"].get("rmse")
                    if type(rmse) not in (int, float) or not math.isfinite(rmse) or not 0 <= rmse <= 1e-6:
                        continue
                evidence[slot] = record
                break
        if correct and slot not in evidence:
            mismatches.append(slot)
    required = [slot for slot, _, _ in _requirements(task)]
    complete = all(slot in evidence for slot in required)
    ordered = True
    registry_valid = None
    if task["kind"] == "reuse":
        ordered = (
            complete
            and evidence["fit"]["end"] < evidence["validate"]["start"]
            and all(evidence["validate"]["end"] < evidence[x]["start"] for x in ("predict", "reject"))
        )
        try:
            card = json.loads(safe_child(workspace, "registry/fixture.json").read_text())
            registry_valid = (
                card.get("status") == "validated"
                and card.get("task") == "synthetic_linear_fixture"
                and type(card.get("validation_rmse")) in (int, float)
                and 0 <= card["validation_rmse"] <= 1e-6
            )
        except (OSError, ValueError, TypeError):
            registry_valid = False
    exists = valid_json = schema_ok = values_ok = False
    artifact_error = None
    try:
        artifact = safe_child(workspace, "output.json")
        exists = artifact.exists()
        if artifact.stat().st_size > 1000000:
            raise ValueError("artifact too large")
        actual = json.loads(artifact.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        valid_json = True
        expected = expected_output(task)
        schema_ok = schema_matches(actual, expected)
        values_ok = matches(actual, expected)
    except (OSError, ValueError, TypeError) as exc:
        artifact_error = type(exc).__name__
    tool_correct = complete and ordered and registry_valid is not False and not trace_errors
    passed = run.get("status") == "completed" and schema_ok and values_ok and tool_correct
    if run.get("status") != "completed":
        reason = "runtime_not_completed"
    elif trace_errors:
        reason = "invalid_execution_trace"
    elif not tool_correct:
        reason = "incorrect_or_missing_tool_evidence"
    elif not schema_ok:
        reason = "invalid_artifact_schema"
    elif not values_ok:
        reason = "incorrect_artifact_values"
    else:
        reason = "artifact_and_execution_checks_passed"
    return {
        "audit_version": 2,
        "task_id": task["task_id"],
        "reason": reason,
        "trace_errors": trace_errors,
        "passed": passed,
        "artifact_exists": exists,
        "artifact_json_valid": valid_json,
        "artifact_schema_valid": schema_ok,
        "artifact_values_correct": values_ok,
        "artifact_error": artifact_error,
        "tool_chain_correct": tool_correct,
        "required_evidence": required,
        "matched_evidence": {k: v["call_id"] for k, v in evidence.items()},
        "correct_output_wrong_arguments": mismatches,
        "execution_order_valid": ordered,
        "registry_valid": registry_valid,
        "runtime_completed": run.get("status") == "completed",
        "computation_delivery_gap": tool_correct and not values_ok,
    }
