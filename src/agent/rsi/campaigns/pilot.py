"""Six engineering task families and external artifact checks.

These generated fixtures are not a protein benchmark and provide no RSI efficacy
claim. This module stays outside the agent's sandbox and capability package.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from agent.rsi.engine.storage import safe_child, sha256_file


def table(header, values):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(values)
    return stream.getvalue()


def build_tasks(variants: int = 1) -> list[dict]:
    tasks = []
    for v in range(variants):
        split = ("development", "development", "selection", "audit", "audit")[v % 5]
        sequence = "ACDEFGHIKLMNPQRSTVWY" * (v + 1)
        common = {"split": split, "variant": v}
        tasks.append(
            {
                **common,
                "task_id": f"sequence-{v}",
                "kind": "sequence",
                "files": {"input.fasta": f">fixture_{v}\n{sequence}\n"},
                "prompt": "Use sequence_properties on input.fasta. Save output.json with sequence_id, sequence_length, aromaticity and molecular_weight from the computed properties.",
                "expected": {"sequence_id": f"fixture_{v}", "sequence_length": len(sequence), "aromaticity": 0.15},
                "required_tools": ["sequence_properties"],
            }
        )
        tasks.append(
            {
                **common,
                "task_id": f"invalid-sequence-{v}",
                "kind": "invalid",
                "files": {"input.fasta": f">invalid_{v}\n{sequence}X*\n"},
                "prompt": "Inspect input.fasta using sequence_properties. Do not silently remove nonstandard residues. Save its valid/error/invalid_residues result as output.json.",
                "expected": {"valid": False, "error": "nonstandard_residues", "invalid_residues": ["*", "X"]},
                "required_tools": ["sequence_properties"],
            }
        )
        celsius = 37.0 + v * 7.5
        tasks.append(
            {
                **common,
                "task_id": f"temperature-{v}",
                "kind": "temperature",
                "files": {},
                "prompt": f"An assay temperature was recorded as {celsius + 273.15} K. Use convert_temperature to report Celsius. Save its value and unit as output.json.",
                "expected": {"value": celsius, "unit": "C"},
                "required_tools": ["convert_temperature"],
            }
        )
        tasks.append(
            {
                **common,
                "task_id": f"mutation-{v}",
                "kind": "mutation",
                "files": {},
                "prompt": f'Using check_mutation, validate C2A and A2V against {sequence}. Save output.json as {{"valid_mutation": <C2A result>, "invalid_mutation": <A2V result>}}. Use one-based residue positions.',
                "expected": {
                    "valid_mutation": {"valid": True, "position": 2, "mutated_sequence": "AA" + sequence[2:]},
                    "invalid_mutation": {"valid": False, "error": "wildtype_mismatch", "observed": "C"},
                },
                "required_tools": ["check_mutation"],
            }
        )
        ids = [f"v{v}_{i}" for i in range(3)]
        tasks.append(
            {
                **common,
                "task_id": f"sample-ids-{v}",
                "kind": "ids",
                "files": {
                    "input.csv": table(["id", "sequence"], [[i, sequence] for i in ids]),
                    "prediction.csv": table(["id", "prediction"], [[ids[0], 1], [ids[0], 2], ["extra", 3]]),
                },
                "prompt": "Use audit_prediction_table to check input.csv and prediction.csv. Save the complete audit result as output.json. Do not invent missing predictions.",
                "expected": {
                    "valid": False,
                    "missing_ids": ids[1:],
                    "extra_ids": ["extra"],
                    "duplicate_ids": [ids[0]],
                    "duplicate_input_ids": [],
                },
                "required_tools": ["audit_prediction_table"],
            }
        )
        intercept, a, b = 1.0 + v, 2.0 + 0.2 * v, -0.5

        def labelled(points, intercept=intercept, a=a, b=b):
            return [[x1, x2, intercept + a * x1 + b * x2] for x1, x2 in points]

        train = labelled([(0, 0), (1, 0), (0, 1), (2, 1), (-1, 2), (3, -2)])
        valid = labelled([(4, 3), (-2, 4), (3, 3)])
        tests = [(f"new{v}_0", 2, 2), (f"new{v}_1", -3, 1)]
        tasks.append(
            {
                **common,
                "task_id": f"model-reuse-{v}",
                "kind": "reuse",
                "files": {
                    "train.csv": table(["x1", "x2", "y"], train),
                    "validation.csv": table(["x1", "x2", "y"], valid),
                    "test_inputs.csv": table(["id", "x1", "x2"], tests),
                },
                "prompt": "This is a synthetic engineering fixture, not biological evidence. Fit model 'fixture' on train.csv using fit_linear_fixture, validate it using validation.csv with validate_linear_fixture, then predict test_inputs.csv with predict_linear_fixture. Also attempt reuse with task='unrelated_protein_property' and verify that it is rejected. Save output.json with predictions (list of id/prediction), registered (boolean), and incompatible_reuse_rejected (boolean).",
                "expected": {
                    "registered": True,
                    "incompatible_reuse_rejected": True,
                    "predictions": [{"id": i, "prediction": intercept + a * x1 + b * x2} for i, x1, x2 in tests],
                },
                "required_tools": ["fit_linear_fixture", "validate_linear_fixture", "predict_linear_fixture"],
            }
        )
    return tasks


def matches(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(matches(a, e) for a, e in zip(actual, expected))
        )
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int, float)):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        )
    return actual == expected


def evaluate(task: dict, workspace: Path, run: dict, events_path: Path) -> dict:
    result = {"task_id": task["task_id"], "passed": False, "reason": "runtime_not_completed"}
    if run.get("status") != "completed":
        return result
    try:
        path = safe_child(workspace, "output.json")
        if path.stat().st_size > 1000000:
            raise ValueError("artifact too large")
        actual = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        result["artifact_sha256"] = sha256_file(path)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        called = {str(e.get("tool_name", "")).split("__")[-1] for e in events if e.get("kind") == "tool_call_start"}
        if not set(task["required_tools"]) <= called:
            return {**result, "reason": "required_tool_not_executed", "observed_tools": sorted(called)}
        if not matches(actual, task["expected"]):
            return {**result, "reason": "incorrect_artifact", "actual": actual}
        if task["kind"] == "sequence":
            if type(actual.get("molecular_weight")) not in (int, float) or actual["molecular_weight"] <= 0:
                return {**result, "reason": "missing_computed_molecular_weight"}
        if task["kind"] == "reuse":
            card = json.loads(safe_child(workspace, "registry/fixture.json").read_text())
            if card.get("status") != "validated" or card.get("validation_rmse", 1) > 1e-6:
                return {**result, "reason": "missing_validation_evidence"}
        return {**result, "passed": True, "reason": "artifact_and_execution_checks_passed"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**result, "reason": "invalid_artifact", "detail": f"{type(exc).__name__}: {exc}"}
