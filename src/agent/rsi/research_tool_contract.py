"""Typed contracts for small, independently checked scientific computations.

This version reuses the bounded JSON numerical ABI. Category and method-family
labels aid discovery; they never certify algorithmic novelty or tool counts.
"""

from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

CATEGORIES = {"data_curation", "sequence_analysis", "structure_analysis", "prediction", "evaluation", "workflow"}


def _schema(value):
    if not isinstance(value, dict) or "type" not in value:
        raise ValueError("An explicit JSON type is required at the interface root")
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded) > 32000:
        raise ValueError("Research tool schema is too large")

    def visit(node):
        if isinstance(node, dict):
            if any(k in node for k in ("$ref", "$dynamicRef", "$recursiveRef")):
                raise ValueError("Research tool schemas must be self-contained; references are not resolved")
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    Draft202012Validator.check_schema(value)
    return Draft202012Validator(value)


def validate_contract(contract, definition):
    fields = {
        "schema_version",
        "tool_id",
        "purpose",
        "category",
        "method_family",
        "inputs",
        "outputs",
        "scope",
        "limitations",
    }
    if not isinstance(contract, dict) or set(contract) != fields or contract["schema_version"] != "research_tool.v1":
        raise ValueError("Invalid research tool contract")
    if not isinstance(contract["tool_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", contract["tool_id"]):
        raise ValueError("Research tool ID must be a stable name")
    if contract["category"] not in CATEGORIES:
        raise ValueError("Unknown research tool category")
    for value in (contract["purpose"], contract["method_family"]):
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError("Purpose and method family must be bounded text")
    if (
        not isinstance(contract["limitations"], list)
        or not contract["limitations"]
        or any(not isinstance(item, str) or not item.strip() or len(item) > 4000 for item in contract["limitations"])
    ):
        raise ValueError("Explicit scientific limitations are required")
    if contract["scope"] != definition["scope"]:
        raise ValueError("Research tool contract must preserve admitted scientific scope")
    inputs, outputs = _schema(contract["inputs"]), _schema(contract["outputs"])
    for cases in definition["cases"].values():
        for case in cases:
            inputs.validate(case["inputs"])
            outputs.validate(case["expected"])
    return contract


def validate_result(contract, inputs, output):
    _schema(contract["inputs"]).validate(inputs)
    _schema(contract["outputs"]).validate(output)
