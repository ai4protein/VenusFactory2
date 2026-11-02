"""Small evidence-bound artifact projection prototype, not a live MCP plugin.

A public output contract selects fields from completed tool results. It never
receives scorer expectations or fills missing values with fabricated defaults.
"""

from __future__ import annotations

import json
from typing import Any


def successful_results(events: list[dict]) -> dict[str, Any]:
    started = set()
    results = {}
    for event in events:
        cid = event.get("tool_call_id")
        if event.get("kind") == "tool_call_start" and cid:
            if cid in started:
                raise ValueError("duplicate tool call id")
            started.add(cid)
        elif event.get("kind") == "tool_result" and cid in started:
            if event.get("is_error"):
                continue
            try:
                value = json.loads(
                    event.get("output", ""), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
                )
            except (TypeError, ValueError):
                continue
            if cid in results:
                raise ValueError("duplicate successful tool result")
            results[cid] = value
    return results


def project(contract: dict, results: dict[str, Any], *, max_depth: int = 20) -> tuple[Any, list[str]]:
    used = set()

    def resolve(node, depth):
        if depth > max_depth or not isinstance(node, dict):
            raise ValueError("invalid projection structure")
        op = node.get("op")
        if op == "ref":
            cid = node["call_id"]
            if cid not in results:
                raise ValueError("referenced result is absent or unsuccessful")
            value = results[cid]
            used.add(cid)
            for key in node.get("path", []):
                if not (isinstance(value, dict) and isinstance(key, str) and key in value):
                    raise ValueError("required result field is absent")
                value = value[key]
            return value
        if op == "object":
            fields = node.get("fields")
            if not isinstance(fields, dict):
                raise ValueError("object fields must be a mapping")
            return {k: resolve(v, depth + 1) for k, v in fields.items()}
        if op == "equals":
            actual = resolve(node["value"], depth + 1)
            wanted = node["expected_literal"]
            if isinstance(wanted, (dict, list)):
                raise ValueError("predicates accept scalar literals only")
            return type(actual) is type(wanted) and actual == wanted
        if op == "all":
            values = [resolve(x, depth + 1) for x in node["values"]]
            if not values or any(type(x) is not bool for x in values):
                raise ValueError("all requires nonempty boolean predicates")
            return all(values)
        raise ValueError("unknown projection operator")

    value = resolve(contract, 0)
    value = json.loads(json.dumps(value, allow_nan=False))
    return value, sorted(used)
