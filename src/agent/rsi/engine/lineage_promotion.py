"""Versioned numerical guard for development-score promotion, not efficacy proof."""

import math


def new_evolution(generations):
    if type(generations) is not int or not 1 <= generations <= 3:
        raise ValueError("Invalid generation count")
    return {
        "schema": 3,
        "generations": generations,
        "promotion": "Development skill gain beyond frozen numerical tolerance",
        "promotion_guard": {"schema": "numerical_margin.v1", "atol": 1e-10, "rtol": 1e-10},
        "scope": "Adaptive selection; numerical distinctness is not held-out efficacy or practical importance",
    }


def validate_evolution(protocol):
    evolution = protocol.get("evolution")
    if not isinstance(evolution, dict):
        raise ValueError("Missing evolution protocol")
    schema = evolution.get("schema", 1)
    if type(schema) is not int or schema not in {1, 2, 3}:
        raise ValueError("Unsupported evolution schema")
    if schema < 3:
        if "promotion_guard" in evolution:
            raise ValueError("Legacy evolution cannot declare a new promotion guard")
        return None
    guard = evolution.get("promotion_guard")
    expected = new_evolution(2)["promotion_guard"]
    if guard != expected or any(type(guard[k]) is not float for k in ("atol", "rtol")):
        raise ValueError("Missing or altered frozen numerical promotion guard")
    return guard


def gain_error(protocol, previous, current):
    """Legacy replay keeps its original strict comparison; v3 rejects roundoff."""
    guard = validate_evolution(protocol)
    if guard is None:
        return None
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in (previous, current)):
        raise ValueError("Promotion requires finite numerical scores")
    tolerance = max(guard["atol"], guard["rtol"] * max(abs(previous), abs(current)))
    if current - previous <= tolerance:
        return "Development gain does not exceed the frozen numerical tolerance"
    return None
