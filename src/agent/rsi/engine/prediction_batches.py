"""Frozen per-observation inference review, separate from exact replay and scoring."""

import math
from copy import deepcopy

from .protein_model import digest

FIELD = "prediction_batch_review"


def contract():
    return {
        "schema": 1,
        "semantics": "independent_rows",
        "rtol": 1e-6,
        "atol": 1e-12,
        "probes": "reverse_sample_single_v1",
    }


def validate_contract(value):
    if not isinstance(value, dict) or digest(value) != digest(contract()):
        raise ValueError("Unsupported prediction batch review contract")


def apply_policy(policy, protocol):
    if FIELD not in protocol:
        return dict(policy)
    validate_contract(protocol[FIELD])
    return {**policy, "schema": 2, FIELD: deepcopy(protocol[FIELD])}


def require_bound_policy(policy, protocol):
    expected = protocol.get(FIELD)
    if expected is None:
        if FIELD in protocol or FIELD in policy or policy.get("schema") != 1:
            raise ValueError("Prediction batch review policy changed")
    else:
        validate_contract(expected)
        if policy.get("schema") != 2 or policy.get(FIELD) != expected:
            raise ValueError("Prediction batch review policy changed")


def freeze_goal(store):
    """Bind the review before the first execution, using the goal's locked store."""
    review = contract()

    def bind(state):
        if (
            state["status"] != "active"
            or state["cycles"]
            or state["inflight"]
            or state["progress"]
            or state["assessment"] is not None
            or any(state["usage"].values())
        ):
            raise ValueError("Prediction batch review must be frozen before execution")
        definition = state["definition"]
        if FIELD in definition:
            validate_contract(definition[FIELD])
        definition[FIELD] = deepcopy(review)
        state["definition_sha256"] = digest(definition)

    store.mutate(bind)
    return review


def verify_goal(state, protocol):
    definition = state["definition"]
    if FIELD in protocol or FIELD in definition:
        validate_contract(protocol.get(FIELD))
        if definition.get(FIELD) != protocol[FIELD]:
            raise ValueError("Frozen prediction batch review changed")


def append_prompt(prompt, protocol, phase):
    if phase != "develop" or FIELD not in protocol:
        return prompt
    validate_contract(protocol[FIELD])
    return prompt + (
        "\n\nPrediction contract: return one independent result per observation, using fitted training "
        "statistics without mutating model state. Host review also checks reversed, sampled and single-row "
        "queries with rtol=1e-6 and atol=1e-12; same-batch replay must remain exact."
    )


def probe_plan(rows):
    if type(rows) is not int or rows < 1:
        raise ValueError("Prediction batch review requires reference rows")
    full = list(range(rows))
    candidates = [("reversed", full[::-1]), ("sampled", sorted({0, rows // 2, rows - 1})), ("single", [rows // 2])]
    seen = {tuple(full)}
    result = []
    for name, indices in candidates:
        if tuple(indices) not in seen:
            result.append({"name": name, "indices": indices})
            seen.add(tuple(indices))
    return result


def collect(plugin, model, inputs, scope):
    """At most three additional predictions; no targets or final-test data."""
    probes = []
    for plan in probe_plan(len(inputs)):
        rows = [deepcopy(inputs[i]) for i in plan["indices"]]
        probe = {**plan, "inputs_sha256": digest(rows)}
        candidate = deepcopy(model)
        before = digest(candidate)
        try:
            predictions = plugin.predict(candidate, deepcopy(rows), deepcopy(scope))
            digest(predictions)
            if digest(candidate) != before:
                probe["error"] = "model_state_changed"
            else:
                probe["predictions"] = predictions
        except Exception as exc:
            # Exception messages can contain paths or data; only the type is diagnostic.
            probe["error"] = type(exc).__name__
        probes.append(probe)
    return {"inputs": deepcopy(inputs), "probes": probes}


def _compare(reference, actual, policy):
    if not isinstance(actual, list) or len(actual) != len(reference):
        return {"passed": False, "rows": len(reference), "reason": "prediction_rows_changed"}
    absolute, relative, overflow, mismatches = 0.0, 0.0, False, 0

    def number(a, b):
        nonlocal absolute, relative, overflow
        if type(a) not in (float, int) or type(b) not in (float, int):
            return False
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        difference = abs(a - b)
        if math.isfinite(difference):
            absolute = max(absolute, difference)
            scale = max(abs(a), abs(b))
            relative = max(relative, difference / scale if scale else 0.0)
        else:
            overflow = True
        return math.isclose(a, b, rel_tol=policy["rtol"], abs_tol=policy["atol"])

    def values(a, b):
        if isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                return False
            return all([values(x, y) for x, y in zip(a, b, strict=True)])
        return number(a, b)

    for expected, observed in zip(reference, actual, strict=True):
        if not isinstance(expected, dict) or not isinstance(observed, dict) or set(expected) != set(observed):
            mismatches += 1
            continue
        metadata = {k: v for k, v in expected.items() if k not in {"prediction", "probabilities"}}
        other = {k: v for k, v in observed.items() if k not in {"prediction", "probabilities"}}
        ok = digest(metadata) == digest(other) and "prediction" in expected
        if "probabilities" in expected:
            # The class is categorical; probability vectors receive numeric tolerance.
            ok = (digest(expected["prediction"]) == digest(observed["prediction"])) and ok
            ok = values(expected["probabilities"], observed["probabilities"]) and ok
        else:
            ok = number(expected.get("prediction"), observed.get("prediction")) and ok
        mismatches += not ok
    return {
        "passed": mismatches == 0,
        "rows": len(reference),
        "mismatched_rows": mismatches,
        "max_absolute_difference": None if overflow else absolute,
        "max_relative_difference": None if overflow else relative,
        "difference_overflow": overflow,
    }


def summarize(predictions, payload, policy, plugin_id):
    from agent.rsi.datasets.model_tasks import inference_inputs

    validate_contract(policy)
    if not isinstance(payload, dict) or set(payload) != {"inputs", "probes"}:
        raise ValueError("Invalid prediction batch evidence")
    inputs, probes = payload["inputs"], payload["probes"]
    if (
        not isinstance(inputs, list)
        or len(inputs) != len(predictions)
        or digest(inputs) != digest(inference_inputs(inputs, plugin_id))
    ):
        raise ValueError("Prediction batch evidence requires canonical unlabeled inputs")
    plans = probe_plan(len(inputs))
    if not isinstance(probes, list) or len(probes) != len(plans):
        raise ValueError("Prediction batch probes are missing")
    comparisons = []
    for probe, plan in zip(probes, plans, strict=True):
        expected_keys = {"name", "indices", "inputs_sha256"}
        if (
            not isinstance(probe, dict)
            or set(probe) not in (expected_keys | {"predictions"}, expected_keys | {"error"})
            or any(probe.get(key) != value for key, value in plan.items())
            or probe["inputs_sha256"] != digest([inputs[i] for i in plan["indices"]])
        ):
            raise ValueError("Prediction batch probe binding changed")
        if "error" in probe:
            if not isinstance(probe["error"], str) or not probe["error"]:
                raise ValueError("Invalid prediction batch error")
            check = {"passed": False, "rows": len(plan["indices"]), "reason": probe["error"]}
        else:
            check = _compare([predictions[i] for i in plan["indices"]], probe["predictions"], policy)
        comparisons.append({"name": plan["name"], **check})
    return {
        "schema": 1,
        "contract": policy,
        "reference_rows": len(inputs),
        "reference_inputs_sha256": digest(inputs),
        "reference_predictions_sha256": digest(predictions),
        "probes_sha256": digest(probes),
        "probes": comparisons,
        "passed": bool(comparisons) and all(p["passed"] for p in comparisons),
    }
