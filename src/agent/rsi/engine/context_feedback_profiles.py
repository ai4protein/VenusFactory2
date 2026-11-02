"""Explicit feedback interventions while retaining byte-identical legacy replay."""

import hashlib
from pathlib import Path

try:
    from agent.rsi import context_feedback as legacy
    from agent.rsi import expression_baselines, generated_context
    from .protein_model import digest
except ImportError:  # Frozen standalone MCP capability.
    import context_feedback as legacy
    import expression_baselines
    import generated_context
    from protein_model import digest

FIELD = legacy.FIELD
POLICY = legacy.POLICY
EXPRESSION_POLICY = "yeast_expression_baselines.v1"
POLICIES = ("none", POLICY, EXPRESSION_POLICY)


def policy_for(declaration, *, plugin_id=None):
    value = declaration.get(FIELD, "none")
    if value != EXPRESSION_POLICY:
        return legacy.policy_for(declaration, plugin_id=plugin_id)
    legacy.policy_for({**declaration, FIELD: POLICY}, plugin_id=plugin_id)
    if "scope" in declaration:
        expression_baselines.require_scope(declaration["scope"])
    elif "task" in declaration:
        expression_baselines.require_scope(declaration["task"]["scope"])
    return value


def verify_goal(state, protocol):
    policy = policy_for(protocol)
    if state["definition"].get(FIELD, "none") != policy:
        raise ValueError("Goal training-baseline feedback binding changed")
    return policy


def verify_bundle(task, protocol):
    expected = policy_for(protocol) if task["phase"] == "develop" else "none"
    if policy_for(task) != expected:
        raise ValueError("Episode training-baseline feedback policy differs from goal")


def description(scope, *, policy=POLICY):
    if policy == POLICY:
        return legacy.description(scope)
    if policy != EXPRESSION_POLICY:
        raise ValueError("Unsupported enabled training-baseline feedback policy")
    expression_baselines.require_scope(scope)
    result = legacy.description(scope)
    result.update(policy=policy, methods=result["methods"] + list(expression_baselines.METHODS))
    result["domain_controls"] = {
        "transcript_only_ols": {"covariates": ["author_log2_tpm"], "intercept": True},
        "culture_metadata_ols": {"covariates": ["od_rna", "od_protein", "ploidy"], "intercept": True},
    }
    return result


def comparison(rows, scope, candidate_metrics, *, policy=POLICY):
    if policy == POLICY:
        return legacy.comparison(rows, scope, candidate_metrics)
    if policy != EXPRESSION_POLICY:
        raise ValueError("Unsupported enabled training-baseline feedback policy")
    expression_baselines.require_scope(scope)
    result = legacy.comparison(rows, scope, candidate_metrics)
    fitting, scoring = generated_context.development_split(rows)
    for outcome in expression_baselines.score_rows(fitting, scoring, scope):
        item = {key: value for key, value in outcome.items() if key != "predictions"}
        item["predictions_sha256"] = digest(outcome["predictions"])
        result["controls"].append(item)
    complete = [x for x in result["controls"] if x["status"] == "completed"]
    best = min(complete, key=lambda x: (x["metrics"]["rmse"], x["method"])) if complete else None
    result.update(
        schema=policy,
        attempted_controls=len(result["controls"]),
        completed_controls=len(complete),
        best_completed_control=best["method"] if best else None,
        candidate_minus_best_rmse=candidate_metrics["rmse"] - best["metrics"]["rmse"] if best else None,
    )
    for path in (Path(__file__), Path(expression_baselines.__file__)):
        result["implementation"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_feedback(task, rows, feedback):
    policy = policy_for(task)
    if policy != EXPRESSION_POLICY:
        return legacy.verify_feedback(task, rows, feedback)
    if feedback.get("baseline_comparison") != comparison(rows, task["scope"], feedback, policy=policy):
        raise ValueError("Training-baseline feedback differs from numerical replay")
