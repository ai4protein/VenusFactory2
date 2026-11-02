"""Domain feedback remains training-only and is replayable across MCP/host boundaries."""

import copy
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from test_context_feedback import candidate
from test_generated_context import CODE
from test_yeast_population_controls import example

from agent.rsi import context_feedback as legacy
from agent.rsi import context_feedback_profiles as feedback
from agent.rsi import expression_baselines, generated_context
from agent.rsi.campaign_evaluation import paired_design
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.tool_contracts import validate_generated_result
from agent.rsi.yeast_population_controls import controls, verify_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_paired_generated import make_bundle


def test_domain_controls_match_independent_existing_evaluator():
    scope, fit, score = example()
    actual = expression_baselines.score_rows(fit, score, scope)
    assert actual == controls(fit, score, scope)[-2:]
    verify_metrics(fit, score, actual)
    assert actual[0]["metrics"]["rmse"] < 1e-10
    altered = [{**r, "target": -12345.0} for r in score]
    assert [x["predictions"] for x in actual] == [
        x["predictions"] for x in expression_baselines.score_rows(fit, altered, scope)
    ]
    with pytest.raises(ValueError, match="overlap"):
        expression_baselines.score_rows(fit, fit[:2], scope)


def test_explicit_six_control_profile_preserves_legacy_packets():
    scope, fit, score = example()
    rows = fit + score
    metrics = candidate(rows)
    old = legacy.comparison(rows, scope, metrics)
    assert feedback.comparison(rows, scope, metrics) == old
    assert feedback.description(scope) == legacy.description(scope)
    assert old["attempted_controls"] == 4
    result = feedback.comparison(rows, scope, metrics, policy=feedback.EXPRESSION_POLICY)
    assert result == feedback.comparison(rows[::-1], scope, metrics, policy=feedback.EXPRESSION_POLICY)
    assert result["schema"] == feedback.EXPRESSION_POLICY
    assert result["attempted_controls"] == result["completed_controls"] == 6
    assert result["controls"][:4] == old["controls"]
    assert result["split"] == old["split"]
    assert all(result["implementation"][k] == v for k, v in old["implementation"].items())
    assert result["new_autonomous_models"] == 0
    assert all("predictions" not in x for x in result["controls"])
    task = {"plugin_id": generated_context.PLUGIN_ID, "scope": scope, feedback.FIELD: feedback.POLICY}
    feedback.verify_feedback(task, rows, {**metrics, "baseline_comparison": old})
    task[feedback.FIELD] = feedback.EXPRESSION_POLICY
    with pytest.raises(ValueError, match="numerical replay"):
        feedback.verify_feedback(task, rows, {**metrics, "baseline_comparison": old})
    feedback.verify_feedback(task, rows, {**metrics, "baseline_comparison": result})
    result["controls"][-1]["metrics"]["rmse"] += 1
    with pytest.raises(ValueError, match="numerical replay"):
        feedback.verify_feedback(task, rows, {**metrics, "baseline_comparison": result})


@pytest.mark.parametrize("bad", ["unknown", None, {}, False, 1])
def test_unknown_policy_rejected(bad):
    with pytest.raises(ValueError):
        feedback.policy_for({feedback.FIELD: bad}, plugin_id=generated_context.PLUGIN_ID)


def test_domain_profile_rejects_wrong_scope_plugin_and_reuse(tmp_path):
    scope, _, _ = example()
    declaration = {"scope": scope, feedback.FIELD: feedback.EXPRESSION_POLICY}
    assert feedback.policy_for(declaration, plugin_id=generated_context.PLUGIN_ID) == feedback.EXPRESSION_POLICY
    with pytest.raises(ValueError, match="contextual"):
        feedback.policy_for(declaration, plugin_id="generated_sequence.v1")
    wrong = copy.deepcopy(scope)
    wrong["covariates"].reverse()
    with pytest.raises(ValueError, match="covariates"):
        feedback.description(wrong, policy=feedback.EXPRESSION_POLICY)
    with pytest.raises(ValueError, match="only during development"):
        make_bundle(tmp_path, "reuse", {}, scope, {}, plugin_id=generated_context.PLUGIN_ID,
                    training_baseline_feedback=feedback.EXPRESSION_POLICY)
    protocol = {"model_plugin_id": generated_context.PLUGIN_ID, feedback.FIELD: feedback.EXPRESSION_POLICY}
    feedback.verify_bundle({"phase": "reuse", "plugin_id": generated_context.PLUGIN_ID}, protocol)
    with pytest.raises(ValueError, match="differs"):
        feedback.verify_bundle({"phase": "develop", "plugin_id": generated_context.PLUGIN_ID}, protocol)
    with pytest.raises(ValueError, match="binding changed"):
        feedback.verify_goal({"definition": {feedback.FIELD: feedback.POLICY}}, protocol)


def test_matched_feedback_comparison_allows_domain_intervention_only():
    scope, _, _ = example()
    left = {"id": "a", "task": {"plugin_id": generated_context.PLUGIN_ID, "scope": scope},
            feedback.FIELD: feedback.POLICY, "generations": 2}
    right = {**left, "id": "b", feedback.FIELD: feedback.EXPRESSION_POLICY}
    pair = [{"id": "domain", "left": "a", "right": "b", "factor": feedback.FIELD}]
    paired_design({"a": left, "b": right}, pair)
    with pytest.raises(ValueError):
        paired_design({"a": left, "b": {**right, "generations": 3}}, pair)
    with pytest.raises(ValueError):
        paired_design({"a": left, "b": {**right, feedback.FIELD: feedback.POLICY}}, pair)


@pytest.mark.asyncio
async def test_real_mcp_domain_feedback_and_joint_tampering_rejected(tmp_path):
    scope, fit, score = example()
    rows = fit + score
    work = tmp_path / "workspace"
    work.mkdir()
    atomic_json(work / "training.json", rows)
    versions, version, caps = make_bundle(
        tmp_path / "capability", "develop", {"training.json": rows}, scope, {"provider": "synthetic"},
        plugin_id=generated_context.PLUGIN_ID, training_baseline_feedback=feedback.EXPRESSION_POLICY,
    )
    assert {"expression_baselines.py", "context_feedback_profiles.py"} <= {p.name for p in caps.iterdir()}
    config = {"mcpServers": {"context": {"command": sys.executable,
               "args": ["-u", str(caps / "generated_tools.py")], "env": {"VENUS_RSI_WORKSPACE": str(work)}}}}
    async with Client(config) as client:
        async def call(name, args):
            result = await client.call_tool(name, args)
            validate_generated_result(name, result.data, workspace=work, capability=caps)
            return result.data

        info = await call("inspect_model_code_task", {})
        assert info[feedback.FIELD]["policy"] == feedback.EXPRESSION_POLICY
        assert len(info[feedback.FIELD]["methods"]) == 6
        trial = await call("evaluate_model_code", {"source_code": CODE, "rationale": "synthetic test"})
        assert trial["training_only_feedback"]["baseline_comparison"]["attempted_controls"] == 6
        submitted = await call("submit_model_code", {"source_sha256": trial["source_sha256"], "rationale": "tested"})
        assert submitted["training_rows"] == 40 and not submitted["registered"]
    versions.verify(version)
    changed = copy.deepcopy(trial)
    changed["training_only_feedback"]["baseline_comparison"]["controls"][-1]["metrics"]["rmse"] += 1
    path = work / "code_attempt_1.json"
    record = json.loads(path.read_text())
    record["training_feedback"] = changed["training_only_feedback"]
    atomic_json(path, record)
    ledger = json.loads((work / ".model_development.json").read_text())
    ledger["attempts"][0]["artifact_sha256"] = sha256_file(path)
    provenance_path = work / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["training_feedback"] = changed["training_only_feedback"]
    atomic_json(provenance_path, provenance)
    ledger["submission"]["artifact_sha256"]["source_provenance.json"] = sha256_file(provenance_path)
    atomic_json(work / ".model_development.json", ledger)
    atomic_json(work / ("code_ledger_" + digest(ledger) + ".json"), ledger)
    with pytest.raises(ValueError, match="numerical replay"):
        validate_generated_result("evaluate_model_code", changed, workspace=work, capability=caps)
