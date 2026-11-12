"""Measured experiment identity is distinct from file names and serialized metadata."""
import json

import pytest

from test_research_training import BASELINE, LINEAR, service_at


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 19])
async def test_duplicate_revision_cannot_be_resubmitted_as_a_sibling(tmp_path, seed):
    service = await service_at(tmp_path)
    baseline = await service.train_research_model("mean", "algorithm.py", "evaluation.json", "Software fixture")
    parent = baseline["experiment"]["id"]
    await service.write_research_file("linear.py", LINEAR)
    revision = await service.train_research_model("linear", "linear.py", "", "Measured revision", parent_experiment_id=parent)
    existing = revision["experiment"]["id"]
    frozen = {p: p.read_bytes() for p in service.development.root.glob("experiments/*/receipt.json")}
    await service.write_research_file("renamed.py", LINEAR + "# Same fitted algorithm under a different name\n")
    with pytest.raises(ValueError, match="already measured as experiment " + existing):
        await service.train_research_model("renamed", "renamed.py", "", "Duplicate sibling",
                                           parent_experiment_id=parent, seed=seed)
    assert len(service.development.records()) == 2
    assert all(path.read_bytes() == raw for path, raw in frozen.items())
    attempts = [json.loads(path.read_text()) for path in service.development.root.glob("training/*/receipt.json")]
    failed = next(row for row in attempts if row["status"] == "failed")
    assert existing in failed["error"] and failed["stage"] == "fitting"
    assert not service.proposals


@pytest.mark.asyncio
async def test_seed_metadata_changes_do_not_hide_identical_predictions(tmp_path):
    service = await service_at(tmp_path)
    code = BASELINE.replace("return {\"mean\":", "return {\"seed\": seed, \"mean\":")
    code = code.replace('return {"value":model["mean"]}', 'return {"value":model["mean"],"diagnostic_seed":model["seed"]}')
    await service.write_research_file("algorithm.py", code)
    baseline = await service.train_research_model("seeded", "algorithm.py", "evaluation.json", "Software fixture", seed=0)
    parent = baseline["experiment"]["id"]
    revised = await service.train_research_model("seeded again", "algorithm.py", "", "Seed-only change",
                                                 parent_experiment_id=parent, seed=19)
    before, after = baseline["experiment"], revised["experiment"]
    # The metadata and full returned objects change, but the scored values do not.
    assert before["predictions_sha256"] != after["predictions_sha256"]
    assert before["prediction_values_sha256"] == after["prediction_values_sha256"]
    assert after["prediction_equivalence"] == {"split":"validation", "method":"exact_declared_prediction_values",
                                               "matched_experiment_ids":[parent]}
    assert "exactly match" in revised["next_step"]
    summary = service.development.summary()["generations"][-1]
    assert summary["prediction_equivalence"] == after["prediction_equivalence"]
    assert not service.proposals


@pytest.mark.asyncio
async def test_equal_scores_do_not_imply_equal_predictions(tmp_path):
    service = await service_at(tmp_path)
    expression = 'model["slope"]*inputs["x"]+model["intercept"]'
    await service.write_research_file("low.py", LINEAR.replace(expression, expression+'-1'))
    first = await service.train_research_model("low", "low.py", "evaluation.json", "Software fixture")
    await service.write_research_file("high.py", LINEAR.replace(expression, expression+'+1'))
    second = await service.train_research_model("high", "high.py", "", "Different equal-error predictions",
                                                parent_experiment_id=first["experiment"]["id"])
    assert first["experiment"]["validation"]["score"] == second["experiment"]["validation"]["score"]
    assert first["experiment"]["prediction_values_sha256"] != second["experiment"]["prediction_values_sha256"]
    assert second["experiment"]["prediction_equivalence"]["matched_experiment_ids"] == []


def test_equivalence_uses_exact_numeric_values_and_preserves_label_types():
    from agent.harness.research_evaluation import prediction_values_digest as identity
    assert identity([1, 0, -0.0, {"b":0.5, "a":1}]) == identity([1.0, 0.0, 0, {"a":1.0, "b":0.5}])
    assert identity([1.0]) != identity([1.0 + 1e-12])
    assert identity([1]) != identity(["1"])
    assert identity([2**53 + 1]) != identity([float(2**53 + 1)])
