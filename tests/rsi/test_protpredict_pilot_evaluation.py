import hashlib
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/rsi/evaluate_protpredict_pilot.py"
spec = importlib.util.spec_from_file_location("protpredict_evaluation_controller", SOURCE)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.fixture
def cohort():
    def rows(prefix, n):
        return [
            {
                "id": f"{prefix}{i}",
                "protein": "P_TEST",
                "group": f"{prefix}_gene_{i}",
                "values": [float(j + 1) for j in range(19)],
                "target": float(i + 1),
            }
            for i in range(n)
        ]

    scope = {"protein": "P_TEST", "covariates": [{"name": f"AA{i}"} for i in range(19)]}
    return rows("fit", 600), rows("future", 200), scope


def test_worker_never_receives_outer_labels(cohort):
    training, scoring, scope = cohort
    before = evaluator.worker_request(training, scoring, scope, {"frozen": 1})
    for row in scoring:
        row["target"] = 99999999
    after = evaluator.worker_request(training, scoring, scope, {"frozen": 1})
    assert before == after
    assert set(after["scoring"]) == {"x", "ids"}
    assert set(after["training"]) == {"x", "y", "groups"}


@pytest.mark.parametrize(
    "fault",
    [
        "row_count",
        "id_duplicate",
        "group_duplicate",
        "id_overlap",
        "group_overlap",
        "protein",
        "features",
        "scope_features",
    ],
)
def test_invalid_frozen_cohorts_fail(cohort, fault):
    training, scoring, scope = cohort
    if fault == "row_count":
        scoring.pop()
    if fault == "id_duplicate":
        scoring[1]["id"] = scoring[0]["id"]
    if fault == "group_duplicate":
        scoring[1]["group"] = scoring[0]["group"]
    if fault == "id_overlap":
        scoring[0]["id"] = training[0]["id"]
    if fault == "group_overlap":
        scoring[0]["group"] = training[0]["group"]
    if fault == "protein":
        scoring[0]["protein"] = "DIFFERENT"
    if fault == "features":
        training[0]["values"].pop()
    if fault == "scope_features":
        scope["covariates"].pop()
    with pytest.raises(ValueError):
        evaluator.worker_request(training, scoring, scope, {})


@pytest.mark.parametrize("fault", ["order", "missing", "nan", "bool", "extra_label"])
def test_invalid_prediction_witness(cohort, fault):
    _, scoring, _ = cohort
    predictions = [{"id": r["id"], "prediction": 1.0} for r in scoring]
    if fault == "order":
        predictions.reverse()
    if fault == "missing":
        predictions.pop()
    if fault == "nan":
        predictions[0]["prediction"] = float("nan")
    if fault == "bool":
        predictions[0]["prediction"] = True
    if fault == "extra_label":
        predictions[0]["target"] = 1.0
    with pytest.raises(ValueError):
        evaluator.prediction_witness(scoring, predictions, lambda x: "digest")


def test_prediction_witness_retains_identity_without_target(cohort):
    _, scoring, _ = cohort
    predictions = [{"id": r["id"], "prediction": 1.0} for r in scoring]
    result = evaluator.prediction_witness(scoring, predictions, lambda x: "numeric-input-hash")
    assert result[0] == {
        "id": "future0",
        "protein": "P_TEST",
        "group": "future_gene_0",
        "values_sha256": "numeric-input-hash",
        "prediction": 1.0,
    }


@pytest.mark.parametrize("fault", [None, "definition", "plans", "owner", "inflight", "policy", "incomplete"])
def test_global_terminal_gate(fault):
    definition = {"campaign_definition_sha256": "sha", "plans": ["frozen"], "execution_policy": "all"}
    state = {
        "definition_sha256": "sha",
        "plans": ["frozen"],
        "owner": None,
        "inflight": None,
        "definition": {"policy": "all"},
    }
    complete = True
    if fault == "definition":
        state["definition_sha256"] = "changed"
    if fault == "plans":
        state["plans"] = []
    if fault == "owner":
        state["owner"] = {"pid": 123}
    if fault == "inflight":
        state["inflight"] = {"allocation": "unfinished"}
    if fault == "policy":
        state["definition"]["policy"] = "until_target"
    if fault == "incomplete":
        complete = False
    args = (definition, state, lambda x: x["policy"], lambda x: {"execution_complete": complete}, "all")
    if fault:
        with pytest.raises(ValueError):
            evaluator.require_terminal(*args)
    else:
        evaluator.require_terminal(*args)


def test_bound_artifact_change_fails(tmp_path):
    path = tmp_path / "params.json"
    path.write_text('{"learning_rate":0.01}')
    binding = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert evaluator.bound(binding) == path
    path.write_text('{"learning_rate":0.1}')
    with pytest.raises(ValueError, match="changed"):
        evaluator.bound(binding)


def test_symlink_evidence_fails(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="Symlinked"):
        evaluator.read(link)


def definition_and_protocol():
    allocations = {}
    for i in range(4):
        allocations[str(i)] = {
            "task": {
                "id": f"task{i}",
                "plugin_id": "generated_context.v2",
                "scope": {
                    "covariates": [None] * 19,
                    "grouping": {"level": "deleted_gene"},
                    "context": {"organism": "Saccharomyces cerevisiae, laboratory S288c-derived strains"},
                },
            }
        }
    return {"allocations": allocations, "pairs": []}, {"selection": {"selected_tasks": [f"task{i}" for i in range(4)]}}


@pytest.mark.parametrize(
    "fault", [None, "duplicate", "missing", "paired", "wrong_plugin", "wrong_organism", "wrong_group", "wrong_features"]
)
def test_only_preselected_complete_task_set(fault):
    definition, protocol = definition_and_protocol()
    task = definition["allocations"]["0"]["task"]
    if fault == "duplicate":
        protocol["selection"]["selected_tasks"][0] = "task1"
    if fault == "missing":
        del definition["allocations"]["0"]
    if fault == "paired":
        definition["pairs"] = [{"left": "0", "right": "1"}]
    if fault == "wrong_plugin":
        task["plugin_id"] = "generated_context.v1"
    if fault == "wrong_organism":
        task["scope"]["context"]["organism"] = "unknown"
    if fault == "wrong_group":
        task["scope"]["grouping"]["level"] = "row"
    if fault == "wrong_features":
        task["scope"]["covariates"].pop()
    if fault:
        with pytest.raises(ValueError):
            evaluator.verify_selected(definition, protocol)
    else:
        evaluator.verify_selected(definition, protocol)
