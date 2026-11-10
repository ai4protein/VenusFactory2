"""Task admission tests use synthetic proteins; they are not model discoveries."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_generated_models import CODE, SCOPE, TRAIN, VALID

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import model_tasks as tasks
from agent.rsi.engine.generated_model import fit_candidate, predict
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.storage import atomic_json, sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal, prepare


@pytest.fixture
def task(tmp_path):
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic independent reference")
    data = {
        "training": TRAIN,
        "validation": VALID,
        "future": [{"id": "HI", "sequence": "HI", "target": 19.0}, {"id": "IH", "sequence": "IH", "target": 20.0}],
    }
    paths = {}
    for phase, rows in data.items():
        path = tmp_path / (phase + ".json")
        atomic_json(path, rows)
        paths[phase] = {"path": str(path), "sha256": sha256_file(path)}
    value = {
        "schema": 1,
        "id": "synthetic_activity",
        "stage": "activity",
        "plugin_id": "generated_python.v1",
        "scope": SCOPE,
        "paper": {"title": "Synthetic task fixture", "doi": "test:fixture"},
        "references": [{"path": str(reference), "sha256": sha256_file(reference)}],
        "data": paths,
        "minimum_skill": 0.2,
    }
    return value


def prepared(tmp_path, task):
    declaration = tmp_path / "task.json"
    atomic_json(declaration, task)
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, model_calls=4, tool_calls=3),
    )
    profile_path = tmp_path / "profile.json"
    atomic_json(profile_path, profile.public())
    args = SimpleNamespace(
        profile=profile_path,
        objective="Learn this admitted protein task",
        task_definition=declaration,
        output=tmp_path / "goal",
        case="not_a_builtin_case",
        source_cache=None,
        max_cycles=4,
        model_calls=16,
        tool_calls=12,
        wall_seconds=300,
    )
    store = prepare(args)
    return store, GeneratedModelGoal(store, None)


def test_real_training_acceptance_and_fresh_inference_from_arbitrary_task(tmp_path, task):
    store, driver = prepared(tmp_path, task)
    assert driver.protocol["case"] == task["id"]
    from agent.rsi.engine import prediction_batches
    assert driver.protocol[prediction_batches.FIELD] == prediction_batches.contract()
    assert store.read()["definition"][prediction_batches.FIELD] == prediction_batches.contract()
    assert driver.acceptance_policy({})["schema"] == 2
    assert driver.acceptance_policy({})["minimum_skill"] == 0.2
    model = fit_candidate(driver.data["training"], driver.scope, {"source_code": CODE, "seed": 1234})
    lc = ModelLifecycle(store.root / "controller", driver.acceptance_policy({}))
    atomic_json(lc.root / "candidate.json", model)
    mid = lc.submit(lc.root / "candidate.json", session_id="synthetic_development", mode="science_expert")
    assert lc.assess(mid, training=driver.data["training"], validation=driver.data["validation"])["passed"]
    inputs = driver.inference_inputs()
    assert all(set(row) == {"id", "sequence"} for row in inputs)
    assert (
        lc.record_reuse(
            mid,
            session_id="synthetic_reuse",
            mode="science_expert",
            inputs=inputs,
            predictions=predict(model, inputs, driver.scope),
            scope=driver.scope,
        )["rows"]
        == 2
    )
    assert GeneratedModelGoal(store, None).scope == task["scope"]


@pytest.mark.parametrize(
    "attack",
    [
        "id_overlap",
        "sequence_overlap",
        "changed_file",
        "changed_reference",
        "missing_phase",
        "threshold",
        "unknown_plugin",
        "empty_unit",
        "short_training",
        "symlink",
    ],
)
def test_invalid_or_changed_task_rejected_before_goal_allocation(tmp_path, task, attack):
    value = copy.deepcopy(task)
    if attack in {"id_overlap", "sequence_overlap", "short_training"}:
        phase = "training" if attack == "short_training" else "future"
        path = Path(value["data"][phase]["path"])
        rows = json.loads(path.read_text())
        if attack == "short_training":
            rows = rows[:2]
        else:
            rows[0]["id" if attack == "id_overlap" else "sequence"] = TRAIN[0][
                "id" if attack == "id_overlap" else "sequence"
            ]
        atomic_json(path, rows)
        value["data"][phase]["sha256"] = sha256_file(path)
    elif attack == "changed_file":
        Path(value["data"]["future"]["path"]).write_text("[]")
    elif attack == "changed_reference":
        Path(value["references"][0]["path"]).write_text("changed")
    elif attack == "missing_phase":
        del value["data"]["future"]
    elif attack == "threshold":
        value["minimum_skill"] = float("inf")
    elif attack == "unknown_plugin":
        value["plugin_id"] = "unverified.plugin"
    elif attack == "empty_unit":
        value["scope"]["unit"] = ""
    else:
        alias = tmp_path / "alias.json"
        alias.symlink_to(value["data"]["future"]["path"])
        value["data"]["future"]["path"] = str(alias)
    with pytest.raises(ValueError):
        tasks.load_task(value)
    assert not (tmp_path / "goal").exists()


@pytest.mark.parametrize("attack", ["threshold", "manifest", "data"])
def test_goal_reopen_revalidates_task_admission(tmp_path, task, attack):
    store, driver = prepared(tmp_path, task)
    if attack == "threshold":
        protocol = driver.protocol
        protocol["model_task_minimum_skill"] = 0.0
        atomic_json(store.root / "protocol.json", protocol)
    elif attack == "manifest":
        task["minimum_skill"] = 0.0
        atomic_json(store.root / "model_task.json", task)
    else:
        Path(task["data"]["future"]["path"]).write_text("[]")
    with pytest.raises(ValueError):
        GeneratedModelGoal(store, None)


def test_feature_inputs_keep_representation_and_exclude_labels(task):
    rows = [{"id": "a", "sequence": "AA", "features": [0.2, 0.4], "target": 999}]
    assert tasks.inference_inputs(rows, "generated_features.v1") == [
        {"id": "a", "sequence": "AA", "features": [0.2, 0.4]}
    ]


def test_evolution_uses_the_admitted_initial_gate(tmp_path, task):
    store, _ = prepared(tmp_path, task)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["evolution"] = {"schema": 2, "generations": 2}
    atomic_json(store.root / "protocol.json", protocol)
    driver = EvolvingModelGoal(store, None)
    assert driver.acceptance_policy({"progress": {}})["minimum_skill"] == 0.2
    checks = driver.evaluate({"progress": {"generations": []}})["criteria"]
    assert not any(c["passed"] for c in checks.values())


@pytest.mark.parametrize(
    "plugin_id",
    sorted(
        set(tasks.PLUGINS)
        - {
            "generated_context.v1",
            "generated_context.v2",
            "generated_context.v3",
            "generated_assay.v1",
            "generated_antibody.v1",
        }
    ),
)
def test_sequence_model_interfaces_accept_source_bound_task_data(tmp_path, task, plugin_id):
    task["plugin_id"] = plugin_id
    task["scope"] = dict(SCOPE)
    if plugin_id == "generated_sequence.v1":
        del task["scope"]["sequence_length"]
        task["scope"].update(minimum_sequence_length=1, maximum_sequence_length=2048)
    if "classifier" in plugin_id:
        task["scope"]["class_labels"] = ["low", "high"]
    if plugin_id == "generated_features.v1":
        task["scope"]["representation"] = {"name": "synthetic_features", "dimensions": 2, "encoder_sha256": "0" * 64}
    amino = "ACDEFGHIKLMNPQRSTVWY"
    for phase, offset, count in [("training", 0, 20), ("validation", 20, 4), ("future", 24, 4)]:
        rows = []
        for i in range(offset, offset + count):
            row = {
                "id": str(i),
                "sequence": amino[i // 20] + amino[i % 20],
                "target": i % 2 if "classifier" in plugin_id else float(i),
            }
            if plugin_id == "generated_features.v1":
                row["features"] = [float(i), 1.0]
            rows.append(row)
        path = Path(task["data"][phase]["path"])
        atomic_json(path, rows)
        task["data"][phase]["sha256"] = sha256_file(path)
    store, driver = prepared(tmp_path, task)
    assert driver.model_plugin_id == plugin_id
    assert driver.acceptance_policy({})["plugin_id"] == plugin_id
    assert len(driver.inference_inputs()) == 4
    assert all("target" not in row for row in driver.inference_inputs())
