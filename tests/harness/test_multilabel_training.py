"""Isolated multilabel fitting through private delivery; synthetic data only."""

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from test_research_development import review_spec

from agent.harness.capability_review import review_proposal
from agent.harness.research_data_memory import prepared_dataset_context
from agent.harness.research_data_store import ResearchDataStore
from agent.harness.research_workspace import (
    ResearchWorkspace,
    invoke_bundle,
    publish_proposals,
    run_capability,
    run_python,
)
from agent.rsi.artifacts.package import build_delivery

LABELS = ["Cytoplasm", "Nucleus", "Plastid"]
CHECK = """import math
def input_value(inputs):
    if set(inputs) != {'x'} or not isinstance(inputs['x'], str):
        raise ValueError('Provide x as a table string')
    x = float(inputs['x'])
    if not math.isfinite(x) or abs(x) > 100:
        raise ValueError('x outside synthetic scope')
    return x
"""
BASELINE = (
    CHECK
    + """def fit(training, seed):
    assert training['labels'] == ['Cytoplasm', 'Nucleus', 'Plastid']
    assert all(row['id'].startswith('train-') for row in training['train'])
    result = {}
    for label in training['labels']:
        values = [row['target'][label] for row in training['train'] if row['target'][label] is not None]
        result[label] = sum(values)/len(values)
    return result
def predict(model, inputs):
    input_value(inputs)
    return {'value': dict(model)}
"""
)
LEARNER = (
    CHECK
    + """def fit(training, seed):
    assert all(row['id'].startswith('train-') for row in training['train'])
    fitted = {}
    for label in training['labels']:
        pairs = sorted((float(row['inputs']['x']), row['target'][label])
                       for row in training['train'] if row['target'][label] is not None)
        candidates = []
        for a, b in zip(pairs, pairs[1:]):
            if a[0] == b[0]:
                continue
            threshold = (a[0]+b[0])/2
            left = [y for x, y in pairs if x <= threshold]
            right = [y for x, y in pairs if x > threshold]
            pleft, pright = (sum(left)+1)/(len(left)+2), (sum(right)+1)/(len(right)+2)
            loss = sum(((pleft if x<=threshold else pright)-y)**2 for x,y in pairs)
            candidates.append((loss, threshold, pleft, pright))
        _, cut, left, right = min(candidates)
        fitted[label] = {'threshold':cut, 'left':left, 'right':right}
    return fitted
def predict(model, inputs):
    x = input_value(inputs)
    return {'value': {label: weights['left'] if x <= weights['threshold'] else weights['right']
                      for label, weights in model.items()}}
"""
)


def setup(tmp_path, *, damage=None):
    work = tmp_path / "workspace"
    store = ResearchDataStore(tmp_path / "private", work)
    episodes = tmp_path / "episodes"
    service = ResearchWorkspace(
        work,
        episodes / "cycle_000/tools",
        {"steps": [], "deliverable": "model"},
        development_root=tmp_path / "development",
        data_store=store,
    )
    records = []
    for split, values in [
        ("train", [-8, -6, -4, -2, 2, 4, 6, 8]),
        ("validation", [-5, -1, 1, 5]),
        ("test", [-7, -3, 3, 7]),
    ]:
        for index, x in enumerate(values):
            target = {
                "Cytoplasm": int(x > 0),
                "Nucleus": int(x < 0),
                "Plastid": int(x > 0) if split == "train" and index != 0 else None,
            }
            records.append(
                {
                    "id": f"{split}-{index}",
                    "group": f"{split}-{index}",
                    "split": split,
                    "x": str(x),
                    "localization": json.dumps(target),
                }
            )
    if damage:
        damage(records)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "group", "split", "x", "localization"])
    writer.writeheader()
    writer.writerows(records)
    raw = buf.getvalue().encode()
    store.ingest("inputs/localization.csv", raw, hashlib.sha256(raw).hexdigest())
    args = {
        "data_file": "inputs/localization.csv",
        "target_column": "localization",
        "input_columns": ["x"],
        "group_column": "group",
        "id_column": "id",
        "task": "multilabel",
        "split_column": "split",
        "provenance": "Synthetic software fixture; not biological measurements",
        "grouping": "Independent synthetic observations in fixed partitions",
    }
    return service, store, episodes, args


@pytest.mark.asyncio
async def test_protected_multilabel_fit_revision_review_package_and_private_invocation(tmp_path):
    service, store, episodes, args = setup(tmp_path)
    prepared = await service.prepare_research_dataset(**args)
    data = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    hidden_ids = {row["id"] for split in ["validation", "test"] for row in data["splits"][split]}
    visible = list(csv.DictReader(io.StringIO((service.workspace / "inputs/localization.csv").read_text())))
    assert {row["id"] for row in visible} == {row["id"] for row in data["splits"]["train"]}
    assert data["labels"] == LABELS
    restored = prepared_dataset_context(service.workspace, episodes, data_store=store)
    assert restored["model_interface"] == prepared["model_interface"]
    assert set(restored["model_interface"]["predict"]["returns"]["value"]) == set(LABELS)
    assert "null" in restored["model_interface"]["fit"]["labels"]

    async def guarded_runner(workspace, control, script, timeout=120, **kwargs):
        for path in Path(workspace).rglob("*"):
            if path.is_file():
                text = path.read_bytes().decode("utf-8", errors="ignore")
                assert not any(identity in text for identity in hidden_ids)
        return await run_python(workspace, control, script, timeout, **kwargs)

    service.runner = guarded_runner
    await service.write_research_file("baseline.py", BASELINE)
    base = await service.train_research_model(
        "Per-label prevalence", "baseline.py", prepared["evaluation_file"], "Training-only reference"
    )
    assert base["experiment"]["validation"]["diagnostics"]["matches_training_reference"]
    await service.write_research_file("learner.py", LEARNER)
    child = await service.train_research_model(
        "Smoothed label stumps",
        "learner.py",
        "",
        "Fit independent thresholds from training labels",
        parent_experiment_id=base["experiment"]["id"],
    )
    training, experiment = child["training"], child["experiment"]
    assert experiment["comparison"]["improved_on_validation"]
    assert experiment["evaluation_sha256"] == base["experiment"]["evaluation_sha256"]
    assert experiment["validation"]["multilabel"]["per_label"]["Plastid"]["observed"] == 0
    assert experiment["validation"]["multilabel"]["observed_cells"] == 8
    assert experiment["validation"]["multilabel"]["unobserved_cells"] == 4
    assert not experiment["test_used_for_feedback"]
    spec = review_spec(experiment["id"])
    spec.update(
        evaluation_file=training["evaluation_file"],
        reproduction_script=training["reproduction_script"],
        fitted_files=training["fitted_files"],
        evidence_files=training["evidence_files"] + ["README.md"],
        dependencies=list(training["dependency_versions"]),
    )
    spec["scope"]["properties"] = ["Synthetic independent binary labels; no biological claim"]
    spec["input_schema"] = {
        "type": "object",
        "required": ["x"],
        "properties": {"x": {"type": "string"}},
        "additionalProperties": False,
    }
    spec["output_schema"] = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
        "properties": {
            "value": {
                "type": "object",
                "required": LABELS,
                "additionalProperties": False,
                "properties": {label: {"type": "number", "minimum": 0, "maximum": 1} for label in LABELS},
            }
        },
    }
    example = {"x": "2.5"}
    expected = await invoke_bundle(
        service.workspace,
        {"source": training["source"], "files": experiment["files"]},
        example,
        tmp_path / "measured-example",
        runner=guarded_runner,
    )
    spec["cases"] = [{"inputs": example, "expected": expected}, {"inputs": {}, "reject": True}]
    await service.write_research_file(
        "README.md",
        "Synthetic multilabel protocol fixture. Null labels are unobserved. No biological or novelty claim.",
    )
    await service.propose_research_capability(
        "Multilabel protocol fixture",
        "Fitted synthetic probability model",
        training["source"],
        training["fitted_files"],
        [example],
        spec,
    )
    bundle = service.proposals[-1]

    async def reviewer(context):
        assert not any(identity in json.dumps(context) for identity in hidden_ids)
        evidence = context["host_evidence"]["evaluation"]
        assert evidence["task"] == "multilabel" and evidence["training_reproduced"]
        assert evidence["multilabel"]["per_label"]["Plastid"]["brier"] is None
        assert len([doc for doc in context["documents"] if doc.get("kind") == "training_data"]) == 1
        return {
            "decision": "approve",
            "summary": "Controlled synthetic protocol test; not scientific validation.",
            "checks": {
                key: {"passed": True, "reason": "Synthetic fixture protocol check"}
                for key in context["required_checks"]
            },
            "issues": [],
            "tests": [
                {"inputs": {"x": "3.5"}, "invariants": {"finite_prediction": True, "probability_labels": LABELS}},
                {"inputs": {"x": "nan"}, "reject": True},
            ],
        }

    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path / "review", runner=guarded_runner)
    assert receipt["status"] == "approved", receipt
    assert receipt["automatic"]["evaluation"]["diagnostics"]["split"] == "test"
    delivery = build_delivery(service.workspace, bundle, receipt, goal_id="multilabel-protocol")
    destination = tmp_path / "downloaded"
    with zipfile.ZipFile(service.workspace / delivery["model_package"]) as archive:
        assert "Multilabel probabilities" in archive.read("evaluation/test-report.md").decode()
        assert "Plastid | 0 | 0 | n/a" in archive.read("evaluation/test-report.md").decode()
        report = json.loads(archive.read("evaluation/test-report.json"))
        assert report["final_test"]["multilabel"]["unobserved_cells"] == 4
        for name in archive.namelist():
            assert not any(identity in archive.read(name).decode("utf-8", errors="ignore") for identity in hidden_ids)
        archive.extractall(destination)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    run = subprocess.run(command, cwd=destination, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert "test_independent_review" in run.stderr
    library = tmp_path / "private-models"
    entry = publish_proposals([bundle], "multilabel-protocol", library)[0]
    result = await run_capability(entry["id"], example, library=library)
    assert result == expected
    assert not (tmp_path / "public-models").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["missing_label", "unknown_only", "nonbinary", "duplicate_key"])
async def test_bad_labels_do_not_expose_training_rows_or_start_workers(tmp_path, bad):
    def damage(rows):
        values = {
            "missing_label": '{"Cytoplasm":1,"Nucleus":0}',
            "unknown_only": '{"Cytoplasm":null,"Nucleus":null,"Plastid":null}',
            "nonbinary": '{"Cytoplasm":0.2,"Nucleus":0,"Plastid":1}',
            "duplicate_key": '{"Cytoplasm":1,"Cytoplasm":0,"Nucleus":0,"Plastid":1}',
        }
        rows[-1]["localization"] = values[bad]

    service, store, _, args = setup(tmp_path, damage=damage)
    before = (service.workspace / "inputs/localization.csv").read_bytes()
    with pytest.raises(ValueError):
        await service.prepare_research_dataset(**args)
    assert (service.workspace / "inputs/localization.csv").read_bytes() == before
    assert not service.development.records()
