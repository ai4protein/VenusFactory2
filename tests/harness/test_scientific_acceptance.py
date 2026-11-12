"""SC policy and real isolated lifecycle tests. Synthetic data, no biological claims."""

import copy
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from test_autonomous_research import profile
from test_research_development import review_spec

from agent.harness import autonomous_research as ar
from agent.harness import scientific_acceptance as sc
from agent.harness.research_data_store import store_for_goal
from agent.harness.research_workspace import invoke_bundle, run_capability
from agent.harness.scientist_role_output import RoleOutputError, specification, validate_output


def packet():
    return {
        "scientific_acceptance": {
            "models": [
                {
                    "candidate_id": "candidate-a",
                    "binding_errors": [],
                    "binding": {"selected_experiment_id": "experiment-b", "runtime_sha256": "measured-version"},
                    "experiments": [{"id": "experiment-a"}, {"id": "experiment-b"}],
                }
            ]
        }
    }


def accepted(context):
    return {
        "decision": "ready_for_host_validation",
        "summary": "Measured value within the declared synthetic scope.",
        "issues": [],
        "model_assessments": [
            {
                "candidate_id": row["candidate_id"],
                "verdict": "useful",
                "use_case": "Predict the quadratic synthetic response in the evaluated interval; software testing only.",
                "value_basis": "predictive",
                "checks": {
                    key: {
                        "passed": True,
                        "reason": "Measured validation and matched control support this bounded software-test use.",
                        "experiment_ids": [row["binding"]["selected_experiment_id"]],
                    }
                    for key in sc.CHECKS
                },
                "limitations": ["Synthetic protocol fixture, not a protein model or novel algorithm."],
                "revision_request": "",
            }
            for row in context["scientific_acceptance"]["models"]
        ],
    }


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_model_sc_schema_is_typed_without_changing_historical_critic_contract(protocol):
    spec = specification("SC", protocol, model_acceptance=True)
    value = accepted(packet())
    assert json.loads(validate_output(json.dumps(value), spec)) == value
    assert sc.parse_model_review(json.dumps(value)) == value
    assert "model_assessments" not in specification("SC", protocol)["schema"]["properties"]
    with pytest.raises(RoleOutputError):
        validate_output(json.dumps({k: v for k, v in value.items() if k != "model_assessments"}), spec)
    with pytest.raises(RoleOutputError):
        validate_output(json.dumps(value).replace('"passed": true', '"passed": "true"', 1), spec)


@pytest.mark.parametrize(
    "fault",
    [
        "omitted",
        "duplicate",
        "foreign_candidate",
        "unknown_experiment",
        "unselected_evidence",
        "failed_check",
        "negative",
        "insufficient",
        "pending_revision",
        "host_binding",
        "global_rejection",
        "issues",
    ],
)
def test_unsupported_or_rejected_usefulness_never_passes(fault):
    context = packet()
    value = accepted(context)
    assessment = value["model_assessments"][0]
    if fault == "omitted":
        value["model_assessments"] = []
    elif fault == "duplicate":
        value["model_assessments"].append(copy.deepcopy(assessment))
    elif fault == "foreign_candidate":
        assessment["candidate_id"] = "unsubmitted"
    elif fault == "unknown_experiment":
        assessment["checks"]["goal_fit"]["experiment_ids"] = ["another-study"]
    elif fault == "unselected_evidence":
        assessment["checks"]["scientific_value"]["experiment_ids"] = ["experiment-a"]
    elif fault == "failed_check":
        assessment["checks"]["evaluation_validity"]["passed"] = False
    elif fault in {"negative", "insufficient"}:
        assessment["verdict"] = "not_useful" if fault == "negative" else "insufficient_evidence"
        assessment["revision_request"] = "PI must measure whether the model predicts beyond a constant response."
    elif fault == "pending_revision":
        assessment["revision_request"] = "The generalization claim still needs evaluation."
    elif fault == "host_binding":
        context["scientific_acceptance"]["models"][0]["binding_errors"] = ["Runtime changed"]
    elif fault == "global_rejection":
        value["decision"] = "needs_revision"
    elif fault == "issues":
        value["issues"] = ["Provenance is unresolved."]
    result = sc.assess(sc.parse_model_review(json.dumps(value)), context)
    assert not result["passed"]
    assert result["issues"] and result["approved_models"] == {}


def test_usefulness_does_not_impose_universal_baseline_superiority():
    context = packet()
    context["scientific_acceptance"]["models"][0]["experiments"] = [
        {"id": "experiment-a", "validation": {"score": 0.1, "baseline": 1, "direction": "minimize"}},
        {
            "id": "experiment-b",
            "validation": {"score": 0.2, "baseline": 1, "direction": "minimize"},
            "comparison": {"improved_on_validation": False},
        },
    ]
    value = accepted(context)
    value["model_assessments"][0]["checks"]["scientific_value"]["reason"] = (
        "The selected learner meets the prespecified error tolerance, predicts better than the naive reference, "
        "and remains worse than the matched conventional learner. No superiority claim."
    )
    assert sc.assess(value, context)["passed"]


def test_pi_receives_sc_rejection_before_any_submission_retry():
    feedback = {
        "issues": ["Reference-equivalent predictions do not address the stated target."],
        "requests": [{"revision_request": "Compare a trained feature model on the frozen validation split."}],
        "experiment_ids": ["rejected"],
    }
    action = ar.development_next_action(
        {"generations": [{"id": "rejected", "variant": "candidate", "parent": "baseline"}]},
        {"sc_revision": feedback, "publication_reviews": [{"status": "needs_revision"}]},
    )
    assert action["stage"] == "revise_scientific_model"
    assert action["issues"] == feedback["issues"]
    assert action["rejected_experiment_ids"] == ["rejected"]


INPUT = """import math

def input_value(inputs):
    if set(inputs) != {'x'} or not isinstance(inputs['x'], str):
        raise ValueError('Provide x as a table string')
    x = float(inputs['x'])
    if not math.isfinite(x) or abs(x) > 100:
        raise ValueError('Input outside fixture scope')
    return x
"""
BASELINE = (
    INPUT
    + """
def fit(training, seed):
    values = [row['target'] for row in training['train']]
    return {'mean': sum(values)/len(values)}
def predict(model, inputs):
    input_value(inputs)
    return {'value': model['mean']}
"""
)
BAD = BASELINE.replace("{'mean': sum(values)/len(values)}", "{'mean': -abs(sum(values)/len(values))}")
LEARNER = (
    INPUT
    + """
def fit(training, seed):
    import numpy as np
    assert all(row['id'].startswith('train-') for row in training['train'])
    xs = [float(row['inputs']['x']) for row in training['train']]
    ys = [row['target'] for row in training['train']]
    return {'coefficients': np.linalg.lstsq(np.array([[1,x,x*x] for x in xs]), np.array(ys), rcond=None)[0].tolist()}
def predict(model, inputs):
    x = input_value(inputs)
    return {'value': sum(w*x**i for i,w in enumerate(model['coefficients']))}
"""
)


@pytest.mark.asyncio
async def test_sc_rejects_then_pi_revises_and_retrained_model_is_accepted(tmp_path, monkeypatch):
    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"

        def __init__(self, **kwargs):
            Path(kwargs["output"]).touch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    monkeypatch.setattr(ar, "audit_requests", lambda path: {"passed": True, "model_calls": 4})
    store = ar.create(
        tmp_path / "goal",
        profile(),
        "Develop a quadratic predictor for synthetic software protocol testing",
        {"max_cycles": 3, "model_calls": 36, "tool_calls": 36, "wall_seconds": 360},
    )
    data_store = store_for_goal(store.root)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "group", "split", "x", "target"])
    writer.writeheader()
    for split, start, count in [("train", 1, 8), ("validation", 9, 3), ("test", 12, 3)]:
        for i in range(start, start + count):
            writer.writerow(
                {"id": f"{split}-{i}", "group": f"{split}-{i}", "split": split, "x": i / 10, "target": (i / 10) ** 2}
            )
    raw = buffer.getvalue().encode()
    data_store.ingest("inputs/observations.csv", raw, hashlib.sha256(raw).hexdigest())
    roles, attempts, measured, sc_packets, review_packets = [], [], [], [], []

    async def consult(role, context):
        roles.append(role)
        if role == "PI":
            if attempts:
                assert context["progress"]["sc_revision"]["requests"][0]["verdict"] == "not_useful"
                assert not list((store.root / "private_capabilities").glob("index/*"))
                assert not context["progress"].get("deliveries")
                assert context["model_development"]["next_action"]["stage"] == "revise_scientific_model"
            return "PI_ENDORSEMENT_SENTINEL"
        if role == "CB":
            return json.dumps(
                {
                    "summary": "Synthetic quadratic protocol",
                    "deliverable": "model",
                    "needs_input": [],
                    "steps": [
                        {
                            "id": "fit",
                            "title": "Fit and measure a predictive model",
                            "acceptance": "Measured input-dependent predictions on frozen synthetic splits",
                            "approach": "Fit quadratic features" if attempts else "Evaluate a constant control",
                            "verification": "model",
                        }
                    ],
                }
            )
        if role == "SC":
            text = json.dumps(context)
            assert "PI_ENDORSEMENT_SENTINEL" not in text and "MLS_ENDORSEMENT_SENTINEL" not in text
            assert "progress" not in context and "PI" not in context
            sc_packets.append(copy.deepcopy(context))
            value = accepted(context)
            if len(attempts) == 1:
                value.update(
                    decision="needs_revision",
                    summary="The constant candidate has no demonstrated input-specific utility.",
                    issues=["PI must fit and evaluate a feature-dependent model."],
                )
                row = value["model_assessments"][0]
                row.update(
                    verdict="not_useful",
                    revision_request="Fit a feature-dependent learner and compare against the measured constant baseline.",
                )
                row["checks"]["scientific_value"].update(
                    passed=False,
                    reason="Host diagnostics show constant predictions; the requested prediction objective is unmet.",
                )
            return json.dumps(value)
        assert role == "REVIEW" and len(attempts) == 2
        assert "scientific_acceptance" not in context
        review_packets.append(context)
        return json.dumps(
            {
                "decision": "approve",
                "summary": "Synthetic interface and training replay checked; no biological claim.",
                "checks": {
                    key: {"passed": True, "reason": "Bounded synthetic software test, verified host execution."}
                    for key in context["required_checks"]
                },
                "issues": [],
                "tests": [
                    {"inputs": {"x": "3"}, "invariants": {"finite_prediction": True, "min": 8.9, "max": 9.1}},
                    {"inputs": {"x": "not-a-number"}, "reject": True},
                ],
            }
        )

    async def execute(request, service, events):
        attempts.append(1)
        if len(attempts) == 1:
            prepared = await service.prepare_research_dataset(
                data_file="inputs/observations.csv",
                target_column="target",
                input_columns=["x"],
                group_column="group",
                id_column="id",
                split_column="split",
                provenance="Synthetic protocol fixture; no biological measurements",
                grouping="Disjoint synthetic observations",
            )
            await service.write_research_file("baseline.py", BASELINE)
            baseline = await service.train_research_model(
                "Baseline", "baseline.py", prepared["evaluation_file"], "Training-only constant reference"
            )
            measured.append(baseline)
        await service.write_research_file("learner.py", BAD if len(attempts) == 1 else LEARNER)
        child = await service.train_research_model(
            "Quadratic candidate",
            "learner.py",
            "",
            "Compare a fitted response against the constant control",
            parent_experiment_id=measured[0]["experiment"]["id"],
        )
        measured.append(child)
        training, experiment = child["training"], child["experiment"]
        await service.write_research_file(
            "README.md", "Synthetic software protocol test. No protein, biological or novelty claim."
        )
        spec = review_spec(experiment["id"])
        spec.update(
            evaluation_file=training["evaluation_file"],
            reproduction_script=training["reproduction_script"],
            fitted_files=training["fitted_files"],
            evidence_files=training["evidence_files"] + ["README.md"],
            dependencies=list(training["dependency_versions"]),
        )
        spec["input_schema"]["properties"]["x"] = {"type": "string"}
        example = {"x": "2"}
        expected = await invoke_bundle(
            service.workspace,
            {"source": training["source"], "files": experiment["files"]},
            example,
            service.control / "example",
            runner=service.runner,
        )
        spec["cases"] = [{"inputs": example, "expected": expected}, {"inputs": {}, "reject": True}]
        await service.propose_research_capability(
            "Quadratic fixture",
            "Synthetic numeric interpolation test only",
            training["source"],
            list(experiment["files"]),
            [example],
            spec,
        )
        await service.submit_research_checkpoint(
            "Measured candidate", ["fit"], ["README.md", "learner.py", "experiments/" + experiment["id"] + ".json"], []
        )
        events.extend(
            [
                {"kind": "tool_call_start"},
                {"kind": "completed", "text": "MLS_ENDORSEMENT_SENTINEL", "event_audit": {"passed": True}},
            ]
        )

    runner = ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[])
    state = await runner.run()
    assert state["status"] == "completed", state
    assert len(attempts) == 2 and roles == ["PI", "CB", "SC", "PI", "CB", "SC", "REVIEW"]
    assert len(review_packets) == 1
    assert state["cycles"][0]["status"] == "needs_revision"
    assert state["progress"]["method_revisions"] and not state["progress"].get("sc_revision")
    assert all(row["passed"] for row in runner.evaluate(state)["criteria"].values())
    rejected = json.loads((store.root / "episodes/cycle_000/scientific_acceptance.json").read_text())
    assert not rejected["passed"] and rejected["approved_models"] == {}
    delivery = state["progress"]["deliveries"][0]
    assert sc.verify_delivery(store.root, delivery)
    with zipfile.ZipFile(store.root / "workspace" / delivery["package"]) as archive:
        acceptance = json.loads(archive.read("evaluation/scientific-acceptance.json"))
        assert acceptance["assessment"]["verdict"] == "useful"
        assert acceptance["selected_experiment_id"] == measured[-1]["experiment"]["id"]
        assert "Scientific acceptance (SC)" in archive.read("evaluation/test-report.md").decode()
    output = await run_capability(delivery["candidate_id"], {"x": "3"}, library=store.root / "private_capabilities")
    assert output["value"] == pytest.approx(9)
    # Evidence packets remain training/validation summaries; no held-out rows.
    assert "test-12" not in json.dumps(sc_packets)
    from agent.harness.research_development import Development

    bundle = (store.root / state["progress"]["publication_reviews"][0]["receipt"]).parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    source = bundle / manifest["source"]
    saved = source.read_bytes()
    source.write_bytes(saved + b"\n# changed after measurement\n")
    changed = sc.evidence_context({}, [bundle], Development(store.root / "development"))
    assert changed["scientific_acceptance"]["models"][0]["binding_errors"] == ["Measured runtime file changed"]
    assert not sc.assess(accepted(sc_packets[-1]), changed)["passed"]
    source.write_bytes(saved)
    # Changing the decision file invalidates completion even if plan steps remain done.
    decision = store.root / "episodes/cycle_001/review.json"
    decision.write_text("{}")
    assert not sc.verify_delivery(store.root, delivery)
    assert not any(row["passed"] for row in runner.evaluate(state)["criteria"].values())


@pytest.mark.parametrize("basis,passed", [("predictive", False), ("reference", True)])
def test_constant_output_requires_an_explicit_reference_purpose(basis, passed):
    context = packet()
    context["scientific_acceptance"]["models"][0]["experiments"][1]["validation"] = {
        "diagnostics": {"constant_predictions": True}
    }
    value = accepted(context)
    value["model_assessments"][0]["value_basis"] = basis
    value["model_assessments"][0]["use_case"] = (
        "Training prevalence reference for comparisons; does not rank individual observations."
        if basis == "reference"
        else "Predict individual outcomes"
    )
    assert sc.assess(value, context)["passed"] is passed


@pytest.mark.parametrize("field", ["use_case", "limitations", "reason"])
def test_empty_sc_justifications_cannot_be_accepted(field):
    context = packet()
    value = accepted(context)
    row = value["model_assessments"][0]
    if field == "reason":
        row["checks"]["scientific_value"]["reason"] = "   "
    elif field == "limitations":
        row[field] = ["  "]
    else:
        row[field] = "  "
    assert not sc.assess(value, context)["passed"]


def test_fresh_sc_context_preserves_user_constraints_but_not_role_endorsements():
    context = {
        "objective": "Develop a useful predictor",
        "PI": "Approve our work",
        "plan": {"steps": []},
        "progress": {
            "user_updates": ["The model must support held-out protein families."],
            "last_review": {"decision": "ready_for_host_validation"},
        },
        "execution": {"report": "Developer claims success", "step_checks": []},
    }
    result = sc.evidence_context(context, [], None)
    assert result["user_updates"] == context["progress"]["user_updates"]
    assert "PI" not in result and "progress" not in result and "report" not in result["execution"]
    assert "Approve our work" not in json.dumps(result)


def test_one_rejected_candidate_prevents_collective_acceptance():
    context = packet()
    second = copy.deepcopy(context["scientific_acceptance"]["models"][0])
    second["candidate_id"] = "candidate-b"
    context["scientific_acceptance"]["models"].append(second)
    value = accepted(context)
    value["model_assessments"][1].update(
        verdict="not_useful", revision_request="PI must replace the unsupported hypothesis."
    )
    result = sc.assess(value, context)
    assert not result["passed"] and not result["approved_models"]


@pytest.mark.asyncio
async def test_failed_sc_response_cannot_inherit_an_old_completion(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    @asynccontextmanager
    async def relay(**kwargs):
        Path(kwargs["output"]).touch()
        yield SimpleNamespace(calls=0, base_url="http://unused", token="fixture")

    monkeypatch.setattr(ar, "EvidenceGateway", relay)
    monkeypatch.setattr(ar, "audit_requests", lambda path: {"passed": True, "model_calls": 4})
    store = ar.create(
        tmp_path / "goal",
        profile(),
        "Document the supplied observation",
        {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240},
    )
    plan = {
        "summary": "Record observation",
        "deliverable": "analysis",
        "needs_input": [],
        "steps": [
            {
                "id": "notes",
                "title": "Record observation",
                "acceptance": "Saved observation",
                "status": "completed",
                "evidence": [],
            }
        ],
    }
    store.mutate(
        lambda state: state["progress"].update(
            plan=plan,
            reviewed_episode=0,
            research_completed=True,
            last_review={"decision": "ready_for_host_validation"},
        )
    )
    folder = store.root / "episodes/cycle_000"
    folder.mkdir(parents=True)
    (folder / "review.json").write_text(
        json.dumps({"decision": "ready_for_host_validation", "summary": "Old accepted notes", "issues": []})
    )

    async def consult(role, context):
        if role == "PI":
            return "Record the observation."
        if role == "CB":
            return json.dumps(plan)
        return "Malformed SC response"

    async def execute(request, service, events):
        await service.write_research_file("notes.md", "Updated observation")
        await service.submit_research_checkpoint("New observation", ["notes"], ["notes.md"], [])
        events.append({"kind": "completed", "text": "New notes", "event_audit": {"passed": True}})

    runner = ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[])
    result = await runner.episode(1, profile(), store.read(), "")
    assert result["status"] == "failed"
    assert "reviewed_episode" not in result["progress"] and not result["progress"]["research_completed"]
    assert not any(
        row["passed"] for row in runner.evaluate({**store.read(), "progress": result["progress"]})["criteria"].values()
    )
