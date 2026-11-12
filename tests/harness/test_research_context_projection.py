"""Projection retains scientific contracts; referenced text is exactly recoverable."""
import copy
import hashlib
import json

import pytest

from agent.harness.research_context_projection import project_research_context


def context_fixture():
    plan = {"summary": "Measure a sequence model against a conventional baseline", "deliverable": "model",
            "needs_input": [], "steps": [
                {"id": f"step{i}", "title": f"Milestone {i}", "acceptance": f"Observed endpoint {i}, matched splits and evidence",
                 "verification": "model", "status": "completed" if i == 0 else "pending",
                 "approach": (f"Method {i}: preserve measured targets and compare 独立 validation. " * 80),
                 "evidence": [{"file": f"result-{i}.json", "sha256": str(i) * 64, "observed": True}]}
                for i in range(4)]}
    return {
        "objective": "Develop an enzyme-affinity predictor using public measurements, not proxy labels.",
        "criteria": ["research_plan_executed", "research_outputs_reviewed", "capability_admission_resolved"],
        "PI": "Current PI: revise the representation using the failed measured comparison. " * 30,
        "plan": plan,
        "progress": {
            "plan": copy.deepcopy(plan), "criteria": {"criterion1": False, "criterion2": False},
            "execution_feedback": {"pending": [{"key": "current", "error": "Missing module-level fit", "source_file": "model.py"}],
                                   "errors": [{"error": "Still unresolved"}], "superseded": [{"reason": "Old run used wrong source SHA"}]},
            "training_progress": {"completed": 1, "failed": 2, "latest_attempt_id": "attempt3"},
            "sc_revision": {"issues": ["No measured revision"], "requests": [{"action": "measure"}],
                            "measurement_state": {"baseline_measured": True}, "study_revision_requests": [{"requirement": "measured_revision"}]},
            "review_evidence_error": {"message": "Incomplete proposal"},
            "step_checks": {"step2": {"ok": False, "reason": "No checkpoint"}},
            "host_review": {"accepted": False, "errors": ["Missing evidence"]},
            "questions": ["Which assay?"], "pending_interaction": {"question_id": "q1"},
            "user_updates": [{"instruction": "Preserve the original assay units"}],
            "revision_history": [{"acceptance": "Retain the independent split"}],
            "new_future_feedback": {"required": "Unknown fields must remain"},
            "method_revisions": [{"episode": i, "step_id": "step2", "previous": "Old " * 500,
                                  "approach": "New " * 500, "acceptance": "Same physical endpoint"} for i in range(6)],
        },
        "model_development": {
            "prepared_dataset": {"evaluation_file": "checked/evaluation.json", "task": "regression",
                                 "model_interface": {"input_fields": {"sequence": "str", "ligand": "str"},
                                                     "target_fields": {"affinity": {"unit": "nM", "missing": None}}},
                                 "partitions": {"training": "visible", "validation": "host", "test": "reserved"}},
            "repair_source": {"code": "def fit(training, seed):\n    return {}\n", "sha256": "c" * 64,
                              "redacted": False, "source_check": {"entrypoints": ["fit"], "valid": False},
                              "error": "Missing predict(model, inputs)"},
            "prior_generations": {"generations": [{"id": "baseline1", "measurement": 0.2}]},
            "selected_tools": ["train_research_model"], "next_action": {"stage": "evaluate_revision"},
        },
        "resources": {"budget": {"model_calls": 64, "tool_calls": 18, "wall_seconds": 600}, "online": False},
        "execution_receipts": [{"attempt_id": "attempt3", "status": "failed"}],
        "execution_followup": {"reason": "Repair the same source before retry"},
        "planning_questions": [],
        "supplied_materials": {"files": [{"path": "measurements.csv", "sha256": "d" * 64}]},
        "previous_episodes": [{"summary": f"Episode {i}: " + "Measured failure 独立. " * 300, "status": "needs_revision"} for i in range(4)],
        "workspace_files": [f"experiments/cycle-{i:03d}/measured-validation.json" for i in range(100)],
    }


def documents(result):
    return {document["name"]: document for document in result.context["context_references"]["documents"]}


def recover(result, name):
    document = documents(result)[name]
    blocks = []
    for part in document["parts"]:
        block = result.reference_payloads[part["path"]].encode("utf-8")
        assert len(block) == part["bytes"]
        assert hashlib.sha256(block).hexdigest() == part["sha256"]
        blocks.append(block)
    raw = b"".join(blocks)
    assert len(raw) == document["bytes"]
    assert hashlib.sha256(raw).hexdigest() == document["sha256"]
    return json.loads(raw)


def test_required_science_and_repair_contracts_remain_complete():
    original = context_fixture()
    before = copy.deepcopy(original)
    result = project_research_context(original, current_step_id="step2")
    assert original == before
    view = result.context
    for key in ("objective", "criteria", "PI", "model_development", "resources", "supplied_materials",
                "execution_receipts", "execution_followup", "planning_questions"):
        assert view[key] == original[key]
    for key, value in original["progress"].items():
        if key not in {"plan", "method_revisions"}:
            assert view["progress"][key] == value
    for i, step in enumerate(original["plan"]["steps"]):
        for key, value in step.items():
            if key != "approach" or step["id"] == "step2":
                assert view["plan"]["steps"][i][key] == value
    assert "approach" not in view["plan"]["steps"][0]
    assert recover(result, "plan") == original["plan"]
    assert view["context_references"]["duplicate_fields"] == {"/progress/plan": "/plan"}
    assert len(json.dumps(view, ensure_ascii=False)) < len(json.dumps(original, ensure_ascii=False))
    view["model_development"]["repair_source"]["code"] = "changed copy"
    assert original == before


def test_unspecified_current_step_preserves_every_approach():
    original = context_fixture()
    result = project_research_context(original)
    assert result.context["plan"] == original["plan"]
    assert recover(result, "plan") == original["plan"]


def test_history_and_inventory_are_losslessly_recoverable():
    original = context_fixture()
    result = project_research_context(original, current_step_id="step2")
    assert recover(result, "previous-episodes") == original["previous_episodes"]
    assert recover(result, "method-revisions") == original["progress"]["method_revisions"]
    assert recover(result, "workspace-files") == original["workspace_files"]
    assert result.context["previous_episodes"] == original["previous_episodes"][-2:]
    assert result.context["progress"]["method_revisions"] == original["progress"]["method_revisions"][-2:]
    assert result.context["workspace_files"] == original["workspace_files"][:20]


def test_utf8_fragments_have_exact_hashes_and_bounded_read_sizes():
    original = context_fixture()
    result = project_research_context(original, current_step_id="step2", max_fragment_bytes=1024)
    assert result.reference_payloads
    assert all(len(text.encode("utf-8")) <= 1024 for text in result.reference_payloads.values())
    for name in documents(result):
        recover(result, name)
    assert recover(result, "plan") == original["plan"]
    assert any(len(value) != len(value.encode("utf-8")) for value in result.reference_payloads.values())


def test_distinct_progress_plan_is_preserved_instead_of_assumed_duplicate():
    original = context_fixture()
    original["progress"]["plan"]["steps"][0]["acceptance"] = "Different historical obligation"
    result = project_research_context(original, current_step_id="step2")
    assert result.context["progress"]["plan"] == original["progress"]["plan"]
    assert result.context["context_references"]["duplicate_fields"] == {}


def test_unknown_history_format_cannot_silently_drop_new_requirements():
    original = context_fixture()
    original["progress"]["method_revisions"][0]["new_unresolved_constraint"] = "Keep this pending"
    original["previous_episodes"][0]["pending_question"] = "Keep this too"
    result = project_research_context(original, current_step_id="step2")
    assert result.context["progress"]["method_revisions"] == original["progress"]["method_revisions"]
    assert result.context["previous_episodes"] == original["previous_episodes"]


def test_small_context_remains_inline_without_larger_reference_overhead():
    original = {"objective": "Inspect protein", "plan": {"steps": []}, "progress": {"pending": "Keep"}}
    result = project_research_context(original)
    assert result.context == original and result.context is not original
    assert result.reference_payloads == {}


@pytest.mark.parametrize("step", ["missing", "", 2])
def test_current_step_must_be_explicit_and_known(step):
    with pytest.raises(ValueError, match="exactly one"):
        project_research_context(context_fixture(), current_step_id=step)


def test_duplicate_step_ids_are_not_guessed():
    context = context_fixture()
    context["plan"]["steps"][1]["id"] = "step2"
    with pytest.raises(ValueError, match="exactly one"):
        project_research_context(context, current_step_id="step2")


@pytest.mark.parametrize("prefix", ["../context", "/workspace/context", "context/../secret", "context\\secret", "context/", ".hidden"])
def test_reference_prefix_never_escapes_workspace(prefix):
    with pytest.raises(ValueError, match="workspace-relative"):
        project_research_context(context_fixture(), reference_prefix=prefix)


@pytest.mark.parametrize("limit", [0, 255, 20001, True])
def test_fragment_limits_respect_read_tool_bound(limit):
    with pytest.raises(ValueError, match="256..20000"):
        project_research_context(context_fixture(), max_fragment_bytes=limit)


@pytest.mark.parametrize("key", ["token", "api_key", "authorization", "base_url", "runtimes", "context_references"])
def test_transport_fields_are_rejected_not_serialized(key):
    value = context_fixture()
    value[key] = "synthetic-private-transport-marker"
    with pytest.raises(ValueError) as failure:
        project_research_context(value)
    assert "synthetic-private-transport-marker" not in str(failure.value)


def test_nonfinite_values_are_not_silently_stringified():
    value = context_fixture()
    value["resources"]["budget"]["wall_seconds"] = float("nan")
    with pytest.raises(ValueError):
        project_research_context(value)


def test_projection_and_reference_paths_are_deterministic():
    original = context_fixture()
    first = project_research_context(original, current_step_id="step2")
    second = project_research_context(dict(reversed(list(original.items()))), current_step_id="step2")
    assert first == second
