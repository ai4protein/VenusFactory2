"""Host checks SC declarations against real captured results and hashed reports."""
import copy
import hashlib
import json

import pytest

from agent.harness import analysis_report_review as review_policy
from agent.harness.research_analysis_evidence import capture
from agent.harness.research_analysis_provenance import build_evidence_index
from agent.harness.research_workspace import ResearchWorkspace
from mcp_tools.scheduler.analysis_evidence import AnalysisCapture
from mcp_tools.scheduler.receipt_binding import value_digest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run_capture(service, args, value, output=None):
    worker = AnalysisCapture({"route": "native", "workspace": str(service.workspace),
                              "session_id": "session", "name": "analysis", "arguments": args},
                             service.workspace, "sequence_analysis")
    if output is not None:
        path, data = output
        (service.workspace / path).write_bytes(data)
    evidence = worker.finish({"value": value, "is_error": False})
    assert evidence is not None
    task_id = format(len(service.receipts) + 1, "032x")
    job = {"id": task_id, "status": "completed", "tool": "analysis",
           "category": "sequence_analysis", "parent_job": "f" * 32, "session_id": "session",
           "arguments_sha256": value_digest(args),
           "result": {"value": value, "is_error": False, "analysis_evidence": evidence}}
    capture(service, job=job, name="analysis", arguments=args, value=value,
            owner="owner", parent_job=job["parent_job"], session_id=job["session_id"])
    return task_id


@pytest.fixture
def make_case(tmp_path):
    counter = 0

    def create(value=1.233, display="1.233", *, args=None, report=None, pointer="/result/data/score"):
        nonlocal counter
        counter += 1
        service = ResearchWorkspace(tmp_path / str(counter) / "workspace",
                                    tmp_path / str(counter) / "private", {"steps": []})
        raw = b">original\nMKWVTFISLLFLFSSAYS\n"
        (service.workspace / "input.fasta").write_bytes(raw)
        inputs = [{"path": "input.fasta", "sha256": sha(raw)}]
        task = run_capture(service, args or {"sequence": "MKWVTFISLLFLFSSAYS"}, {"data": {"score": value}})
        report = "Reported value: " + display + "\n" if report is None else report
        (service.workspace / "report.md").write_text(report)
        artifact = {"path": "report.md", "sha256": sha(report.encode())}
        checkpoint = {"completed_steps": ["compute"], "artifacts": [artifact]}
        context = {"objective": "Analyze the supplied protein.", "plan": {"steps": [
            {"id": "compute", "verification": "execution"}]},
            "execution": {"checkpoint": checkpoint, "step_checks": [
                {"id": "compute", "requirements": ["execution"], "passed": True}]},
            "artifact_previews": [{**artifact, "content": report}],
            "analysis_evidence": build_evidence_index(service.workspace, receipts=service.receipts,
                                                      inputs=inputs, checkpoint=checkpoint)}
        review = {"decision": "ready_for_host_validation", "summary": "Supported by the reported measurement.",
                  "issues": [], "claims": [{"id": "c1", "task_id": task, "result_pointer": pointer,
                  "report_path": "report.md", "report_quote": report.strip(), "display_value": display}],
                  "step_assessments": [{"step_id": "compute", "verdict": "supported",
                                       "reason": "Recorded output matches the report.", "claim_ids": ["c1"]}],
                  "input_assessments": [{"task_id": task, "kind": "registered_input",
                      "addresses_objective": True, "reason": "Exact registered sequence.",
                      "source_task_ids": []}]}
        return service, inputs, context, review

    return create


def test_real_capture_and_exact_hashed_report_are_accepted(make_case):
    _, _, context, review = make_case()
    assert review_policy.validate(review, context) is None
    assert review_policy.parse_review(json.dumps(review), context) == review
    wrapped = {**context, "analysis_evidence": {"index": context["analysis_evidence"]}}
    assert review_policy.parse_review("```json\n" + json.dumps(review) + "\n```", wrapped) == review


@pytest.mark.parametrize("value,display", [
    (1.233, "1.23"), (1.235, "1.24"), (1.245, "1.24"), (-0.256, "-0.26"),
    (2140.54, "2140.5"), (0.0001234, "1.23e-4"), (0, "0.00"),
    (True, "true"), (False, "False"), (None, "null"), ("active", "active"),
])
def test_scalar_types_and_declared_rounding(make_case, value, display):
    _, _, context, review = make_case(value, display)
    review_policy.validate(review, context)


@pytest.mark.parametrize("value,display", [
    (1.233, "1.24"), (1.233, "2"), (18, "9"), (2140.54, "2.14054"),
    (True, "1"), (False, "0"), (None, "None"), ("active", "inactive"),
    (-0.256, "0.256"), (0.3, "30"), (1.2, "1e999999"),
])
def test_wrong_values_units_and_bool_number_coercion_are_rejected(make_case, value, display):
    _, _, context, review = make_case(value, display)
    with pytest.raises(ValueError, match="display_value"):
        review_policy.validate(review, context)


def test_cannot_claim_missing_value_as_reported(make_case):
    _, _, context, review = make_case(1.233, "1.233", report="Hydrophobicity: Not provided by the tool\n")
    with pytest.raises(ValueError, match="displayed value"):
        review_policy.validate(review, context)


def test_numeric_substring_inside_another_number_is_not_a_quote(make_case):
    _, _, context, review = make_case(18, "18", report="Reported value: 118\n")
    with pytest.raises(ValueError, match="displayed value"):
        review_policy.validate(review, context)


@pytest.mark.parametrize("fault", ["preview", "quote", "checkpoint", "index", "changed", "truncated"])
def test_report_binding_is_not_satisfied_by_forged_or_partial_preview(make_case, fault):
    _, _, context, review = make_case()
    if fault == "preview":
        context["artifact_previews"][0]["content"] += "tampered"
    elif fault == "quote":
        review["claims"][0]["report_quote"] = "Not in the report: 1.233"
    elif fault == "checkpoint":
        context["execution"]["checkpoint"]["artifacts"] = []
    elif fault == "index":
        context["analysis_evidence"]["checkpoint_artifacts"]["records"][0]["sha256"] = "0" * 64
    elif fault == "changed":
        context["analysis_evidence"]["checkpoint_artifacts"]["records"][0]["status"] = "changed"
    else:
        context["artifact_previews"][0]["content"] = "Reported value"
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


@pytest.mark.parametrize("fault", ["pointer", "task", "omitted", "duplicate_claim", "extra_field", "missing_input"])
def test_claim_reference_integrity(make_case, fault):
    _, _, context, review = make_case()
    if fault == "pointer":
        review["claims"][0]["result_pointer"] = "/result/data/missing"
    elif fault == "task":
        review["claims"][0]["task_id"] = "a" * 32
    elif fault == "omitted":
        field = context["analysis_evidence"]["computations"][0]["result_fields"][0]
        field.pop("value")
        field["truncated"] = True
    elif fault == "duplicate_claim":
        review["claims"].append(copy.deepcopy(review["claims"][0]))
    elif fault == "extra_field":
        review["claims"][0]["host_verified"] = True
    else:
        review["input_assessments"] = []
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "unknown_claim", "no_execution_claim", "unsupported"])
def test_each_completed_step_has_its_own_supported_evidence(make_case, fault):
    _, _, context, review = make_case()
    if fault == "missing":
        review["step_assessments"] = []
    elif fault == "duplicate":
        review["step_assessments"].append(copy.deepcopy(review["step_assessments"][0]))
    elif fault == "extra":
        review["step_assessments"][0]["step_id"] = "another"
    elif fault == "unknown_claim":
        review["step_assessments"][0]["claim_ids"] = ["invented"]
    elif fault == "no_execution_claim":
        review["claims"] = []
        review["input_assessments"] = []
        review["step_assessments"][0]["claim_ids"] = []
    else:
        review["step_assessments"][0]["verdict"] = "insufficient_evidence"
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


def test_rejection_can_have_no_claims_without_fabricating_evidence(make_case):
    _, _, context, review = make_case()
    review.update(decision="needs_revision", claims=[], input_assessments=[])
    review["step_assessments"][0].update(verdict="insufficient_evidence", claim_ids=[])
    review_policy.validate(review, context)


@pytest.mark.parametrize("kind,relevant", [("unverified", True), ("registered_input", False), ("new_subject", False)])
def test_unverified_or_unrelated_inputs_cannot_support_positive_step(make_case, kind, relevant):
    _, _, context, review = make_case()
    review["input_assessments"][0].update(kind=kind, addresses_objective=relevant)
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


def test_unmatched_sequence_cannot_be_claimed_as_registered(make_case):
    _, _, context, review = make_case(args={"sequence": "MAESGPGQV"})
    with pytest.raises(ValueError, match="registered_input"):
        review_policy.validate(review, context)
    review["input_assessments"][0].update(kind="new_subject", reason="Exploratory design judged relevant by SC.")
    review_policy.validate(review, context)  # Relevance remains an explicit SC judgment.


def test_registered_file_input_is_supported_without_sequence_guessing(make_case):
    service, inputs, context, review = make_case()
    task = run_capture(service, {"fasta_file": str(service.workspace / "input.fasta")}, {"data": {"score": 1.233}})
    context["analysis_evidence"] = build_evidence_index(service.workspace, receipts=service.receipts,
        inputs=inputs, checkpoint=context["execution"]["checkpoint"])
    review["claims"][0]["task_id"] = task
    review["input_assessments"][0]["task_id"] = task
    review_policy.validate(review, context)


def derived_case(make_case):
    service, inputs, context, review = make_case()
    producer = run_capture(service, {"fasta_file": str(service.workspace / "input.fasta"),
                                    "output_file": str(service.workspace / "derived.fasta")},
                           {"data": {"created": True}}, ("derived.fasta", b">variant\nMAESGPGQV\n"))
    consumer = run_capture(service, {"fasta_file": str(service.workspace / "derived.fasta")},
                           {"data": {"score": 1.233}})
    context["analysis_evidence"] = build_evidence_index(service.workspace, receipts=service.receipts,
        inputs=inputs, checkpoint=context["execution"]["checkpoint"])
    review["claims"][0]["task_id"] = consumer
    review["input_assessments"][0].update(task_id=consumer, kind="derived_input",
                                        source_task_ids=[producer], reason="Recorded transformation chain.")
    return context, review


def test_true_hash_linked_derivation_is_allowed(make_case):
    context, review = derived_case(make_case)
    review_policy.validate(review, context)


@pytest.mark.parametrize("fault", ["empty", "unrelated", "self", "unverified_source", "changed_hash", "wrong_order"])
def test_explanation_cannot_replace_derivation_receipts(make_case, fault):
    context, review = derived_case(make_case)
    sources = review["input_assessments"][0]["source_task_ids"]
    tasks = context["analysis_evidence"]["computations"]
    if fault == "empty":
        sources.clear()
    elif fault == "unrelated":
        sources.insert(0, tasks[0]["task_id"])
    elif fault == "self":
        sources.append(review["claims"][0]["task_id"])
    elif fault == "unverified_source":
        tasks[1]["status"] = "unverified"
    elif fault == "changed_hash":
        tasks[2]["input_files"] = {"derived.fasta": "0" * 64}
    else:
        tasks[1], tasks[2] = tasks[2], tasks[1]
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


def test_parse_rejects_duplicate_json_keys_and_closed_schema_extensions(make_case):
    _, _, context, review = make_case()
    raw = json.dumps(review)
    with pytest.raises(ValueError):
        review_policy.parse_review('{"decision":"needs_revision",' + raw[1:], context)
    review["override"] = True
    with pytest.raises(ValueError):
        review_policy.parse_review(json.dumps(review), context)


def test_uncited_exploratory_tasks_need_not_be_assessed(make_case):
    _, _, context, review = make_case()
    context["analysis_evidence"]["computations"].append(
        {"status": "unverified", "task_id": "d" * 32, "reason": "changed_exploratory_output"})
    review_policy.validate(review, context)


@pytest.mark.parametrize("target", ["summary", "step", "input", "host_check"])
def test_blank_reason_or_failed_host_step_is_not_accepted(make_case, target):
    _, _, context, review = make_case()
    if target == "summary":
        review["summary"] = " "
    elif target == "step":
        review["step_assessments"][0]["reason"] = "\n "
    elif target == "input":
        review["input_assessments"][0]["reason"] = " "
    else:
        context["execution"]["step_checks"][0]["passed"] = False
    with pytest.raises(ValueError):
        review_policy.validate(review, context)


def test_a_derived_chain_cannot_hide_an_unrooted_branch(make_case):
    context, review = derived_case(make_case)
    tasks = context["analysis_evidence"]["computations"]
    tasks[1]["input_files"] = {"unknown-source.fasta": "0" * 64}
    tasks[1]["sequence_relations"]["records"] = []
    # Even if the final tool also receives the original sequence, its cited
    # transformation chain must actually originate from registered material.
    tasks[2]["sequence_relations"] = copy.deepcopy(tasks[0]["sequence_relations"])
    with pytest.raises(ValueError, match="chain rooted"):
        review_policy.validate(review, context)


@pytest.mark.parametrize("value,display,report", [
    (2, "2", "Reported value: -2 units\n"),
    (2, "2", "Reported value: −2 units\n"),
    (2, "2", "Reported value: +2 units\n"),
    (2, "+2", "Reported value: -2 units\n"),
    (3, "3", "Reported value: 1e-3 units\n"),
    (3, "3", "Reported value: 1E−3 units\n"),
    (3, "3", "Reported value: 1e+3 units\n"),
    (1, "1", "Reported value: 1.2 units\n"),
    (2, "2", "Reported value: 1.2 units\n"),
    (234, "234", "Reported value: 1,234 units\n"),
    (1, "1", "Reported value: 1,234 units\n"),
    (2, "2", "Reported value: –2 units\n"),
    (2, "2", "Reported range: 2–3 units\n"),
    (2, "2", "Reported value: −2.0 units\n"),
])
def test_numeric_quote_cannot_take_part_of_another_numeric_token(make_case, value, display, report):
    _, _, context, review = make_case(value, display, report=report)
    with pytest.raises(ValueError, match="displayed value"):
        review_policy.validate(review, context)


@pytest.mark.parametrize("value,display,report", [
    (-2, "-2", "Reported value: -2 units\n"),
    (-2, "-2", "Reported value: −2 units\n"),
    (-2, "−2", "Reported value: −2 units\n"),
    (2, "+2", "Reported value: +2units\n"),
    (0.001, "1e-3", "Reported value: 1e-3 M\n"),
    (0.001, "1E-3", "Reported value: 1E−3 M\n"),
    (1000, "1e+3", "Reported value: 1e+3 units\n"),
    (1000, "1E+3", "Reported value: 1E+3 units\n"),
    (1.2, "1.2", "Reported value: 1.2 units\n"),
    (2, "2", "- 2 units\n"),
    (2, "2", "Measured: 2.\n"),
    (2, "2", "Reported value: 2°C\n"),
    (2, "2", "Reported value: 2mg\n"),
    (2.5, "2.5", "Reported value: 2.5 μM.\n"),
    (-0.002, "-2e-3", "Reported value: −2e−3 units\n"),
])
def test_numeric_quote_preserves_sign_exponent_units_and_markdown(make_case, value, display, report):
    _, _, context, review = make_case(value, display, report=report)
    review_policy.validate(review, context)
