"""Real host capture survives an episode boundary, but never artifact tampering."""
import copy
import hashlib
import json

import pytest
import test_analysis_report_review as cases

from agent.harness import analysis_report_review
from agent.harness.research_analysis_history import load_history, merge_receipts
from agent.harness.research_analysis_provenance import build_evidence_index
from agent.harness.research_plan_evidence import checkpoint_steps
from agent.harness.research_workspace import ResearchWorkspace


@pytest.fixture
def history_case(tmp_path):
    goal = tmp_path / "goal"
    workspace = goal / "workspace"
    service = ResearchWorkspace(workspace, goal / "episodes/cycle_000/tools", {"steps": []})
    raw = b">original\nMKWVTFISLLFLFSSAYS\n"
    (workspace / "input.fasta").write_bytes(raw)
    inputs = [{"path": "input.fasta", "sha256": hashlib.sha256(raw).hexdigest()}]
    task = cases.run_capture(service, {"sequence": "MKWVTFISLLFLFSSAYS",
                                      "fasta_file": str(workspace / "input.fasta"),
                                      "output_file": str(workspace / "result.json")},
                             {"data": {"score": 1.233}},
                             output=("result.json", b'{"score":1.233}'))
    return goal, service, inputs, task


def history(goal, workspace, cycles=None):
    return load_history(goal, workspace=workspace, cycles=[{"index": 0}] if cycles is None else cycles)


def test_two_episodes_reuse_actual_capture_without_new_execution(history_case):
    goal, first, inputs, task = history_case
    original_receipts = (first.control / "receipts.json").read_bytes()
    second = ResearchWorkspace(first.workspace, goal / "episodes/cycle_001/tools", {"steps": []})
    recovered = merge_receipts(history(goal, first.workspace), second.receipts, first.workspace)
    assert len(recovered["receipts"]) == 1
    assert second.receipts == []
    assert (first.control / "receipts.json").read_bytes() == original_receipts
    assert not (second.control / "receipts.json").exists()
    text = "Recorded score: 1.233\n"
    (first.workspace / "report.md").write_text(text)
    artifact = {"path": "report.md", "sha256": hashlib.sha256(text.encode()).hexdigest()}
    evidence = first.receipts[0]["result"]["artifact"]
    cp = {"completed_steps": ["compute"], "artifacts": [artifact, evidence]}
    plan = {"deliverable": "analysis", "steps": [{"id": "compute", "verification": "execution"}]}
    checks = checkpoint_steps(plan, cp, workspace=first.workspace, receipts=recovered["receipts"],
                              experiments=[], progress={})
    assert checks[0]["passed"] is True
    assert checks[0]["analysis_job_ids"] == [task]
    context = {"objective": "Analyze the supplied protein", "plan": plan,
               "execution": {"checkpoint": cp, "step_checks": checks},
               "artifact_previews": [{**artifact, "content": text}],
               "analysis_evidence": build_evidence_index(first.workspace, receipts=recovered["receipts"],
                                                         inputs=inputs, checkpoint=cp)}
    review = {"decision": "ready_for_host_validation", "summary": "The report matches the recorded result.",
              "issues": [], "claims": [{"id": "c", "task_id": task, "result_pointer": "/result/data/score",
              "report_path": "report.md", "report_quote": text.strip(), "display_value": "1.233"}],
              "step_assessments": [{"step_id": "compute", "verdict": "supported",
                                   "reason": "Reported actual result.", "claim_ids": ["c"]}],
              "input_assessments": [{"task_id": task, "kind": "registered_input",
                                    "addresses_objective": True, "reason": "Registered input hash matches.",
                                    "source_task_ids": []}]}
    assert analysis_report_review.validate(review, context) is None


@pytest.mark.parametrize("fault", ["input", "output", "artifact", "different_workspace", "symlink", "hardlink"])
def test_changed_or_cross_goal_records_are_unavailable(history_case, tmp_path, fault):
    goal, first, _, _ = history_case
    if fault == "different_workspace":
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        copied = merge_receipts(history(goal, first.workspace), [], foreign)
        assert copied["receipts"] == []
        with pytest.raises(ValueError, match="this goal workspace"):
            history(goal, foreign)
        return
    if fault == "input":
        (first.workspace / "input.fasta").write_text(">other\nAAAA\n")
    elif fault == "output":
        (first.workspace / "result.json").write_text('{"score":9}')
    elif fault == "artifact":
        (first.workspace / first.receipts[0]["result"]["artifact"]["path"]).write_text("{}")
    else:
        p = first.control / "receipts.json"
        other = tmp_path / "receipts.json"
        other.write_bytes(p.read_bytes())
        p.unlink()
        if fault == "symlink":
            p.symlink_to(other)
        else:
            p.hardlink_to(other)
    merged = merge_receipts(history(goal, first.workspace), [], first.workspace)
    assert not merged["receipts"]
    assert merged["issues"]


def test_unlisted_episode_and_workspace_claim_do_not_become_history(history_case):
    goal, first, _, _ = history_case
    (first.workspace / "receipts.json").write_bytes((first.control / "receipts.json").read_bytes())
    loaded = history(goal, first.workspace, [])
    assert loaded["records"] == []
    assert merge_receipts(loaded, [], first.workspace)["receipts"] == []


def test_same_task_conflict_quarantines_both_even_if_new_variant_is_invalid(history_case):
    goal, first, _, task = history_case
    bad = copy.deepcopy(first.receipts[0])
    bad["result"]["evidence"]["result_sha256"] = "0" * 64
    current = [bad]
    merged = merge_receipts(history(goal, first.workspace), current, first.workspace)
    assert not merged["receipts"]
    assert any(row["code"] == "conflicting_private_task" and row["task_id"] == task for row in merged["issues"])
    assert current == [bad]
    duplicate = merge_receipts(history(goal, first.workspace), copy.deepcopy(first.receipts), first.workspace)
    assert len(duplicate["receipts"]) == 1


def test_invalid_old_output_does_not_block_unrelated_current_receipts(history_case):
    goal, first, _, _ = history_case
    (first.workspace / "result.json").write_text("changed")
    current = [{"tool": "run_research_python", "arguments_sha256": "1" * 64,
                "result": {"exit_code": 0, "submitted_source_sha256": "2" * 64}}]
    merged = merge_receipts(history(goal, first.workspace), current, first.workspace)
    assert merged["receipts"] == current
    assert merged["issues"]


def test_history_limit_is_explicit_and_bounded(history_case, monkeypatch):
    from agent.harness import research_analysis_history as module
    goal, first, _, _ = history_case
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 10)
    loaded = history(goal, first.workspace)
    assert not loaded["records"]
    assert loaded["issues"][0]["code"] == "private_receipts_unavailable_or_invalid"


def test_malformed_private_json_never_downgrades_to_workspace_receipt(history_case):
    goal, first, _, _ = history_case
    p = first.control / "receipts.json"
    p.write_text('[{"tool":"queued_analysis","tool":"run_research_python"}]')
    merged = merge_receipts(history(goal, first.workspace), [], first.workspace)
    assert not merged["receipts"]
    assert merged["issues"]


def test_current_epoch_owner_can_differ_without_rewriting_original_binding(history_case):
    goal, first, _, _ = history_case
    captured = first.receipts[0]["result"]
    merged = merge_receipts(history(goal, first.workspace), [], first.workspace)
    assert merged["receipts"][0]["result"]["owner_sha256"] == captured["owner_sha256"]
    assert merged["receipts"][0]["result"]["parent_job"] == captured["parent_job"]
    assert json.loads((first.control / "receipts.json").read_text()) == first.receipts


def fresh_capture(service, task_id, score):
    from agent.harness.research_analysis_evidence import capture
    from mcp_tools.scheduler.analysis_evidence import AnalysisCapture
    from mcp_tools.scheduler.receipt_binding import value_digest
    args = {"sequence": "MKWVTFISLLFLFSSAYS"}
    value = {"data": {"score": score}}
    worker = AnalysisCapture({"route": "native", "workspace": str(service.workspace),
                              "session_id": "same-goal", "name": "analysis", "arguments": args},
                             service.workspace, "sequence_analysis")
    evidence = worker.finish({"value": value, "is_error": False})
    job = {"id": task_id, "status": "completed", "tool": "analysis", "category": "sequence_analysis",
           "parent_job": "f" * 32, "session_id": "same-goal", "arguments_sha256": value_digest(args),
           "result": {"value": value, "is_error": False, "analysis_evidence": evidence}}
    capture(service, job=job, name="analysis", arguments=args, value=value,
            owner="owner", parent_job=job["parent_job"], session_id=job["session_id"])


def test_current_and_checkpoint_cited_history_survive_projection_limit(tmp_path):
    goal = tmp_path / "goal"
    first = ResearchWorkspace(goal / "workspace", goal / "episodes/cycle_000/tools", {"steps": []})
    for i in range(14):
        fresh_capture(first, format(i + 1, "032x"), i)
    second = ResearchWorkspace(first.workspace, goal / "episodes/cycle_001/tools", {"steps": []})
    current_id = "e" * 32
    fresh_capture(second, current_id, 18)
    cited = first.receipts[-1]["result"]["artifact"]
    cp = {"artifacts": [cited]}
    merged = merge_receipts(history(goal, first.workspace), second.receipts, first.workspace, checkpoint=cp)
    index = build_evidence_index(first.workspace, receipts=merged["receipts"], inputs=[], checkpoint=cp,
                                 current_task_ids=[current_id], task_provenance=merged["task_provenance"])
    tasks = [row["task_id"] for row in index["computations"]]
    assert current_id in tasks and first.receipts[-1]["result"]["task_id"] in tasks
    assert tasks[-1] == current_id  # Keep historical parents before current descendants.
    assert len(tasks) == 12
    assert any(row["reason"] == "computation_limit" and row["count"] == 3 for row in index["omissions"])
    assert index["computations"][-1]["host_origin"]["scope"] == "current"
    assert index["computations"][0]["host_origin"]["scope"] == "prior_episode"


def test_verification_has_shared_actual_byte_budget_and_current_priority(tmp_path, monkeypatch):
    from agent.harness import research_analysis_evidence as evidence
    goal = tmp_path / "goal"
    first = ResearchWorkspace(goal / "workspace", goal / "episodes/cycle_000/tools", {"steps": []})
    for i in range(6):
        fresh_capture(first, format(i + 1, "032x"), i)
    second = ResearchWorkspace(first.workspace, goal / "episodes/cycle_001/tools", {"steps": []})
    current_id = "e" * 32
    fresh_capture(second, current_id, 18)
    monkeypatch.setattr(evidence, "MAX_VERIFICATION_BYTES", 1500)
    read_sizes = []
    real_read = evidence.VerificationBudget.read

    def measured(self, *args, **kwargs):
        data = real_read(self, *args, **kwargs)
        read_sizes.append(len(data))
        return data

    monkeypatch.setattr(evidence.VerificationBudget, "read", measured)
    merged = merge_receipts(history(goal, first.workspace), second.receipts, first.workspace)
    assert sum(read_sizes) <= 1500
    assert current_id in [row["result"]["task_id"] for row in merged["receipts"]]
    assert len(merged["receipts"]) < 7
    assert merged["issues"]


def test_current_small_result_survives_large_history_index_byte_limit(tmp_path, monkeypatch):
    from agent.harness import research_analysis_provenance as provenance

    goal = tmp_path / "goal"
    first = ResearchWorkspace(goal / "workspace", goal / "episodes/cycle_000/tools", {"steps": []})
    for i in range(3):
        fresh_capture(first, format(i + 1, "032x"), ["historical" * 60] * 12)
    second = ResearchWorkspace(first.workspace, goal / "episodes/cycle_001/tools", {"steps": []})
    current_id = "e" * 32
    fresh_capture(second, current_id, 18)
    merged = merge_receipts(history(goal, first.workspace), second.receipts, first.workspace)
    monkeypatch.setattr(provenance, "MAX_INDEX_CHARS", 3500)
    index = provenance.build_evidence_index(first.workspace, receipts=merged["receipts"], inputs=[],
                                             current_task_ids=[current_id],
                                             task_provenance=merged["task_provenance"])
    assert current_id in [row["task_id"] for row in index["computations"]]
    assert len(json.dumps(index, separators=(",", ":"), ensure_ascii=False)) <= 3500
    assert any(row["reason"] == "index_size_limit" for row in index["omissions"])
