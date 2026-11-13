"""Checked whole-file repair stays version-bound and does not grant model success."""
import ast
import asyncio
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness.contracts import digest
from agent.harness.research_execution_feedback import repair_source_context
from agent.harness.research_focus import repair_action, selected_tools
from agent.harness.research_workspace import ResearchWorkspace

OLD = "class Candidate:\n    def fit(self, rows): return {}\n    def predict(self, inputs): return 0\n"
NEW = "def fit(training, seed):\n    return {'offset': 1}\ndef predict(model, inputs):\n    return {'value': model['offset']}\n"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def workspace(tmp_path):
    return ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", {"steps": []})


@pytest.mark.asyncio
async def test_checked_alias_replaces_complete_module_without_host_code_rewriting(tmp_path):
    service = workspace(tmp_path)
    original = await service.write_research_file("src/model.py", OLD)
    tool = next(tool for tool in service.workspace_tools() if tool.name == "write_session_file")
    schema = tool.args_schema.model_json_schema()
    assert "expected_sha256" in schema["properties"] and "expected_sha256" not in schema.get("required", [])
    result = await tool.ainvoke({"path": "src/model.py", "content": NEW, "expected_sha256": original["sha256"]})
    assert result["before_sha256"] == sha(OLD) and result["sha256"] == sha(NEW)
    actual = (service.workspace / "src/model.py").read_text()
    assert actual == NEW
    nodes = ast.parse(actual).body
    assert [node.name for node in nodes if isinstance(node, ast.FunctionDef)] == ["fit", "predict"]
    assert not any(isinstance(node, ast.ClassDef) for node in nodes)
    assert not service.proposals and not (service.development.root / "training").exists()
    assert service.receipts[-1]["arguments_sha256"] == digest({"path": "src/model.py", "content": NEW, "expected_sha256": sha(OLD)})


@pytest.mark.asyncio
async def test_stale_complete_replacement_preserves_newer_bytes_and_receipts(tmp_path):
    service = workspace(tmp_path)
    await service.write_research_file("src/model.py", OLD)
    await service.write_research_file("src/model.py", NEW)
    before = copy.deepcopy(service.receipts)
    with pytest.raises(ValueError, match="changed since"):
        await service.write_research_file("src/model.py", OLD, sha(OLD))
    assert (service.workspace / "src/model.py").read_text() == NEW and service.receipts == before


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, 0, False, "a" * 63, "A" * 64, "g" * 64])
async def test_invalid_explicit_digest_cannot_fall_back_to_unchecked_write(tmp_path, invalid):
    service = workspace(tmp_path)
    await service.write_research_file("report.md", "Existing")
    with pytest.raises(ValueError, match="full-file SHA256"):
        await service.write_research_file("report.md", "Replacement", invalid)
    assert (service.workspace / "report.md").read_text() == "Existing"


@pytest.mark.asyncio
async def test_legacy_create_and_revise_remain_available_for_non_model_files(tmp_path):
    service = workspace(tmp_path)
    created = await service.write_research_file("reports/result.md", "Draft")
    revised = await service.write_research_file("reports/result.md", "Revised")
    assert created == {"path": "reports/result.md", "sha256": sha("Draft")}
    assert revised == {"path": "reports/result.md", "sha256": sha("Revised")}
    assert service.receipts[-1]["arguments_sha256"] == digest({"path": "reports/result.md", "content": "Revised"})


@pytest.mark.asyncio
@pytest.mark.parametrize("linked_parent", [False, True])
async def test_checked_write_rejects_symlink_escape_and_preserves_external_file(tmp_path, linked_parent):
    service = workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.py").write_text(OLD)
    if linked_parent:
        (service.workspace / "linked").symlink_to(outside, target_is_directory=True)
        target = "linked/model.py"
    else:
        (service.workspace / "model.py").symlink_to(outside / "model.py")
        target = "model.py"
    with pytest.raises((OSError, ValueError)):
        await service.write_research_file(target, NEW, sha(OLD))
    assert (outside / "model.py").read_text() == OLD


def checked_writer(root, identity, barrier, output):
    service = ResearchWorkspace(Path(root) / "workspace", Path(root) / ("writer-" + str(identity)), {"steps": []})
    barrier.wait(timeout=10)
    try:
        asyncio.run(service.write_research_file("src/model.py", NEW + f"# writer {identity}\n", sha(OLD)))
    except ValueError as exc:
        output.put((identity, "rejected", str(exc)))
    else:
        output.put((identity, "written", ""))


def test_two_process_checked_writers_cannot_both_replace_one_version(tmp_path):
    service = workspace(tmp_path)
    asyncio.run(service.write_research_file("src/model.py", OLD))
    process_context = multiprocessing.get_context("fork")
    barrier, output = process_context.Barrier(2), process_context.Queue()
    processes = [process_context.Process(target=checked_writer, args=(str(tmp_path), i, barrier, output)) for i in range(2)]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        assert sorted(row[1] for row in results) == ["rejected", "written"]
        winner = next(row[0] for row in results if row[1] == "written")
        assert (service.workspace / "src/model.py").read_text() == NEW + f"# writer {winner}\n"
        assert "changed since" in next(row[2] for row in results if row[1] == "rejected")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        output.close()


def repair_fixture(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text(OLD)
    feedback = {"pending": [{"tool": "train_research_model", "source_file": "src/model.py",
                             "error": "Module entrypoints are missing", "training_attempt_id": "a" * 32}]}
    return repair_source_context(tmp_path, feedback)


@pytest.mark.parametrize("stage", ["evaluate_baseline", "evaluate_revision"])
def test_actual_source_repair_overrides_only_execution_action_preserving_bindings(tmp_path, stage):
    repair = repair_fixture(tmp_path)
    action = {"stage": stage, "tool": "write_session_file", "evaluation_file": "data/evaluation.json",
              "eligible_parent_ids": ["baseline-id"], "scientific_feedback": {"issues": ["Measure a justified revision"]}}
    before = copy.deepcopy(action)
    revised = repair_action(action, repair)
    assert action == before and revised["stage"] == "repair_model" and revised["resume_stage"] == stage
    assert revised["source_file"] == "src/model.py" and revised["expected_sha256"] == sha(OLD)
    for key in ("evaluation_file", "eligible_parent_ids", "scientific_feedback"):
        assert revised[key] == action[key]
    assert "write_session_file(path, content, expected_sha256)" in repair["instruction"]
    names = selected_tools({"plan": {"deliverable": "model"}, "model_development": {"next_action": revised}},
                           [SimpleNamespace(name=name) for name in ("train_research_model", "get_research_baseline")])
    assert names == ["train_research_model", "get_research_baseline"]


@pytest.mark.parametrize("stage", ["revise_scientific_model", "submit_for_review", "revise_submission", "finish_research"])
def test_repair_never_overrides_scientific_review_or_delivery_priority(tmp_path, stage):
    action = {"stage": stage, "instruction": "Keep independent Review"}
    assert repair_action(action, repair_fixture(tmp_path)) is action


@pytest.mark.parametrize("failure", ["absent", "missing", "oversized"])
def test_no_readable_verified_source_does_not_create_repair_priority(tmp_path, failure):
    action = {"stage": "evaluate_baseline", "tool": "write_session_file"}
    feedback = {"pending": []}
    if failure != "absent":
        feedback["pending"] = [{"tool": "train_research_model", "source_file": "src/model.py", "error": "Failed"}]
        if failure == "oversized":
            (tmp_path / "src").mkdir()
            (tmp_path / "src/model.py").write_text("x" * 12001)
    repair = repair_source_context(tmp_path, feedback)
    assert repair is None and repair_action(action, repair) is action


def test_redacted_source_does_not_recommend_complete_replacement(tmp_path, monkeypatch):
    from agent.harness import session_boundary
    original = session_boundary.public_value
    monkeypatch.setattr(session_boundary, "public_value", lambda value, root: "[redacted source]" if value == OLD else original(value, root))
    repair = repair_fixture(tmp_path)
    assert repair["redacted"] and "do not use it as complete replacement" in repair["instruction"]
    assert "write_session_file(path, content, expected_sha256)" not in repair["instruction"]
    assert repair_action({"stage": "evaluate_baseline"}, repair)["tool"] == "edit_session_file"


@pytest.mark.asyncio
@pytest.mark.parametrize("within_iteration", [False, True])
async def test_real_preflight_failure_reaches_next_native_request_with_repair_priority(tmp_path, monkeypatch, within_iteration):
    from test_focused_repair import test_repair_context_reaches_executor_after_failed_fit_without_budget_reset

    from agent.harness import autonomous_research as module

    original = module.AutonomousResearch
    observed = []

    class ObserveNativeRequests(original):
        def __init__(self, *args, executor, **kwargs):
            async def capture(request, service, events):
                context = json.loads(request["text"])
                if "repair_source" in context["model_development"]:
                    action = context["model_development"]["next_action"]
                    assert action["stage"] == "repair_model"
                    assert action["expected_sha256"] == context["model_development"]["repair_source"]["sha256"]
                    assert action["evaluation_file"] == context["model_development"]["prepared_dataset"]["evaluation_file"]
                    assert "train_research_model" in context["model_development"]["selected_tools"]
                    observed.append(action)
                await executor(request, service, events)
            super().__init__(*args, executor=capture, **kwargs)

    monkeypatch.setattr(module, "AutonomousResearch", ObserveNativeRequests)
    await test_repair_context_reaches_executor_after_failed_fit_without_budget_reset(
        tmp_path, monkeypatch, within_iteration, "regression")
    assert len(observed) == 1
