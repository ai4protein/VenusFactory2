"""Real-file evidence boundaries and queued-worker integration; no model/API calls."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_tools.scheduler import analysis_evidence as evidence
from mcp_tools.scheduler.receipt_binding import value_digest


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def case(tmp_path):
    workspace = tmp_path / "session"
    workspace.mkdir()
    (workspace / "sequence.fasta").write_text(">control\nACDE\n")
    spec = {
        "name": "calculate_physchem_from_fasta", "route": "native",
        "workspace": str(workspace), "session_id": "session-a",
        "arguments": {"fasta_file": "/workspace/sequence.fasta", "output_file": "/workspace/result.json"},
    }
    return workspace, spec


def successful_result():
    return {"is_error": False, "value": {"status": "success", "length": 4}}


def test_real_files_and_original_arguments_bound_to_worker_receipt(case):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text('{"length":4}')
    result = successful_result()
    receipt = capture.finish(result)
    assert receipt == {
        "version": 1, "tool": spec["name"], "category": "sequence_analysis",
        "workspace_sha256": digest(str(workspace).encode()),
        "arguments_sha256": value_digest(spec["arguments"]),
        "result_sha256": value_digest(result["value"]),
        "input_files": {"sequence.fasta": digest((workspace / "sequence.fasta").read_bytes())},
        "output_files": {"result.json": digest(b'{"length":4}')},
    }
    assert capture.finish(result) is None


@pytest.mark.parametrize("category", sorted(evidence.ANALYSIS_CATEGORIES))
def test_only_scientific_categories_produce_evidence(case, category):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, category)
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result())["category"] == category


@pytest.mark.parametrize("category", ["workspace", "model_training", "protein_database", "web_search", "platform_help", "sandbox_execution"])
def test_non_analysis_tools_cannot_mint_analysis_receipts(case, category):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, category)
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("status", ["pending", "queued", "running", "submitted", "accepted", "processing", "failed", "error", "cancelled"])
def test_remote_ticket_or_failed_value_is_not_computation_evidence(case, status):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, "remote_analysis")
    (workspace / "result.json").write_text("{}")
    result = {"is_error": False, "value": json.dumps({"status": status, "ticket": "remote-1"})}
    assert capture.finish(result) is None


@pytest.mark.parametrize("result", [
    {"is_error": True, "value": {"status": "success"}},
    {"is_error": False, "value": {"result": {"state": "running"}}},
    {"is_error": False, "value": {"is_error": True}},
    {"is_error": False, "value": ""},
    {"is_error": False, "value": "  "},
    {"is_error": False, "value": "null"},
    {"is_error": False, "value": "{}"},
    {"is_error": False, "value": []},
    {"is_error": False},
    {"value": {"status": "success"}},
])
def test_only_explicit_successful_nonempty_result_is_attested(case, result):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text("{}")
    assert capture.finish(result) is None


def test_scalar_input_and_result_are_bound_without_reading_returned_host_paths(case, monkeypatch):
    workspace, spec = case
    spec["arguments"] = {"sequence": "ACDE", "accession": "P12345"}
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    result = {"is_error": False, "value": {"length": 4, "file_info": {"file_path": "/outside/credentials"}}}
    original_open = os.open

    def forbid_untrusted_path(path, *args, **kwargs):
        assert path != "/outside/credentials"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", forbid_untrusted_path)
    receipt = capture.finish(result)
    assert receipt["input_files"] == receipt["output_files"] == {}
    assert receipt["arguments_sha256"] == value_digest(spec["arguments"])
    assert receipt["result_sha256"] == value_digest(result["value"])


@pytest.mark.parametrize("mutation", ["edit", "replace_same_bytes", "remove"])
def test_input_changed_during_execution_cannot_be_attested(case, mutation):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    source = workspace / "sequence.fasta"
    if mutation == "edit":
        source.write_text(">different\nAAAA\n")
    elif mutation == "replace_same_bytes":
        other = workspace / "replacement"
        other.write_bytes(source.read_bytes())
        other.replace(source)
    else:
        source.unlink()
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("field", ["fasta_file", "output_file"])
@pytest.mark.parametrize("unsafe", ["../other/file", "/elsewhere/file", "/workspace/.private", "bad\\file"])
def test_file_arguments_cannot_leave_session(case, field, unsafe):
    workspace, spec = case
    spec["arguments"][field] = unsafe
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "parent_symlink"])
def test_untrusted_input_inode_types_fail_without_blocking(case, kind):
    workspace, spec = case
    source = workspace / "sequence.fasta"
    content = source.read_bytes()
    source.unlink()
    outside = workspace.parent / "outside"
    outside.mkdir()
    target = outside / "sequence.fasta"
    target.write_bytes(content)
    if kind == "symlink":
        source.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, source)
    elif kind == "fifo":
        os.mkfifo(source)
    else:
        (workspace / "linked").symlink_to(outside, target_is_directory=True)
        spec["arguments"]["fasta_file"] = "/workspace/linked/sequence.fasta"
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_output_replaced_with_link_or_special_file_is_not_attested(case, kind):
    workspace, spec = case
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    output = workspace / "result.json"
    outside = workspace.parent / "secret"
    outside.write_text("{}")
    if kind == "symlink":
        output.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, output)
    else:
        os.mkfifo(output)
    assert capture.finish(successful_result()) is None


def test_output_directory_only_attests_current_products(case):
    workspace, spec = case
    out = workspace / "outputs"
    out.mkdir()
    (out / "old.json").write_text("old")
    spec["arguments"].pop("output_file")
    spec["arguments"]["out_dir"] = "/workspace/outputs"
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (out / "new.json").write_text("new")
    receipt = capture.finish(successful_result())
    assert receipt["output_files"] == {"outputs/new.json": digest(b"new")}
    # Later modification cannot silently refresh the recorded product hash.
    (out / "new.json").write_text("altered")
    assert receipt["output_files"]["outputs/new.json"] != digest((out / "new.json").read_bytes())


def test_unchanged_preexisting_output_is_not_a_current_product(case):
    workspace, spec = case
    (workspace / "result.json").write_text("old")
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    assert capture.finish(successful_result()) is None


def test_directory_symlink_swap_between_capture_and_finish_is_refused(case):
    workspace, spec = case
    out = workspace / "outputs"
    out.mkdir()
    spec["arguments"]["output_file"] = "/workspace/outputs/result.json"
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    out.rmdir()
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "result.json").write_text("{}")
    out.symlink_to(outside, target_is_directory=True)
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_capture_is_bounded_across_inputs_and_outputs(case, monkeypatch, limit):
    workspace, spec = case
    if limit == "files":
        monkeypatch.setattr(evidence, "MAX_FILES", 1)
    else:
        monkeypatch.setattr(evidence, "MAX_BYTES", (workspace / "sequence.fasta").stat().st_size + 1)
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.parametrize("field,value", [("route", "mcp"), ("workspace", None), ("session_id", None)])
def test_unscoped_or_non_native_spec_has_no_attestation(case, field, value):
    workspace, spec = case
    spec[field] = value
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    (workspace / "result.json").write_text("{}")
    assert capture.finish(successful_result()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["success", "failure", "mutate_input"])
async def test_real_worker_owns_receipt_not_tool_self_report(case, monkeypatch, behavior):
    from mcp_tools.scheduler.worker import execute

    workspace, spec = case
    original_arguments = dict(spec["arguments"])
    monkeypatch.setattr("mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed", lambda *args: None)
    for key in ("VENUS_TOOL_WORKSPACE", "TEMP_OUTPUTS_DIR", "VENUS_RSI_WORKSPACE", "VENUS_RSI_REGISTRY_ROOT", "VENUS_RSI_SKILLS_ROOT"):
        monkeypatch.setenv(key, "")

    async def compute(arguments, config):
        assert config["configurable"]["session_id"] == "session-a"
        assert Path(arguments["fasta_file"]).is_absolute()
        sequence = "".join(Path(arguments["fasta_file"]).read_text().splitlines()[1:])
        value = {"status": "error" if behavior == "failure" else "success", "length": len(sequence),
                 "analysis_evidence": {"version": 999, "output_files": {"forged": "0" * 64}}}
        Path(arguments["output_file"]).write_text(json.dumps(value))
        if behavior == "mutate_input":
            Path(arguments["fasta_file"]).write_text(">replacement\nAAAAA\n")
        return value

    tool = SimpleNamespace(ainvoke=compute, metadata={"tool_category": "sequence_analysis"})
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {spec["name"]: tool})
    result = await execute(spec)
    assert result["value"]["analysis_evidence"]["version"] == 999  # Ordinary result is unchanged.
    if behavior == "success":
        receipt = result["analysis_evidence"]
        assert receipt["version"] == 1
        assert receipt["arguments_sha256"] == value_digest(original_arguments)
        assert receipt["output_files"] == {"result.json": digest((workspace / "result.json").read_bytes())}
    else:
        assert "analysis_evidence" not in result
    assert result["is_error"] is (behavior == "failure")


@pytest.mark.asyncio
async def test_unscoped_legacy_worker_does_not_trust_tool_receipt(monkeypatch):
    from mcp_tools.scheduler.worker import execute

    async def compute(arguments, config):
        return {"status": "success", "analysis_evidence": {"version": 999}}

    tool = SimpleNamespace(ainvoke=compute, metadata={"tool_category": "sequence_analysis"})
    monkeypatch.setattr("mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed", lambda *args: None)
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {"calculate_physchem_from_sequence": tool})
    result = await execute({"name": "calculate_physchem_from_sequence", "route": "native",
                            "arguments": {"sequence": "ACDE"}, "session_id": "legacy"})
    assert "analysis_evidence" not in result
    assert result["value"]["analysis_evidence"]["version"] == 999


def test_whole_workspace_replacement_cannot_attest_scalar_result(case):
    workspace, spec = case
    spec["arguments"] = {"sequence": "ACDE"}
    capture = evidence.AnalysisCapture(spec, workspace, "sequence_analysis")
    workspace.rename(workspace.parent / "old-session")
    workspace.mkdir()
    assert capture.finish(successful_result()) is None
