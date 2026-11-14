"""Offline acceptance-contract tests; no SDK, provider or goal is launched."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.rsi import run_scientist_runtime_matrix as matrix
from agent.harness import Profile
from agent.harness.contracts import file_hash

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    from agent.harness import llm_policy
    monkeypatch.setattr(llm_policy, "assert_listed", lambda *args: None)
    raw = json.loads((ROOT / "configs/harness/free_execution_catalog.json").read_text())
    enabled = [row for row in raw["models"] if row.get("enabled", True) and matrix.zero_price(row)]
    raw["models"] = [enabled[0], enabled[1], {**copy.deepcopy(enabled[0]), "id": "disabled", "enabled": False},
                     {**copy.deepcopy(enabled[1]), "id": "paid", "pricing": {"input_per_million": 1, "output_per_million": 1}}]
    raw["harnesses"] = {"science_expert": ["expert-deepseek-harness", "expert-codex"]}
    raw.pop("defaults")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(raw))
    profile = Profile.from_dict(json.loads((ROOT / "configs/harness/free_science_goal.json").read_text()))
    runtime = tmp_path / "native-runtime"
    runtime.touch()
    return path, profile, {"dsh_bin": str(runtime), "codex_bin": str(tmp_path / "missing")}


def test_inventory_retains_every_free_combination_and_runtime_failure(inventory, tmp_path, monkeypatch):
    import agent.harness.catalog as catalog
    monkeypatch.setattr(catalog, "resolve_credential", lambda *args: pytest.fail("dry-run resolved credentials"))
    path, profile, runtimes = inventory
    result = matrix.build_plan(path, profile, runtimes, tmp_path)
    assert result["coverage"] == {"models": 2, "harnesses": 2, "cells": 4}
    assert result["execution_policy"] == matrix.EXECUTION_POLICY
    assert all(c["execution_policy"] == matrix.EXECUTION_POLICY for c in result["cells"])
    assert len({(c["model_id"], c["harness"]) for c in result["cells"]}) == 4
    assert all(c["availability"]["effective"] == "not_probed" for c in result["cells"])
    assert all(c["compression"]["status"] == "not_observed" for c in result["cells"])
    blocked = [c for c in result["cells"] if c["harness"] == "expert-codex"]
    assert all("runtime_missing" in c["compatibility"]["reasons"] for c in blocked)
    assert all(c["model_id"] not in {"paid", "disabled"} for c in result["cells"])


def test_policy_rejections_are_cells_not_dropped_models(inventory, tmp_path, monkeypatch):
    from agent.harness import llm_policy
    path, profile, runtimes = inventory
    monkeypatch.setattr(llm_policy, "assert_listed", lambda *args: (_ for _ in ()).throw(ValueError("unpublished")))
    result = matrix.build_plan(path, profile, runtimes, tmp_path)
    assert len(result["cells"]) == 4
    assert all(c["compatibility"]["status"] == "blocked" for c in result["cells"])


@pytest.mark.asyncio
async def test_dry_run_writes_every_cell_without_creating_goals(inventory, tmp_path, monkeypatch):
    path, profile, runtimes = inventory
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.public()))
    seen = []
    monkeypatch.setattr(matrix, "default_runtimes", lambda root: (seen.append(root), runtimes)[1])
    async def forbidden(*args):
        pytest.fail("dry-run started a goal")
    monkeypatch.setattr(matrix, "execute_cell", forbidden)
    args = SimpleNamespace(execute=False, case=None, catalog=path, profile=profile_path,
                           runtime_root=tmp_path / "install", output=tmp_path / "out")
    result = await matrix.run(args)
    assert seen == [args.runtime_root]
    assert result["status"] == "dry_run"
    assert len(list(args.output.glob("cell_*/report.json"))) == 4
    assert not list(args.output.rglob("goal.json"))
    assert (args.output / "allocation.json").is_file()


@pytest.mark.asyncio
async def test_execute_requires_explicit_case_before_output_or_runtime_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "default_runtimes", lambda *_: pytest.fail("premature discovery"))
    with pytest.raises(ValueError, match="predeclared"):
        await matrix.run(SimpleNamespace(execute=True, case=None, output=tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def smoke():
    return {"schema": 1, "id": matrix.SMOKE_ID, "kind": "non_research_smoke",
            "limits": {"max_cycles": 1, "model_calls": 12, "tool_calls": 8, "wall_seconds": 120}}


@pytest.mark.parametrize("change", [
    {"objective": "Invent a new scientific project"},
    {"id": "automatic-topic"},
    {"kind": "automatic"},
    {"limits": {"max_cycles": True, "model_calls": 12, "tool_calls": 8, "wall_seconds": 120}},
])
def test_smoke_cannot_be_relabelled_as_automatic_science(tmp_path, change):
    value = {**smoke(), **change}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        matrix.load_case(path)


def test_approved_topic_requires_approval_and_bound_inputs(tmp_path):
    path = tmp_path / "case.json"
    data = tmp_path / "input.json"
    data.write_text("{}")
    value = {**smoke(), "kind": "approved_topic", "id": "explicit-topic", "objective": "Previously reviewed objective",
             "inputs": [{"path": "input.json", "sha256": file_hash(data)}]}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="approval"):
        matrix.load_case(path)
    value["approval"] = {"status": "approved", "reference": "user-approved-topic-record"}
    path.write_text(json.dumps(value))
    assert matrix.load_case(path)[1] == [data]
    data.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        matrix.load_case(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("status,code", [(401, "authentication"), (402, "quota"), (429, "rate_limit")])
def test_provider_blocks_are_not_scientific_or_platform_failures(tmp_path, status, code):
    write_rows(tmp_path / "episodes/cycle_000/model_requests.jsonl",
               [{"event": "request", "call": 1}, {"event": "response", "call": 1, "status": status, "completed": False}])
    result = matrix.classify(tmp_path, {"status": "interrupted", "cycles": [{"status": "failed"}]}, "approved_topic")
    assert result["availability"]["effective"] == code
    assert result["platform"]["status"] == "unverified"
    assert result["research"]["status"] == "incomplete"


def successful_transport(tmp_path):
    folder = tmp_path / "episodes/cycle_000"
    write_rows(folder / "model_requests.jsonl",
               [{"event": "request", "call": 1}, {"event": "response", "call": 1, "status": 200, "completed": True}])
    (folder / "events.json").write_text(json.dumps([{"kind": "completed", "event_audit": {"passed": True}}]))
    return folder


def test_scientific_rejection_does_not_erase_verified_runtime(tmp_path):
    successful_transport(tmp_path)
    result = matrix.classify(tmp_path, {"status": "budget_exhausted", "cycles": [{"status": "needs_revision"}],
                                       "progress": {"sc_revision": {"issues": ["No measured benefit"]}}}, "approved_topic")
    assert result["platform"]["status"] == "pass"
    assert result["research"]["status"] == "rejected"
    assert result["compression"]["status"] == "not_observed"


def test_runtime_import_error_is_separate_from_research_incompletion(tmp_path):
    folder = successful_transport(tmp_path)
    (folder / "failure.json").write_text(json.dumps({"error_type": "ModuleNotFoundError", "message": "private detail"}))
    result = matrix.classify(tmp_path, {"status": "interrupted", "cycles": [{"status": "failed"}]}, "approved_topic")
    assert result["platform"]["status"] == "error"
    assert "private detail" not in json.dumps(result)


@pytest.mark.parametrize("variant,expected", [
    ("valid", "continuation_unverified"), ("other_session", "continuation_unverified"),
    ("reverse_order", "continuation_unverified"), ("summary", "continuation_unverified"),
    ("blocked", "continuation_unverified"), ("prose", "not_observed"),
])
def test_compression_requires_native_boundary_and_paired_continuation(tmp_path, variant, expected):
    boundary = {"kind": "context_compaction", "phase": "completed", "session_id": "native-1"}
    start = {"kind": "tool_call_start", "session_id": "native-1", "tool_call_id": "call-1"}
    end = {"kind": "tool_result", "session_id": "native-1", "tool_call_id": "call-1", "is_error": False}
    if variant == "other_session":
        start["session_id"] = end["session_id"] = "native-2"
    if variant in {"summary", "blocked"}:
        boundary["phase"] = variant
    rows = [boundary, start, end]
    if variant == "reverse_order":
        rows = [boundary, end, start]
    if variant == "prose":
        rows = [{"kind": "assistant", "text": "Context compaction completed and continued"}]
    path = tmp_path / "episodes/cycle_000/native/events.jsonl"
    if variant != "prose":
        raw_boundary = native_boundary(path, phase=boundary["phase"],
                                       error="blocked" if variant == "blocked" else None)
        boundary.update({k: v for k, v in raw_boundary.items() if k not in {"phase", "session_id"}})
    write_rows(path, rows)
    assert matrix.compression_evidence(tmp_path)["status"] == expected


@pytest.mark.asyncio
async def test_one_failed_cell_does_not_drop_later_allocations(inventory, tmp_path, monkeypatch):
    path, profile, runtimes = inventory
    runtimes["codex_bin"] = runtimes["dsh_bin"]
    monkeypatch.setattr(matrix, "default_runtimes", lambda *_: runtimes)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.public()))
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(smoke()))
    observed = []
    async def fake(cell, folder, case, inputs, catalog, runtime):
        observed.append(cell["id"])
        if len(observed) == 1:
            raise RuntimeError("Do not copy private provider text")
        cell["execution"] = "finished"
    monkeypatch.setattr(matrix, "execute_cell", fake)
    monkeypatch.setenv("VENUS_EXECUTION_CATALOG", str(path))
    result = await matrix.run(SimpleNamespace(execute=True, case=case_path, catalog=path, profile=profile_path,
                                              runtime_root=tmp_path, output=tmp_path / "out"))
    assert len(observed) == len(result["cells"]) == 4
    assert result["cells"][0]["platform"] == {"status": "error", "error_type": "RuntimeError"}
    assert len(list((tmp_path / "out").glob("cell_*/report.json"))) == 4
    assert "private provider" not in (tmp_path / "out/report.json").read_text()


def native_boundary(path, *, phase="completed", error=None):
    raw = {"type": "compaction/end", "data": {"error": error}}
    raw_path = path.parent / "runtime_events.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as stream:
        stream.write(json.dumps(raw) + "\n")
    return {"kind": "context_compaction", "phase": phase, "session_id": "native-1",
            "native_type": "compaction/end", "native_event_path": raw_path.name,
            "native_event_sha256": matrix.hashlib.sha256(json.dumps(
                raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}


def test_compaction_hash_and_recovery_are_independent_of_old_failures(tmp_path):
    path = tmp_path / "episodes/cycle_000/native/events.jsonl"
    failed = native_boundary(path, phase="failed", error="native segment too short")
    completed = native_boundary(path)
    continuation = [
        {"kind": "tool_call_start", "session_id": "native-1", "tool_call_id": "c"},
        {"kind": "tool_result", "session_id": "native-1", "tool_call_id": "c", "is_error": False},
    ]
    write_rows(path, [failed, completed, *continuation])
    result = matrix.compression_evidence(tmp_path)
    assert result["status"] == "continuation_unverified"
    assert result["recovered"] is False and result["observed_failure_count"] == 1
    assert result["recovery_candidate"] is True
    completed["native_event_sha256"] = "0" * 64
    write_rows(path, [completed, *continuation])
    assert matrix.compression_evidence(tmp_path)["status"] == "continuation_unverified"


def test_compaction_completion_cannot_reference_a_native_start_or_foreign_path(tmp_path):
    path = tmp_path / "episodes/cycle_000/native/events.jsonl"
    boundary = native_boundary(path)
    raw = {"type": "compaction/start", "data": {}}
    write_rows(path.parent / "runtime_events.jsonl", [raw])
    boundary["native_type"] = "compaction/start"
    boundary["native_event_sha256"] = matrix.hashlib.sha256(json.dumps(
        raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert not matrix.native_type_hash_matches(boundary, path)
    boundary["native_event_path"] = "../runtime_events.jsonl"
    assert not matrix.native_type_hash_matches(boundary, path)


def test_readonly_policy_never_opens_the_mutating_store_connection(tmp_path, monkeypatch):
    import os
    import sqlite3
    from mcp_tools.tools import capability_policy
    from mcp_tools.tools.capability_snapshot import read
    database = tmp_path / "capabilities.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE switches(key TEXT, enabled INTEGER, revision INTEGER, selector TEXT)")
        connection.execute("INSERT INTO switches VALUES('llm:fixture', 0, 3, '{}')")
    before = (database.read_bytes(), database.stat().st_mtime_ns)
    monkeypatch.setattr(capability_policy.CapabilityStore, "connect",
                        lambda *_: pytest.fail("mutating connection used"))
    monkeypatch.setattr(capability_policy, "CapabilityStore",
                        lambda: SimpleNamespace(path=database, snapshot=None))
    monkeypatch.delenv("VENUS_CAPABILITY_POLICY_SNAPSHOT", raising=False)
    with matrix.readonly_policy():
        snapshot = Path(os.environ["VENUS_CAPABILITY_POLICY_SNAPSHOT"])
        assert snapshot.stat().st_mode & 0o077 == 0
        assert read(snapshot)["llm:fixture"]["enabled"] == 0
    assert "VENUS_CAPABILITY_POLICY_SNAPSHOT" not in os.environ
    assert not snapshot.exists()
    assert (database.read_bytes(), database.stat().st_mtime_ns) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("goal_status", ["budget_exhausted", "waiting_for_input"])
async def test_execute_cell_uses_full_goal_entry_without_persisting_credentials(inventory, tmp_path, monkeypatch, goal_status):
    import agent.harness.autonomous_research as research
    import agent.harness.catalog as catalog_module
    from agent.harness.catalog import ExecutionCatalog
    path, profile, runtimes = inventory
    cell = matrix.build_plan(path, profile, runtimes, tmp_path)["cells"][0]
    catalog = ExecutionCatalog(ROOT, profile, path=path, free_only=True, include_unlisted=True)
    folder = tmp_path / "cell"
    folder.mkdir()
    calls = []
    def create(root, selected, objective, limits, *, unattended):
        assert unattended is True
        calls.append(("create", selected, objective, limits))
        root.mkdir()
        return SimpleNamespace(root=root)
    class Runner:
        def __init__(self, store, credential, selected_runtimes, **kwargs):
            assert credential == "synthetic-private-credential"
            assert selected_runtimes == runtimes
            self.store = store
        async def run(self):
            calls.append(("run",))
            return {"status": goal_status, "cycles": []}
    monkeypatch.setattr(research, "create", create)
    monkeypatch.setattr(research, "AutonomousResearch", Runner)
    monkeypatch.setattr(catalog_module, "resolve_credential", lambda *args: "synthetic-private-credential")
    case = {**smoke(), "kind": "approved_topic", "objective": "Explicit previously approved objective"}
    await matrix.execute_cell(cell, folder, case, [], catalog, runtimes)
    assert [call[0] for call in calls] == ["create", "run"]
    assert calls[0][2] == case["objective"]
    assert cell["execution"] == ("blocked" if goal_status == "waiting_for_input" else "finished")
    if goal_status == "waiting_for_input":
        assert cell["blocker"] == "missing_research_input"
        assert cell["research"]["status"] == "blocked_missing_input"
        assert cell["platform"]["status"] == "unverified"
    assert "synthetic-private-credential" not in json.dumps(cell)
    assert not any("synthetic-private-credential" in f.read_text()
                   for f in folder.rglob("*") if f.is_file())


@pytest.mark.parametrize("status,signals,expected", [
    (400, ["context_limit"], "context_limit"),
    (413, ["context_limit"], "context_limit"),
    (401, [], "authentication"),
    (403, [], "access"),
    (429, [], "rate_limit"),
])
def test_generic_native_error_keeps_matching_external_diagnostic(tmp_path, status, signals, expected):
    folder = successful_transport(tmp_path)
    write_rows(folder / "model_requests.jsonl", [
        {"event": "request", "call": 1},
        {"event": "upstream_rejection", "call": 1, "status": status, "reported_signals": signals},
        {"event": "response", "call": 1, "status": status, "completed": False},
    ])
    (folder / "failure.json").write_text(json.dumps({"error_type": "RuntimeError", "message": "runtime_error"}))
    result = matrix.classify(tmp_path, {"status": "interrupted", "cycles": [{"status": "failed"}]}, "approved_topic")
    assert result["availability"]["effective"] == expected
    assert result["platform"]["status"] == "unverified"


def test_existing_context_diagnostic_is_not_overwritten_by_generic_message(tmp_path):
    folder = successful_transport(tmp_path)
    (folder / "failure.json").write_text(json.dumps(
        {"error_type": "RuntimeError", "message": "runtime_error", "code": "context_limit"}))
    result = matrix.classify(tmp_path, {"status": "interrupted", "cycles": [{"status": "failed"}]}, "approved_topic")
    assert result["availability"]["effective"] == "context_limit"
    assert result["platform"]["status"] == "unverified"


@pytest.mark.parametrize("kind", ["empty", "timeout"])
def test_empty_or_timed_out_provider_is_not_platform_pass(tmp_path, kind):
    folder = successful_transport(tmp_path)
    response = {"event": "response", "call": 1, "status": 200, "completed": True,
                "response_completion": {"malformed": False, "provider_error": False,
                    "streaming": False, "content_present": False, "tools_present": False,
                    "finish_reasons": ["stop"]}}
    if kind == "timeout":
        response = {"event": "response", "call": 1, "completed": False, "error_type": "ReadTimeout"}
    write_rows(folder / "model_requests.jsonl", [{"event": "request", "call": 1}, response])
    result = matrix.classify(tmp_path, {"status": "interrupted", "cycles": [{"status": "completed"}]}, "approved_topic")
    assert result["availability"]["effective"] == ("model_response_empty" if kind == "empty" else "upstream_timeout")
    assert result["platform"]["status"] == "unverified"


@pytest.mark.parametrize("proof,expected", [
    ("completed", "observed_continued"),
    ("notice", "continuation_unverified"),
    ("rejected", "continuation_unverified"),
    ("outside_goal", "continuation_unverified"),
])
def test_compression_success_requires_shared_verifier_and_host_binding(tmp_path, monkeypatch, proof, expected):
    from agent.harness import native_context_recovery
    path = tmp_path / "episodes/cycle_000/native/events.jsonl"
    boundary = native_boundary(path)
    write_rows(path, [boundary,
        {"kind": "tool_call_start", "session_id": "native-1", "tool_call_id": "c"},
        {"kind": "tool_result", "session_id": "native-1", "tool_call_id": "c", "is_error": False}])
    calls = []
    def verify(native):
        calls.append(native)
        if proof == "rejected":
            raise ValueError("native_notification_binding_changed")
        return [{"boundary_phase": proof, "session_id": "native-1"}]
    monkeypatch.setattr(native_context_recovery, "verify_native_continuation", verify)
    native = {"root": tmp_path.parent if proof == "outside_goal" else path.parent}
    result = matrix.compression_evidence(tmp_path, native_contexts=[native])
    assert result["status"] == expected
    assert len(calls) == (0 if proof == "outside_goal" else 1)


@pytest.mark.parametrize("followup", [False, True])
def test_persisted_web_binding_roundtrip_reuses_strict_native_audit(tmp_path, followup):
    from agent.harness.adapters import _dsh_observation
    folder = tmp_path / "episodes/cycle_000"
    native_roots = [folder / "native/run"]
    if followup:
        native_roots.append(folder / "execution_retry/native/run")
    failed_calls = []
    for index, native_root in enumerate(native_roots):
        native_root.mkdir(parents=True)
        before, after = 2 + index * 4, 6 + index * 4
        sid = "native-session-" + str(index)
        failed_calls.append(before + 1)
        raw_events = [
            (before + 1, {"type": "compaction/start", "data": {"compactionId": "c"}}),
            (before + 2, {"type": "compaction/end", "data": {"compactionId": "c"}}),
            (before + 3, {"type": "tool/call", "data": {
                "callId": "t", "name": "mcp__venusfactory__list_session_files", "arguments": {}}}),
            (before + 3, {"type": "tool/result", "data": {"message": {"content": [{
                "type": "tool-result", "toolCallId": "t", "isError": False,
                "content": [{"type": "text", "text": "{}"}]}]}}}),
            (after, {"type": "turn/end", "data": {"reason": {"kind": "completed"}}}),
        ]
        normalized, raw, envelopes = [], [], []
        for call, event in raw_events:
            envelope = {"sessionId": sid, "event": event}
            normalized.extend(_dsh_observation(event, sid, call, envelope))
            raw.append(event)
            envelopes.append(envelope)
        write_rows(native_root / "events.jsonl", normalized)
        write_rows(native_root / "runtime_events.jsonl", raw)
        write_rows(native_root / "runtime_notifications.jsonl", envelopes)
        result = {"status": "completed", "session_id": sid, "sandbox_stopped": True}
        (native_root / "native_result.json").write_text(json.dumps(result))
        tools = ["list_session_files"]
        binding = {"schema": "web_native_phase.v1", "harness": "deepseek-harness",
                   "session_id": sid, "native_result_sha256": matrix.digest(result),
                   "tools": tools, "tools_sha256": matrix.digest(tools), "tool_limit": 4,
                   "calls_before": before, "calls_after": after}
        (native_root / "native_binding.json").write_text(json.dumps(binding))
    records = []
    for call in range(1, after + 2):
        status = 400 if call in failed_calls else 200
        records.append({"event": "request", "call": call})
        if status == 400:
            records.append({"event": "upstream_rejection", "call": call, "status": 400,
                            "reported_signals": ["context_limit"]})
        records.append({"event": "response", "call": call, "status": status,
                        "completed": status == 200, "error_type": None})
    write_rows(folder / "model_requests.jsonl", records)
    (folder / "events.json").write_text(json.dumps([{"kind": "completed", "event_audit": {"passed": True}}]))
    state = {"definition": {"profile": {"harness": "expert-deepseek-harness"}},
             "status": "budget_exhausted", "cycles": [{"status": "needs_revision"}]}
    contexts = matrix.load_cycle_native_contexts(folder, state)
    assert [(c["calls_before"], c["calls_after"]) for c in contexts] == (
        [(2, 6), (6, 10)] if followup else [(2, 6)])
    classified = matrix.classify(tmp_path, state, "approved_topic")
    assert classified["platform"]["status"] == "pass"
    assert classified["research"]["status"] == "incomplete"
    assert classified["compression"]["status"] == "observed_continued"
    # A persisted binding can never relabel a changed raw session as recovery.
    notifications = native_roots[-1] / "runtime_notifications.jsonl"
    rows = [json.loads(line) for line in notifications.read_text().splitlines()]
    rows[1]["sessionId"] = "foreign-session"
    write_rows(notifications, rows)
    classified = matrix.classify(tmp_path, state, "approved_topic")
    assert classified["platform"]["status"] == "unverified"
    assert classified["compression"]["status"] == "continuation_unverified"


def test_automatic_confirmations_are_not_scientific_acceptance(tmp_path):
    folder = successful_transport(tmp_path)
    (folder / "events.json").write_text(json.dumps([
        {"kind": "automatic_approval", "policy": matrix.EXECUTION_POLICY, "decision": "approve"},
        {"kind": "automatic_approval", "policy": "unknown", "decision": "approve"},
        {"kind": "completed", "event_audit": {"passed": True}},
    ]))
    result = matrix.classify(tmp_path, {"status": "waiting_for_input", "cycles": [{"status": "needs_revision"}]}, "approved_topic")
    assert result["automation"] == {"policy": matrix.EXECUTION_POLICY, "automatic_confirmations": 1, "input_blocked": True}
    assert result["research"]["status"] == "blocked_missing_input"
    assert result["compression"]["status"] == "not_observed"
