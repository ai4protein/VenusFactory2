"""Interrupted jobs keep their failed requests, provenance and cumulative limits."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from test_mcba_plugin import complete_report

from agent.harness import ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.rsi.case_evidence import canonical_claims
from agent.rsi.reproduction_recovery import RecoveryLedger, load_source, read
from agent.rsi.versions import VersionStore


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def phase(run, name, profile, job, calls=4):
    folder = run / name
    folder.mkdir(parents=True)
    capsource = run / (name + "_capability/source")
    capsource.mkdir(parents=True)
    (capsource / "generated_tools.py").write_text("# test capability")
    versions = VersionStore(run / (name + "_capability/bundles"))
    manifest = versions.snapshot(capsource)
    cap = versions.root / "versions" / manifest["version_id"]
    tool = "submit_reproduction" if name == "launch" else "inspect_case_evidence"
    args = {"plugin_id": job["plugin_id"], "parameters_json": json.dumps(job["parameters"])}
    events = [
        {"kind": "tool_call_start", "tool_call_id": "call_a", "tool_name": "mcp__venusfactory__" + tool, "args": args},
        {
            "kind": "tool_result",
            "tool_call_id": "call_a",
            "is_error": False,
            "output": json.dumps({"success": True, **job}),
        },
    ]
    jsonl(folder / "events.jsonl", events)
    requests = []
    for i in range(calls):
        requests += [
            {"event": "request", "call": i},
            {"event": "response", "call": i, "status": 400 if i == calls - 1 else 200, "completed": i != calls - 1},
        ]
    jsonl(folder / "model_requests.jsonl", requests)
    lock = {
        "profile": profile,
        "task": {"tools": [tool], "inputs": {}, "capabilities": {p.name: file_hash(p) for p in cap.iterdir()}},
    }
    lock["sha256"] = digest(lock)
    save(folder / "profile.lock.json", lock)
    runtime = {
        "profile_sha256": lock["sha256"],
        "binding_intact": True,
        "host_wall_seconds": 10,
        "status": "runtime_error",
        "session_id": name,
        "audit": {
            "events": audit_trace(folder / "events.jsonl", [tool], profile["budget"]["tool_calls"]),
            "requests": audit_requests(folder / "model_requests.jsonl"),
        },
    }
    save(folder / "run.json", runtime)
    return runtime


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "run"
    entry = "scripts/rsi/run_mcba_case.py"
    impl = run / "implementation" / entry
    impl.parent.mkdir(parents=True)
    impl.write_text("# original source")
    profile = Profile(
        "science_agent", "langgraph", ModelBinding("fixture", "glm-5.1", "https://example.com/v1")
    ).public()
    protocol = {
        "profile": profile,
        "plugin": "mcba.multisubstrate.v1",
        "parameters": {"operation": "reproduce"},
        "maximum_jobs": 1,
        "implementation": {entry: file_hash(impl)},
    }
    domain = run / "controller/job/report.json"
    raw = complete_report()
    save(domain, raw)
    job = {
        "job_id": "job",
        "plugin_id": protocol["plugin"],
        "parameters": protocol["parameters"],
        "status": "completed",
        "exit_code": 0,
        "output": str(domain.parent),
        "entry_sha256": file_hash(impl),
        "report_sha256": file_hash(domain),
        "evidence": canonical_claims(protocol["plugin"], raw),
    }
    save(run / "controller/jobs.json", {"jobs": {"job": job}})
    runtime = phase(run, "launch", profile, job)
    save(run / "protocol.json", protocol)
    save(
        run / "report.json",
        {
            "status": "launch_or_job_failed",
            "protocol_sha256": digest(protocol),
            "launch": {"run": runtime},
            "jobs": [job],
        },
    )
    return run


def test_failed_final_reply_preserves_submitted_job_and_charges_all_requests(source):
    before = file_hash(source / "report.json")
    proof = load_source(source)
    assert proof["spent"] == {"model_calls": 4, "tool_calls": 1, "wall_seconds": 10}
    assert proof["original_status"] == "launch_or_job_failed"
    assert proof["submission_witnesses"] == ["call_a"]
    assert proof["facts"]["computational_agreement_passed"] is False
    assert file_hash(source / "report.json") == before


@pytest.mark.parametrize(
    "mutation", ["domain", "submission", "running", "missing_request", "budget_rejection", "entry"]
)
def test_invalid_source_cannot_be_recovered(source, mutation):
    if mutation == "domain":
        (source / "controller/job/report.json").write_text("{}")
    elif mutation == "submission":
        path = source / "launch/events.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]["args"]["parameters_json"] = '{"operation":"verify_existing"}'
        jsonl(path, rows)
    elif mutation == "running":
        path = source / "report.json"
        report = read(path)
        report["status"] = "running"
        save(path, report)
    elif mutation == "entry":
        (source / "implementation/scripts/rsi/run_mcba_case.py").write_text("changed")
    else:
        path = source / "launch/model_requests.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if mutation == "missing_request":
            rows.pop()
        else:
            rows.append({"event": "budget_rejected"})
        jsonl(path, rows)
    with pytest.raises(ValueError):
        load_source(source)


def test_concurrent_recoveries_cannot_spend_same_grant(source, tmp_path):
    proof = load_source(source)

    def reserve(i):
        try:
            return RecoveryLedger(tmp_path / "ledger", proof).reserve(tmp_path / str(i))
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, [1, 2]))
    admitted = [r for r in results if r]
    assert len(admitted) == 1
    assert admitted[0]["grant"] == {"model_calls": 20, "tool_calls": 11, "wall_seconds": 890}


def test_retry_charges_failed_recovery_and_detects_changed_prior_record(source, tmp_path):
    proof = load_source(source)
    ledger = RecoveryLedger(tmp_path / "ledger", proof)
    output = tmp_path / "recovery"
    reservation = ledger.reserve(output)
    profile = deepcopy(proof["protocol"]["profile"])
    profile["budget"].update(reservation["grant"])
    runtime = phase(output, "review", profile, proof["job"], calls=3)
    save(
        output / "report.json",
        {
            "status": "review_failed",
            "review": {"run": runtime},
            "spent": {"model_calls": 3, "tool_calls": 1, "wall_seconds": 10},
        },
    )
    ledger.settle(reservation["index"])
    new = ledger.reserve(tmp_path / "retry")
    assert new["grant"] == {"model_calls": 17, "tool_calls": 10, "wall_seconds": 880}
    assert new["spent_before"] == {"model_calls": 7, "tool_calls": 2, "wall_seconds": 20}
    (output / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Prior recovery changed"):
        ledger.reserve(tmp_path / "again")
