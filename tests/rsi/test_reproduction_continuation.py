"""Synthetic integration: real lifecycle/ledger/receipt checks, no LLM or GPU."""

import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_mcba_plugin import complete_report

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.rsi.engine.case_evidence import canonical_claims
from agent.rsi.engine.reproduction_goal_contract import CRITERIA, allocation
from agent.rsi.engine.reproduction_recovery import RecoveryLedger, load_source
from agent.rsi.engine.scientific_episode import episode_contract
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalSpec, GoalStore, run_goal


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def runtime(path, profile, tools, capability, *, outputs=None, args=None, failed=False):
    path.mkdir(parents=True, exist_ok=True)
    events = []
    for i, tool in enumerate(tools):
        events += [
            {
                "kind": "tool_call_start",
                "tool_call_id": str(i),
                "tool_name": "mcp__venusfactory__" + tool,
                "args": (args or {}).get(tool, {}),
            },
            {
                "kind": "tool_result",
                "tool_call_id": str(i),
                "is_error": False,
                "output": json.dumps((outputs or {}).get(tool, {"success": True})),
            },
        ]
    jsonl(path / "events.jsonl", events)
    jsonl(
        path / "model_requests.jsonl",
        [
            {"event": "request", "call": 1},
            {"event": "response", "call": 1, "status": 403 if failed else 200, "completed": not failed},
        ],
    )
    lock = {
        "profile": profile.public(),
        "task": {
            "tools": tools,
            "inputs": {},
            "capabilities": {
                str(p.relative_to(capability)): file_hash(p) for p in capability.rglob("*") if p.is_file()
            },
        },
    }
    lock["sha256"] = digest(lock)
    save(path / "profile.lock.json", lock)
    value = {
        "profile_sha256": lock["sha256"],
        "binding_intact": True,
        "host_wall_seconds": 0.1,
        "status": "runtime_error" if failed else "completed",
        "session_id": str(path),
        "audit": {
            "passed": not failed,
            "events": audit_trace(path / "events.jsonl", tools, profile.budget.tool_calls),
            "requests": audit_requests(path / "model_requests.jsonl"),
        },
    }
    save(path / "run.json", value)
    return value


def source_run(output, profile, recipe=None):
    declaration = allocation("mcba_all_tasks")
    entry = "scripts/rsi/cases/run_mcba_case.py"
    save(output / "implementation" / entry, {"synthetic": True})
    protocol = {
        "profile": profile.public(),
        "plugin": declaration["plugin"],
        "parameters": declaration["parameters"],
        "maximum_jobs": 1,
        "implementation": {entry: file_hash(output / "implementation" / entry)},
        "registered_recipe_sha256": digest(recipe) if recipe else None,
    }
    jid = "case_" + digest(str(output))[:20]
    domain = output / "controller" / jid / "report.json"
    raw = complete_report()
    save(domain, raw)
    job = {
        "job_id": jid,
        "plugin_id": declaration["plugin"],
        "parameters": declaration["parameters"],
        "status": "completed",
        "exit_code": 0,
        "output": str(domain.parent),
        "entry_sha256": protocol["implementation"][entry],
        "report_sha256": file_hash(domain),
        "evidence": canonical_claims(declaration["plugin"], raw),
    }
    save(output / "controller/jobs.json", {"jobs": {jid: job}})
    save(output / "protocol.json", protocol)
    save(
        output / "workspace/case_submission.json",
        {
            "schema": 1,
            "job_id": jid,
            "plugin_id": job["plugin_id"],
            "parameters": job["parameters"],
            "submission_only": True,
        },
    )
    report = {
        "status": "review_failed",
        "protocol_sha256": digest(protocol),
        "jobs": [job],
        "agent_requested": True,
        "submission_receipt_verified": True,
        "implementation_unchanged": True,
        "end_to_end_verified": False,
        "new_computation": False,
    }
    for stage in ["launch", "review"]:
        capsource = output / (stage + "_capability/source")
        save(capsource / "generated_tools.py", {"synthetic": True})
        versions = VersionStore(output / (stage + "_capability/bundles"))
        manifest = versions.snapshot(capsource)
        capability = versions.root / "versions" / manifest["version_id"]
        tools = (
            ["discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"]
            if stage == "launch"
            else ["inspect_case_evidence"]
        )
        if recipe and stage == "launch":
            tools.append("inspect_registered_reproduction")
        run = runtime(
            output / stage,
            profile,
            tools,
            capability,
            outputs={"submit_reproduction": {"success": True, **job}},
            args={
                "submit_reproduction": {"plugin_id": job["plugin_id"], "parameters_json": json.dumps(job["parameters"])}
            },
            failed=stage == "review",
        )
        report[stage] = {"run": run, "source_report_sha256": file_hash(domain)}
    save(output / "report.json", report)
    return report


@pytest.fixture
def system(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    goal_module = importlib.import_module("run_reproduction_goal")
    continuation = importlib.import_module("reproduction_continuation")
    recovery = importlib.import_module("resume_harness_reproduction")
    repository = tmp_path / "repository"
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "glm-fixture", "https://example.org/v1"),
        Budget(model_calls=24, tool_calls=24, wall_seconds=60),
    )
    store = GoalStore.create(
        repository / "research_runs/rsi/goal",
        GoalSpec("Reproduce then reuse", CRITERIA, max_cycles=6, model_calls=96, tool_calls=96, wall_seconds=600),
        profile,
        admission={"review_continuation": continuation.POLICY},
    )
    save(
        store.root / "protocol.json",
        {
            "case": "mcba_all_tasks",
            "allocation": allocation("mcba_all_tasks"),
            "implementation": {},
            "review_continuation": continuation.POLICY,
            "cycle_wall_seconds": 60,
        },
    )
    native = []
    monkeypatch.setattr(continuation, "verify_job_report", lambda *a: native.append(a[1]["job_id"]))
    monkeypatch.setattr(recovery, "ROOT", repository)
    monkeypatch.setattr(recovery, "archive_implementation", lambda *a: {})

    # bundle copies the real reviewed tool code; only the model and native
    # scientific computation are synthetic. Its immutable contract is checked.
    async def execute(profile, task, runtimes, credential):
        contract = episode_contract(task.capability, "chat", task.tools)
        assert contract.required_tools == ("inspect_case_evidence", "submit_evidence_review")
        assert set(task.tools) == {"inspect_case_evidence", "submit_evidence_review"}
        facts = json.loads((task.capability / "evidence.json").read_text())["facts"]
        save(task.workspace / "review.json", {"claims": facts})
        return runtime(task.output, profile, sorted(task.tools), task.capability)

    monkeypatch.setattr(recovery, "execute", execute)
    submissions = []

    async def reproduce(args, **kwargs):
        submissions.append(args.output)
        return source_run(args.output, kwargs["profile"], kwargs["registered_recipe"])

    monkeypatch.setattr(goal_module, "reproduce", reproduce)
    return SimpleNamespace(
        module=goal_module,
        continuation=continuation,
        recovery=recovery,
        store=store,
        profile=profile,
        submissions=submissions,
        native=native,
    )


@pytest.mark.asyncio
async def test_persistent_goal_resumes_review_without_resubmitting_compute(system):
    s = system
    driver = s.module.ReproductionGoal(s.store, "unused-test-credential")
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input" and len(s.submissions) == 1
    first = s.submissions[0]
    failed_sha = file_hash(first / "report.json")
    assert state["progress"]["pending_review"] and not state["progress"].get("recipe_version")
    s.store.resume()
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input" and len(s.submissions) == 2
    assert state["progress"]["recipe_version"] and state["progress"]["pending_review"]
    s.store.resume()
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "completed" and len(s.submissions) == 2 and len(state["cycles"]) == 4
    assert state["usage"]["model_calls"] == 6
    assert state["usage"]["tool_calls"] == 13
    assert "pending_review" not in state["progress"]
    assert file_hash(first / "report.json") == failed_sha
    assert [c["status"] for c in state["cycles"]] == ["failed", "completed", "failed", "completed"]
    assert s.native and all(v["passed"] for v in state["assessment"]["criteria"].values())
    recipe = driver.registered(state)
    assert recipe["proof"]["recovery_trigger"] == "persistent_goal"
    assert recipe["proof"]["new_computation_in_recovery"] is False
    assert recipe["proof"]["uninterrupted_end_to_end_verified"] is False
    recovered = s.store.root / recipe["source_run"]
    assert not (recovered / "controller/jobs.json").exists()
    receipt = json.loads((s.store.root / state["progress"]["reuse_receipt"]).read_text())
    assert not set(recipe["proof"]["sessions"]) & set(receipt["proof"]["sessions"])
    (first / "workspace/case_submission.json").write_text("{}")
    with pytest.raises(ValueError):
        driver.evaluate(state)


@pytest.mark.asyncio
async def test_exhausted_original_allowance_retains_compute_and_requests_input(system):
    s = system
    profile = Profile.from_dict(s.profile.public())
    profile = Profile.from_dict({**profile.public(), "budget": {**profile.public()["budget"], "model_calls": 2}})
    folder = s.store.root / "episodes/cycle_000"
    output = folder / "reproduction"
    source_run(output, profile)
    record = s.continuation.capture_pending(s.store.root, folder, output, allocation("mcba_all_tasks"), None)
    result = await s.continuation.continue_review(
        s.store.root,
        s.store.root / "episodes/cycle_001/reproduction",
        record,
        allocation("mcba_all_tasks"),
        None,
        s.profile,
        "unused",
        "goal",
    )
    assert result["status"] == "review_budget_exhausted" and result["needs_input"]
    assert not s.submissions


@pytest.mark.parametrize("change", ["report", "receipt", "scope", "recipe", "runtime", "domain", "pending", "symlink"])
def test_changed_or_incompatible_pending_evidence_is_rejected(system, change):
    s = system
    folder = s.store.root / "episodes/cycle_000"
    output = folder / "reproduction"
    source_run(output, s.profile)
    declaration = allocation("mcba_all_tasks")
    record = s.continuation.capture_pending(s.store.root, folder, output, declaration, None)
    recipe = None
    if change == "report":
        save(output / "report.json", {})
    elif change == "receipt":
        save(output / "workspace/case_submission.json", {})
    elif change == "scope":
        declaration["parameters"] = {"operation": "verify_existing"}
    elif change == "recipe":
        recipe = {"not": "original recipe"}
    elif change == "runtime":
        save(output / "launch/run.json", {})
    elif change == "domain":
        save(next((output / "controller").glob("case_*/report.json")), {})
    elif change == "pending":
        record["sha256"] = "0" * 64
    else:
        original = folder / "pending_review.json"
        alias = folder / "alias.json"
        alias.symlink_to(original)
        record["path"] = str(alias.relative_to(s.store.root))
    with pytest.raises((ValueError, KeyError)):
        s.continuation.pending_evidence(s.store.root, record, declaration, recipe)


def test_legacy_protocol_does_not_silently_enable_continuation(system):
    s = system
    state = s.store.read()
    legacy = deepcopy(state)
    legacy["definition"].pop("admission", None)
    assert not s.continuation.policy_enabled({}, legacy)
    with pytest.raises(ValueError, match="not admitted"):
        s.continuation.policy_enabled({"review_continuation": s.continuation.POLICY}, legacy)


def test_cycle_ceiling_cannot_reset_original_budget(system):
    output = system.store.root / "episodes/cycle_000/reproduction"
    source_run(output, system.profile)
    source = load_source(output)
    ledger = RecoveryLedger(system.store.root / "test_ledger", source)
    grant = ledger.reserve(
        system.store.root / "recovery", ceiling={"model_calls": 3, "tool_calls": 2, "wall_seconds": 5}
    )
    assert grant["grant"] == {"model_calls": 3, "tool_calls": 2, "wall_seconds": 5}
    # An unresolved attempt keeps its entire grant charged.
    next_grant = ledger.reserve(
        system.store.root / "another", ceiling={"model_calls": 100, "tool_calls": 100, "wall_seconds": 100}
    )
    assert next_grant["grant"]["model_calls"] == 19


@pytest.mark.asyncio
async def test_operator_recovery_cannot_be_registered_as_automatic(system):
    s = system
    original = s.store.root / "episodes/cycle_000/reproduction"
    source_run(original, s.profile)
    recovered = s.store.root / "episodes/cycle_001/reproduction"
    result = await s.recovery.main(SimpleNamespace(source_run=original, output=recovered), credential="unused")
    assert result["recovered_review_verified"] and result["operator_recovery_interventions"] == 1
    with pytest.raises(ValueError, match="operator recovery"):
        s.module.verify_run(recovered, allocation("mcba_all_tasks"))


@pytest.mark.asyncio
async def test_failed_continuation_is_charged_and_retried_without_new_job(system, monkeypatch):
    s = system
    driver = s.module.ReproductionGoal(s.store, "unused")
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input"
    successful_execute = s.recovery.execute

    async def failed_execute(profile, task, runtimes, credential):
        return runtime(task.output, profile, sorted(task.tools), task.capability, failed=True)

    monkeypatch.setattr(s.recovery, "execute", failed_execute)
    s.store.resume()
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input" and len(s.submissions) == 1
    assert state["usage"]["model_calls"] == 3 and state["progress"]["pending_review"]
    failed_path = s.store.root / "episodes/cycle_001/reproduction/report.json"
    failed_hash = file_hash(failed_path)
    monkeypatch.setattr(s.recovery, "execute", successful_execute)
    s.store.resume()
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input" and len(s.submissions) == 2
    recipe = driver.registered(state)
    assert len(recipe["proof"]["recovery_attempts"]) == 2
    assert file_hash(failed_path) == failed_hash
    recovered = s.store.root / recipe["source_run"]
    report = json.loads((recovered / "report.json").read_text())
    assert report["cumulative_spent"]["model_calls"] == 4
    assert report["cumulative_spent"]["tool_calls"] == 8


@pytest.mark.asyncio
async def test_recovered_paper_publishes_and_imported_goal_still_requires_fresh_compute(system):
    s = system
    campaign_module = importlib.import_module("run_paper_campaign")
    library_module = importlib.import_module("paper_library")
    from agent.rsi.campaigns.paper_campaign import campaign_cases
    from agent.rsi.engine.paper_capability_library import PaperCapabilityLibrary

    papers = campaign_cases(cases=["mcba_all_tasks"])
    admission = {"kind": "paper_campaign.v1", "papers": papers, "review_continuation": s.continuation.POLICY}

    def create(name, declared):
        store = GoalStore.create(
            s.store.root.parent / name,
            GoalSpec("Paper accumulation", CRITERIA, max_cycles=6, model_calls=96, tool_calls=96, wall_seconds=600),
            s.profile,
            admission=declared,
        )
        save(store.root / "protocol.json", {**declared, "implementation": {}, "cycle_wall_seconds": 60})
        return store

    async def finish(store):
        driver = campaign_module.PaperCampaign(store, "unused")
        state = await run_goal(store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
        while state["status"] == "waiting_for_input":
            store.resume()
            state = await run_goal(store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
        assert state["status"] == "completed"
        return state

    donor = create("donor", admission)
    donor_state = await finish(donor)
    assert len(donor_state["cycles"]) == 4 and len(s.submissions) == 2
    library = PaperCapabilityLibrary(s.store.root.parent / "library")
    identity = library.publish(donor.root, "mcba_all_tasks", library_module.validate_completed_recipe)
    item = library.resolve(identity, papers[0], library_module.validate_completed_recipe)
    imported = create(
        "recipient",
        {
            **admission,
            "capability_library": {
                "path": str(library.root),
                "imports": {"mcba_all_tasks": identity},
            },
        },
    )
    library_module.import_recipes(imported, library, {"mcba_all_tasks": item})
    initial = campaign_module.PaperCampaign(imported, "unused").evaluate(imported.read())
    assert initial["criteria"]["registered_recipe"]["passed"]
    assert not initial["criteria"]["fresh_session_reuse"]["passed"]
    state = await finish(imported)
    assert len(state["cycles"]) == 2 and len(s.submissions) == 3
    assert library.publish(imported.root, "mcba_all_tasks", library_module.validate_completed_recipe) == identity
    assert len(list((library.root / "entries").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_missing_review_trace_stops_without_losing_or_resubmitting_completed_job(system, monkeypatch):
    s = system

    async def reproduce(args, **kwargs):
        s.submissions.append(args.output)
        report = source_run(args.output, kwargs["profile"], kwargs["registered_recipe"])
        path = args.output / "review"
        (path / "events.jsonl").unlink()
        audit = audit_trace(path / "events.jsonl", ["inspect_case_evidence"], kwargs["profile"].budget.tool_calls)
        report["review"]["run"]["audit"]["events"] = audit
        save(path / "run.json", report["review"]["run"])
        save(args.output / "report.json", report)
        return report

    monkeypatch.setattr(s.module, "reproduce", reproduce)
    driver = s.module.ReproductionGoal(s.store, "unused")
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input"
    assert len(s.submissions) == 1 and state["progress"]["review_reconciliation"]
    assert state["usage"]["model_calls"] == s.profile.budget.model_calls
    assert state["usage"]["tool_calls"] == s.profile.budget.tool_calls
    s.store.resume()
    state = await run_goal(s.store, driver.cycle, driver.evaluate, cycle_wall_seconds=60)
    assert state["status"] == "waiting_for_input" and len(s.submissions) == 1
    assert state["cycles"][-1]["reason"] == "completed_job_review_not_recoverable"
    assert state["cycles"][-1]["usage"]["model_calls"] == 0
    assert not state["assessment"]["criteria"]["verified_reproduction"]["passed"]
