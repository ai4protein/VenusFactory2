"""Multi-paper supervision tests with synthetic jobs; no scientific claims."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.paper_campaign import PAPERS, SCOPES, SUITES, campaign_cases, paper_case, validate_campaign
from agent.rsi.reproduction_goal_contract import CRITERIA, allocation
from agent.science_goal import GoalSpec, GoalStore, run_goal
from backend.web_v2.science_goals import PaperCampaignDriver, driver_limits, public_paper_progress, public_reproduction_recipes


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    base = importlib.import_module("run_reproduction_goal")
    campaign = importlib.import_module("run_paper_campaign")
    monkeypatch.setattr(base, "verify_job_report", lambda *a: {"synthetic_fixture": True})
    monkeypatch.setattr(base, "canonical_claims", lambda *a: {"synthetic_fixture": True})
    return base, campaign


def setup(modules, tmp_path, cycles=8):
    base, campaign = modules
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=2, model_calls=12, tool_calls=12),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec(
            "Reproduce two declared paper scopes and verify reuse",
            CRITERIA,
            max_cycles=cycles,
            model_calls=96,
            tool_calls=96,
        ),
        profile,
        admission={"kind": "paper_campaign.v1", "papers": campaign_cases("cpu_papers")},
    )
    base.write_json(
        store.root / "protocol.json",
        {
            "kind": "paper_campaign.v1",
            "papers": campaign_cases("cpu_papers"),
            "implementation": {},
            "cycle_wall_seconds": 4,
        },
    )
    return store, campaign.PaperCampaign(store, "unused-fixture")


def install_jobs(modules, monkeypatch, fail_case=None):
    from test_reproduction_goal import run_fixture

    base, _ = modules
    seen = []

    async def reproduce(args, **kwargs):
        declaration = {"plugin": args.plugin, "parameters": json.loads(args.parameters)}
        seen.append((args.plugin, kwargs))
        report = run_fixture(base, args.output, kwargs["registered_recipe"], declaration=declaration)
        if args.plugin == fail_case:
            report.update(status="launch_or_job_failed", end_to_end_verified=False)
            base.write_json(args.output / "report.json", report)
        return report

    monkeypatch.setattr(base, "reproduce", reproduce)
    return seen


def test_catalog_is_admitted_explicit_and_defensive():
    assert len(PAPERS) == 9
    assert len(campaign_cases("seven_papers")) == 7
    for case in SCOPES:
        paper = paper_case(case)
        assert paper["allocation"] == allocation(case)
        assert paper["covered_steps"] and paper["excluded_steps"]
        assert paper["full_paper_reproduced"] is False
        paper["covered_steps"].clear()
        assert paper_case(case)["covered_steps"]
    assert paper_case("alde_partial_three_acquisition")["allocation"]["parameters"]["seeds"] == 5
    assert allocation("alde_full_three_acquisition")["parameters"]["seeds"] == 70
    for cases in [[], ["unknown"], ["phl7_all_opes"] * 2]:
        with pytest.raises(ValueError):
            campaign_cases(cases=cases)


@pytest.mark.asyncio
async def test_two_papers_register_then_recompute_in_fresh_sessions(modules, tmp_path, monkeypatch):
    store, campaign = setup(modules, tmp_path)
    seen = install_jobs(modules, monkeypatch)
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "completed"
    assert [c["case"] for c in state["cycles"]] == list(SUITES["cpu_papers"]) * 2
    assert len(seen) == 4 and state["usage"]["model_calls"] == 8
    assert all(kwargs["registered_recipe"] is None for _, kwargs in seen[:2])
    assert all(kwargs["registered_recipe"]["paper"]["full_paper_reproduced"] is False for _, kwargs in seen[2:])
    assert (
        len(
            {
                sid
                for unit in campaign.units.values()
                for sid in unit.registered(campaign.view(state, unit.paper["case"]))["proof"]["sessions"]
            }
        )
        == 4
    )
    assert len(public_reproduction_recipes(store.root, state)) == 2
    assert all(p["verified"] and not p["full_paper_reproduced"] for p in public_paper_progress(store.root, state))
    # Registration is bound to the exact paper scope, not just the plugin ID.
    broken = json.loads(json.dumps(state))
    broken["progress"]["papers"]["gfp_epinnet"] = broken["progress"]["papers"]["phl7_all_opes"]
    with pytest.raises(ValueError):
        campaign.evaluate(broken)


@pytest.mark.asyncio
async def test_failed_paper_does_not_starve_other_paper_or_complete_campaign(modules, tmp_path, monkeypatch):
    store, campaign = setup(modules, tmp_path, cycles=4)
    install_jobs(modules, monkeypatch, fail_case="phl7.opes_analysis.v1")
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "budget_exhausted"
    assert [c["case"] for c in state["cycles"]] == list(SUITES["cpu_papers"]) * 2
    assert not state["progress"]["papers"]["phl7_all_opes"]
    assert state["progress"]["papers"]["gfp_epinnet"]["reuse_receipt"]
    assert not any(v["passed"] for v in state["assessment"]["criteria"].values())
    papers = public_paper_progress(store.root, state)
    assert [p["verified"] for p in papers] == [False, True]
    assert (store.root / "episodes/cycle_000/outcome.json").exists()


@pytest.mark.asyncio
async def test_exception_keeps_reserved_usage_and_advances_next_paper(modules, tmp_path, monkeypatch):
    store, campaign = setup(modules, tmp_path, cycles=4)
    install_jobs(modules, monkeypatch)

    async def fail(*args, **kwargs):
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(campaign.units["phl7_all_opes"], "cycle", fail)
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    assert [c["case"] for c in state["cycles"]] == list(SUITES["cpu_papers"]) * 2
    assert state["usage"]["model_calls"] == 12 * 2 + 4
    assert state["progress"]["papers"]["gfp_epinnet"]["reuse_receipt"]
    assert state["cycles"][0]["error_type"] == "RuntimeError"


def test_changed_scope_and_unadmitted_unit_are_rejected(modules, tmp_path):
    store, campaign = setup(modules, tmp_path)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["papers"][0]["excluded_steps"] = []
    with pytest.raises(ValueError, match="scope or allocation"):
        validate_campaign(protocol)
    with pytest.raises(ValueError, match="not admitted"):
        modules[0].ReproductionGoal(store, "unused", case="mcba_all_tasks")


def test_interrupted_episode_keeps_scheduler_identity(modules, tmp_path):
    store, campaign = setup(modules, tmp_path)
    path = store.root / "episodes/cycle_000"
    path.mkdir(parents=True)
    (path / "paper.json").write_text(json.dumps(paper_case("phl7_all_opes")))
    state = store.read()
    state["cycles"] = [{"index": 0, "status": "interrupted"}]
    assert campaign.select(state) == "gfp_epinnet"


@pytest.mark.parametrize("suite", list(SUITES))
def test_web_driver_admits_complete_budget_and_preserves_scope(tmp_path, suite):
    driver = PaperCampaignDriver()
    detail = driver.case_details(None)[suite]
    service = SimpleNamespace(root=tmp_path, python="python", prepared_profile=lambda p: p / "profile.json")
    spec = {
        "parameters": {"case": suite},
        "objective": "Reproduce declared paper scopes",
        "budget": detail["suggested_budget"],
    }
    command = driver.prepare_command(service, tmp_path, spec)
    assert command[1].endswith("run_paper_campaign.py")
    assert command[command.index("--case") + 1] == suite
    assert all(spec["budget"][k] <= driver_limits(driver)[k] for k in spec["budget"])
    assert detail["minimum_cycles"] == len(SUITES[suite]) * 2
    spec["budget"]["max_cycles"] = detail["minimum_cycles"] - 1
    with pytest.raises(ValueError, match="two episodes"):
        driver.prepare_command(service, tmp_path, spec)


@pytest.mark.asyncio
async def test_verified_source_mutation_invalidates_campaign(modules, tmp_path, monkeypatch):
    store, campaign = setup(modules, tmp_path)
    install_jobs(modules, monkeypatch)
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    (store.root / "episodes/cycle_000/reproduction/workspace/case_submission.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt"):
        campaign.evaluate(state)


@pytest.mark.parametrize("operation", ["remove", "add", "reorder"])
def test_campaign_cannot_change_requested_paper_set_on_resume(modules, tmp_path, operation):
    store, _ = setup(modules, tmp_path)
    path = store.root / "protocol.json"
    protocol = json.loads(path.read_text())
    if operation == "remove":
        protocol["papers"].pop()
    elif operation == "add":
        protocol["papers"].append(paper_case("mcba_all_tasks"))
    else:
        protocol["papers"].reverse()
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="original admitted"):
        modules[1].PaperCampaign(store, "unused")
    assert public_paper_progress(store.root, store.read()) == []


def test_goal_admission_is_copied_and_integrity_checked(modules, tmp_path):
    store, _ = setup(modules, tmp_path)
    state = store.read()
    state["definition"]["admission"]["papers"].pop()
    (store.root / "goal.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="definition changed"):
        store.read()


def test_web_library_stays_within_owning_session(tmp_path):
    driver = PaperCampaignDriver()
    service = SimpleNamespace(root=tmp_path, python="python", prepared_profile=lambda p: p / "profile.json")
    spec = {
        "parameters": {"case": "cpu_papers"},
        "objective": "Reproduce declared paper scopes",
        "budget": driver.case_details(None)["cpu_papers"]["suggested_budget"],
    }
    libraries = []
    for session, goal in [("owner-a", "first"), ("owner-a", "second"), ("owner-b", "first")]:
        folder = tmp_path / session / goal
        command = driver.prepare_command(service, folder, spec)
        library = command[command.index("--capability-library") + 1]
        assert library == str(folder.parent / ".paper_capabilities")
        libraries.append(library)
    assert libraries[0] == libraries[1]
    assert libraries[0] != libraries[2]


@pytest.mark.asyncio
async def test_provider_access_failure_waits_then_preserves_reproduction_progress(modules, tmp_path, monkeypatch):
    store, campaign = setup(modules, tmp_path)
    base, _ = modules

    async def unavailable(args, **kwargs):
        execution = args.output / "launch"
        execution.mkdir(parents=True)
        run = {"status": "runtime_error", "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": 0}}}
        base.write_json(execution / "run.json", run)
        records = [
            {"event": "request", "call": 0},
            {"event": "response", "call": 0, "status": 403, "completed": False},
        ]
        (execution / "model_requests.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
        return {"status": "launch_or_job_failed", "end_to_end_verified": False, "launch": {"run": run}}

    monkeypatch.setattr(base, "reproduce", unavailable)
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "waiting_for_input" and len(state["cycles"]) == 1
    assert state["usage"]["model_calls"] == 1
    assert state["cycles"][0]["dependency_failure"]["http_status"] == 403
    history = (store.root / "cycle_000.json").read_bytes()
    seen = install_jobs(modules, monkeypatch)
    store.resume()
    state = await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)
    assert state["status"] == "completed" and len(seen) == 4
    assert len(state["cycles"]) == 5 and state["usage"]["model_calls"] == 9
    assert (store.root / "cycle_000.json").read_bytes() == history
