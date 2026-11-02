"""All-arm execution remains necessary after coverage succeeds; no real models here."""

import pytest
import test_scientist_model_campaign as fixtures

from agent.rsi import scientist_model_campaign as campaign

task = fixtures.task
setup = fixtures.setup


@pytest.mark.asyncio
async def test_all_arms_execute_after_one_scope_or_duplicate_entry_is_enough(setup):
    create, _, settings, *_ = setup
    settings["duplicate"] = True
    store = create(target=1, execution=campaign.ALL_ALLOCATIONS)
    calls = []

    async def execute(allocation, goal):
        calls.append(allocation["id"])
        return fixtures.checkpoint(allocation, goal)

    state = await campaign.run_campaign(store, execute)
    assert calls == ["allocation_0", "allocation_1"]
    assert state["status"] == "completed" and len(state["verified_entry_ids"]) == 1
    report = store.audit_publications()
    assert report["target_met"] and report["execution_complete"] and report["completion_met"]
    assert report["terminal_attempted_allocations"] == report["required_allocations"] == 2
    assert report["unfinished_allocations"] == []


@pytest.mark.asyncio
async def test_failed_comparator_is_attempted_and_retained_after_target_success(setup):
    create, *_ = setup
    store = create(target=1, execution=campaign.ALL_ALLOCATIONS)

    async def execute(allocation, goal):
        if allocation["id"] == "allocation_1":
            raise RuntimeError("Synthetic failed comparator")
        return fixtures.checkpoint(allocation, goal)

    state = await campaign.run_campaign(store, execute)
    assert state["status"] == "completed"
    assert state["allocations"]["allocation_1"]["status"] == "failed"
    assert state["allocations"]["allocation_1"]["attempts"][0]["usage_known"] is False
    assert state["usage"]["model_calls"] == 13
    assert store.audit_publications()["execution_complete"]


@pytest.mark.asyncio
async def test_execution_completion_does_not_fabricate_coverage(setup):
    create, *_ = setup
    store = create(target=1, execution=campaign.ALL_ALLOCATIONS)

    async def fail(*args):
        raise RuntimeError("Synthetic failure")

    state = await campaign.run_campaign(store, fail)
    report = store.audit_publications()
    assert state["status"] == "needs_tasks"
    assert report["execution_complete"] and not report["target_met"] and not report["completion_met"]
    assert not state["verified_entry_ids"]


@pytest.mark.asyncio
async def test_coverage_success_cannot_override_remaining_arm_budget(setup):
    create, *_ = setup
    store = create(target=1, calls=12, execution=campaign.ALL_ALLOCATIONS)
    state = await campaign.run_campaign(store, fixtures.complete)
    report = store.audit_publications()
    assert state["status"] == "budget_exhausted"
    assert report["target_met"] and not report["completion_met"]
    assert report["unfinished_allocations"] == ["allocation_1"]
    assert not store.goal("allocation_1").exists()


@pytest.mark.asyncio
async def test_provider_block_after_target_requires_explicit_resume(setup):
    create, *_ = setup
    store = create(count=3, target=1, calls=36, execution=campaign.ALL_ALLOCATIONS)

    async def execute(allocation, goal):
        return fixtures.checkpoint(
            allocation,
            goal,
            status="waiting_for_input" if allocation["id"] == "allocation_1" else "completed",
        )

    state = await campaign.run_campaign(store, execute)
    assert state["status"] == "waiting_for_model"
    assert not store.goal("allocation_2").exists()
    assert not store.audit_publications()["execution_complete"]
    store.resume()
    state = await campaign.run_campaign(store, fixtures.complete)
    assert state["status"] == "completed"
    assert len(state["allocations"]["allocation_1"]["attempts"]) == 2
    assert len(state["allocations"]["allocation_0"]["attempts"]) == 1
    assert store.audit_publications()["execution_complete"]


def test_default_keeps_historical_acquisition_behavior(setup):
    create, *_ = setup
    state = create().read()
    assert "execution_policy" not in state["definition"]
    assert campaign.execution_policy(state["definition"]) == campaign.COVERAGE_TARGET


@pytest.mark.parametrize("value", ["all", "unknown.v1", None, {}, [], True])
def test_unknown_execution_contract_is_rejected_before_creation(setup, value):
    create, *_ = setup
    with pytest.raises(ValueError, match="execution policy"):
        create(execution=value)


def test_terminal_status_without_an_attempt_cannot_close_a_study(setup):
    create, *_ = setup
    store = create(execution=campaign.ALL_ALLOCATIONS)
    store.mutate(lambda s: [x.update(status="failed") for x in s["allocations"].values()])
    report = store.audit_publications()
    assert report["terminal_attempted_allocations"] == 0
    assert not report["execution_complete"]


def test_definition_policy_cannot_be_changed_after_admission(setup):
    create, *_ = setup
    store = create(execution=campaign.ALL_ALLOCATIONS)
    store.mutate(lambda s: s["definition"].update(execution_policy=campaign.COVERAGE_TARGET))
    with pytest.raises(ValueError, match="definition changed"):
        store.read()


@pytest.mark.asyncio
async def test_cli_cannot_silently_change_an_existing_execution_policy(setup, monkeypatch):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    create, *_ = setup
    store = create()
    before = store.read()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    runner = importlib.import_module("run_scientist_model_campaign")
    monkeypatch.setattr(
        runner, "credential_from_toml", lambda *a: pytest.fail("Resolved credentials before policy check")
    )
    args = SimpleNamespace(
        action="run",
        output=store.root,
        execution_policy=campaign.ALL_ALLOCATIONS,
        credential_toml=Path("unused"),
    )
    with pytest.raises(ValueError, match="existing campaign"):
        await runner.main(args)
    assert store.read() == before
