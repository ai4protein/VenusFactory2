"""Synthetic campaign wiring; no provider calls or scientific publications."""

import copy
import json
import sys
from pathlib import Path

import pytest
import test_scientist_model_campaign as campaign_fixtures

from agent.rsi import scientist_model_campaign as campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_scientist_model_campaign as cli

checkpoint = campaign_fixtures.checkpoint
setup = campaign_fixtures.setup
task = campaign_fixtures.task


@pytest.mark.parametrize(
    "policy",
    [
        {"condition": "source", "selection": "prior_campaign", "max_entries": 6},
        {"condition": "source", "selection": "prior_campaign", "max_entries": True},
        {"condition": "unknown", "selection": "prior_campaign", "max_entries": 1},
        {"condition": "source", "selection": "explicit", "entry_ids": ["not-a-model"]},
        {"condition": "none", "selection": "explicit", "entry_ids": [], "library_root": "/other-owner"},
    ],
)
def test_invalid_transfer_policies_fail_during_admission(setup, policy):
    _, base, _, _, _ = setup
    with pytest.raises(ValueError):
        campaign.admit_allocations([{**copy.deepcopy(base), "experience": policy}])


def test_prior_publications_exclude_later_and_unverified_models(setup):
    create, _, _, _, _ = setup
    store = create(count=4, target=1000)

    def populate(state):
        for index in range(4):
            state["allocations"][f"allocation_{index}"]["entries"] = [f"{index + 1:064x}"]
        state["verified_entry_ids"] = [f"{index + 1:064x}" for index in (0, 1, 3)]

    store.mutate(populate)
    assert campaign.prior_publications(store, "allocation_0") == []
    assert campaign.prior_publications(store, "allocation_3") == [f"{2:064x}", f"{1:064x}"]


def test_executor_cannot_ignore_admitted_transfer_condition(setup):
    create, base, _, _, _ = setup
    base["experience"] = {"condition": "source", "selection": "explicit", "entry_ids": []}
    store = create(count=1)
    allocation = store.specifications()["allocation_0"]
    root = store.goal(allocation["id"])
    checkpoint(allocation, root)
    with pytest.raises(ValueError, match="Missing campaign library experience"):
        campaign.bind_goal(allocation, root)


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["none", "summary", "source"])
async def test_executor_passes_frozen_cold_start_to_actual_prepare_boundary(setup, monkeypatch, condition):
    create, base, _, _, _ = setup
    base["experience"] = {"condition": condition, "selection": "prior_campaign", "max_entries": 3}
    store = create(count=1)
    allocation = store.specifications()["allocation_0"]
    captured = {}

    class Registry:
        def preflight(self, profile, runtimes):
            return {"ready": True}

    class StopBeforeExecution(Exception):
        pass

    def prepare(args):
        captured.update(json.loads(args.experience_selection.read_text()))
        raise StopBeforeExecution

    monkeypatch.setattr(cli, "builtin_registry", Registry)
    monkeypatch.setattr(cli, "prepare", prepare)
    with pytest.raises(StopBeforeExecution):
        await cli.execute_allocation(allocation, store.goal(allocation["id"]))
    assert captured == {
        "condition": condition,
        "library_root": store.read()["definition"]["library_root"],
        "entry_ids": [],
    }
    assert store.read()["usage"]["model_calls"] == 0
