"""Synthetic scientific artifacts exercise cross-goal ownership and replay gates."""

import json
from types import SimpleNamespace

import pytest
from test_paper_campaign import install_jobs
from test_paper_campaign import modules as campaign_modules

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi.campaigns.paper_campaign import admitted_campaign, paper_case
from agent.rsi.engine.paper_capability_library import PaperCapabilityLibrary
from agent.science_goal import run_goal
from backend.web_v2.science_goals import public_reproduction_recipes


@pytest.fixture
def modules(monkeypatch):
    return campaign_modules.__wrapped__(monkeypatch)


def create(modules, monkeypatch, root, name, library):
    _, campaign = modules
    monkeypatch.setattr(campaign, "invocation", lambda *a, **kw: None)
    monkeypatch.setattr(campaign, "research_implementation_paths", lambda _: [])
    profile = root / "fixture-profile.json"
    write_json(
        profile,
        Profile(
            "science_expert",
            "langgraph",
            ModelBinding("fixture", "fixture", "https://example.org/v1"),
            Budget(wall_seconds=2, model_calls=12, tool_calls=12),
        ).public(),
    )
    store = campaign.prepare(
        SimpleNamespace(
            output=root / name,
            profile=profile,
            case="cpu_papers",
            paper_case=None,
            objective="Synthetic cross-goal verification",
            max_cycles=8,
            model_calls=96,
            tool_calls=96,
            wall_seconds=1000,
            capability_library=library,
        )
    )
    return store, campaign.PaperCampaign(store, "fixture-no-credential")


async def complete(store, campaign):
    return await run_goal(store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=4)


@pytest.mark.asyncio
async def test_two_goals_share_verified_recipes_but_require_new_computation(modules, monkeypatch, tmp_path):
    from paper_library import publish_campaign

    library = tmp_path / ".paper_capabilities"
    seen = install_jobs(modules, monkeypatch)
    first, source = create(modules, monkeypatch, tmp_path, "source", library)
    assert (await complete(first, source))["status"] == "completed"
    published = publish_campaign(first)
    before = {name: file_hash(first.root / name) for name in ["goal.json", "protocol.json"]}
    second, recipient = create(modules, monkeypatch, tmp_path, "recipient", library)
    state = second.read()
    assert len(state["progress"]["papers"]) == 2
    checks = recipient.evaluate(state)["criteria"]
    assert checks["verified_reproduction"]["passed"] and checks["registered_recipe"]["passed"]
    assert not checks["fresh_session_reuse"]["passed"]
    result = await complete(second, recipient)
    assert result["status"] == "completed" and len(result["cycles"]) == 2
    assert all(c["phase"] == "reuse" for c in result["cycles"])
    assert len(seen) == 6 and all(kwargs["registered_recipe"].get("library_origin") for _, kwargs in seen[-2:])
    assert publish_campaign(second)["entries"] == published["entries"]
    assert len(list((library / "entries").glob("*.json"))) == 2
    assert all(row["imported"] and row["reused"] for row in public_reproduction_recipes(second.root, result))
    assert before == {name: file_hash(first.root / name) for name in before}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["entry", "origin", "domain", "scope", "admission"])
async def test_changed_capability_evidence_cannot_be_imported(modules, monkeypatch, tmp_path, change):
    from paper_library import publish_campaign, validate_completed_recipe

    path = tmp_path / ".paper_capabilities"
    install_jobs(modules, monkeypatch)
    store, campaign = create(modules, monkeypatch, tmp_path, "source", path)
    await complete(store, campaign)
    publication = publish_campaign(store)
    identity = publication["entries"]["gfp_epinnet"]
    library = PaperCapabilityLibrary(path)
    if change == "entry":
        p = path / "entries" / (identity + ".json")
        p.chmod(0o644)
        value = json.loads(p.read_text())
        value["paper"]["case"] = "different"
        write_json(p, value)
    elif change == "origin":
        store.mutate(lambda s: s.update(status="paused"))
    elif change == "domain":
        paper = campaign.units["gfp_epinnet"]
        recipe = paper.registered(campaign.view(store.read(), "gfp_epinnet"))
        p = store.root / recipe["source_run"] / "controller" / recipe["proof"]["job_id"] / "report.json"
        write_json(p, {"status": "changed"})
    elif change == "scope":
        with pytest.raises(ValueError, match="incompatible"):
            library.resolve(identity, paper_case("phl7_all_opes"), validate_completed_recipe)
        return
    else:
        target, _ = create(modules, monkeypatch, tmp_path, "recipient", path)
        protocol = json.loads((target.root / "protocol.json").read_text())
        protocol["capability_library"]["imports"].clear()
        with pytest.raises(ValueError, match="original admitted"):
            admitted_campaign(protocol, target.read())
        return
    with pytest.raises(ValueError):
        library.find(paper_case("gfp_epinnet"), validate_completed_recipe)


def test_library_enforces_owner_paths_and_scope_match(tmp_path):
    library = PaperCapabilityLibrary(tmp_path / "owner" / ".paper_capabilities")
    with pytest.raises(ValueError, match="owner namespace"):
        library.goal(tmp_path / "another-owner" / "goal")
    with pytest.raises(ValueError):
        library.goal(library.owner)
    link = tmp_path / "alias"
    link.symlink_to(tmp_path / "owner", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        PaperCapabilityLibrary(link / ".paper_capabilities")
    assert library.find(paper_case("alde_full_three_acquisition"), lambda *_: None) is None


def test_incomplete_goal_cannot_publish(modules, monkeypatch, tmp_path):
    from paper_library import publish_campaign

    store, _ = create(modules, monkeypatch, tmp_path, "source", tmp_path / ".paper_capabilities")
    with pytest.raises(ValueError, match="completed"):
        publish_campaign(store)


@pytest.mark.asyncio
async def test_preparation_defers_numeric_replay_but_worker_cannot_skip_it(modules, monkeypatch, tmp_path):
    from paper_library import publish_campaign

    path = tmp_path / ".paper_capabilities"
    install_jobs(modules, monkeypatch)
    first, source = create(modules, monkeypatch, tmp_path, "source", path)
    await complete(first, source)
    publish_campaign(first)

    def unavailable_native_replay(*args):
        raise ValueError("Native artifacts must be replayed in the worker")

    monkeypatch.setattr(modules[0], "verify_job_report", unavailable_native_replay)
    second, recipient = create(modules, monkeypatch, tmp_path, "recipient", path)
    assert second.read()["status"] == "active"
    assert not second.read()["cycles"]
    with pytest.raises(ValueError, match="Native artifacts must be replayed"):
        recipient.evaluate(second.read())
    assert second.read()["status"] == "active"
