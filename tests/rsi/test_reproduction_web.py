"""Reproduction goal web admission and evidence presentation; no model calls."""

import json
import sys
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.engine.reproduction_goal_contract import CASES, CRITERIA
from agent.rsi.engine.versions import VersionStore
from backend.web_v2.science_goals import (
    GeneratedModelDriver,
    ReproductionDriver,
    ScienceGoalService,
    driver_limits,
    goal_activity,
    public_reproduction_recipes,
)


def profile():
    return Profile(
        "science_expert", "langgraph", ModelBinding("fixture", "glm-fixture", "https://example.org/v1"), Budget()
    )


@pytest.mark.parametrize("case", list(CASES))
def test_reproduction_driver_selects_registered_case_and_preserves_arguments(tmp_path, case):
    driver = ReproductionDriver()
    service = SimpleNamespace(
        root=tmp_path,
        python=sys.executable,
        prepared_profile=lambda folder: folder / "profile.json",
        frozen_profile=lambda folder: profile(),
        frozen_credential_path=lambda folder: None,
    )
    spec = {
        "parameters": {"case": case},
        "objective": "Reproduce and verify workflow reuse",
        "budget": {"max_cycles": 4, "model_calls": 96, "tool_calls": 48, "wall_seconds": 86400},
    }
    command = driver.prepare_command(service, tmp_path / "goal", spec)
    assert command[1].endswith("run_reproduction_goal.py")
    assert command[command.index("--case") + 1] == case
    assert command[command.index("--wall-seconds") + 1] == "86400"
    assert driver.run_command(service, tmp_path / "goal", "resume")[2].endswith("run_reproduction_goal.py")
    assert driver.criteria == CRITERIA


@pytest.mark.parametrize(
    "parameters,cycles",
    [
        ({"case": "unknown"}, 4),
        ({"case": "phl7_wt_hrex_cuda", "command": "arbitrary"}, 4),
        ({"case": "phl7_wt_hrex_cuda"}, 1),
    ],
)
def test_invalid_reproduction_request_is_rejected_before_preparation(tmp_path, parameters, cycles):
    service = SimpleNamespace(
        root=tmp_path, python=sys.executable, prepared_profile=lambda folder: folder / "profile.json"
    )
    with pytest.raises(ValueError):
        ReproductionDriver().prepare_command(
            service, tmp_path, {"parameters": parameters, "objective": "scope", "budget": {"max_cycles": cycles}}
        )


def test_only_reproduction_driver_admits_long_compute_and_catalog_explains_scope(monkeypatch):
    assert driver_limits(GeneratedModelDriver())["wall_seconds"] == 7200
    assert driver_limits(ReproductionDriver())["wall_seconds"] == 7 * 86400
    svc = ScienceGoalService()
    monkeypatch.setattr(svc, "profile", profile)
    monkeypatch.setattr(svc, "runtime_paths", lambda: {})
    monkeypatch.setattr(
        svc, "execution_catalog", lambda: SimpleNamespace(public=lambda paths: {"combinations": [{"ready": True}]})
    )
    entry = next(d for d in svc.catalog()["drivers"] if d["id"] == "reproduction_goal.v1")
    assert set(entry["cases"]) == set(CASES)
    assert "catapro_full_cohorts" in entry["cases"]
    assert "catapro_full_evidence_reuse" not in entry["cases"]
    vhh = entry["case_details"]["vhh_ha_64x4"]
    assert vhh["suggested_budget"]["wall_seconds"] == 2 * (36000 + 900)
    assert vhh["suggested_budget"]["wall_seconds"] <= entry["maximum_budget"]["wall_seconds"]
    assert "binding" in vhh["scope"]
    assert "original training" in entry["case_details"]["turbopetase_all_checkpoints"]["scope"]


def test_both_expert_phases_are_visible_and_private_request_files_are_ignored(tmp_path):
    for phase in ["launch", "review"]:
        folder = tmp_path / "episodes/cycle_000/reproduction" / phase
        folder.mkdir(parents=True)
        (folder / "graph_history.json").write_text(
            json.dumps([None, {"role_id": "scientific_critic", "content": phase + " public evidence"}])
        )
        (folder / "model_requests.jsonl").write_text("private fixture must not appear")
    rows = goal_activity(tmp_path, {"cycles": [], "inflight": {"index": 0}})
    assert {r["stage"] for r in rows} == {"launch", "review"}
    assert len({r["id"] for r in rows}) == 2
    assert all("private" not in r["content"] for r in rows)


def test_recipe_display_checks_immutable_package_and_verified_reuse(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    recipe = {
        "kind": "verified_reproduction_recipe",
        "plugin": "phl7.hrex_validation.v1",
        "scope": "four-ps validation only",
    }
    (source / "recipe.json").write_text(json.dumps(recipe))
    versions = VersionStore(tmp_path / "recipes")
    version = versions.snapshot(source)["version_id"]
    state = {
        "progress": {"recipe_version": version, "reuse_receipt": "unverified.json"},
        "assessment": {"criteria": {"fresh_session_reuse": {"passed": False}}},
    }
    displayed = public_reproduction_recipes(tmp_path, state)[0]
    assert displayed["available"] and not displayed["reused"]
    state["assessment"]["criteria"]["fresh_session_reuse"]["passed"] = True
    assert public_reproduction_recipes(tmp_path, state)[0]["reused"]
    path = versions.root / "versions" / version / "recipe.json"
    path.chmod(0o644)
    path.write_text("{}")
    assert public_reproduction_recipes(tmp_path, state) == [
        {"version_id": version, "available": False, "reused": False}
    ]
