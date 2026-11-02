"""Failure closeout preserves denominators and refuses live or partial publication."""

import hashlib
import json
from itertools import permutations

import pytest

from agent.rsi import matched_failure_closeout as module
from agent.rsi.engine.matched_history_analysis import ARMS
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.shared_sequence_evaluation import require_terminal


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def study_fixture(tmp_path):
    study = tmp_path / "study"
    repository = study / "repository"
    source = repository / "src/runtime_fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Metadata fixture, never a scientific model.\n")
    snapshot = {"src/runtime_fixture.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    write(study / "snapshot.json", {"files": snapshot, "sha256": digest(snapshot)})
    runner = {"pid": 2147483647, "start_ticks": "0", "boot_id": "no-such-boot"}
    write(study / "runner_identity.json", {"study": str(study), **runner})
    budget = {"max_cycles": 1, "model_calls": 3, "tool_calls": 2, "wall_seconds": 30}
    model = {"provider": "test_fixture", "model": "fixture"}
    policies = {arm: arm + "_fixture" for arm in ARMS}
    units, tasks, failed = [], [], None
    for block, order in enumerate(permutations(ARMS)):
        task = {
            "id": f"task_{block}",
            "plugin_id": "generated_context.v2",
            "scope": {"protein": block},
            "data": {},
            "references": [],
        }
        tasks.append(task)
        for arm in order:
            uid = f"block_{block}_{arm}"
            folder = study / uid
            library = folder / "model_library"
            library.mkdir(parents=True)
            plan = {
                "target_scopes": 1,
                "resources": {k: v for k, v in budget.items() if k != "max_cycles"},
                "batches": [
                    {
                        "id": "b",
                        "tasks": [task],
                        "goal_budget": budget,
                        "profile": {"model": model},
                        "shared_training_history": policies[arm],
                    }
                ],
            }
            psha = write(folder / "plan.json", plan)
            unit = {
                "id": uid,
                "arm": arm,
                "project": "fixture",
                "source_block": f"b{block}",
                "campaign": str(folder / "campaign"),
                "library": str(library),
                "plan": str(folder / "plan.json"),
                "plan_sha256": psha,
            }
            units.append(unit)
            write(folder / "runner_identity.json", {**runner, "campaign": unit["campaign"]})
            is_failed = block == 0 and arm == "selective"
            if is_failed:
                failed = folder
            charge = {k: v for k, v in budget.items() if k != "max_cycles"}
            attempts = (
                [
                    {
                        "status": "failed",
                        "charged": charge,
                        "usage_known": False,
                        "publication": None,
                        "error_type": "JSONDecodeError",
                    }
                ]
                if is_failed
                else []
            )
            audit = {"verified_scope_count": 0, "batches": []}
            aid = digest(audit)
            asha = write(folder / "campaign/audits" / (aid + ".json"), audit)
            campaign = {
                "status": "needs_tasks",
                "owner": None,
                "inflight": None,
                "last_audit": aid,
                "definition": {"library_root": str(library)},
                "usage": charge,
                "batches": {"b": {"status": "failed" if is_failed else "budget_exhausted", "attempts": attempts}},
            }
            csha = write(folder / "campaign/campaign.json", campaign)
            if is_failed:
                admission = {"shared_training_history": policies[arm]}
                definition = {"admission": admission}
                goal = {
                    "definition": definition,
                    "definition_sha256": digest(definition),
                    "inflight": None,
                    "status": "budget_exhausted",
                    "usage": {"model_calls": 2, "tool_calls": 1, "wall_seconds": 20},
                    "progress": {"targets": {"accepted_but_unpublished": {}}},
                }
                write(folder / "campaign/goals/b/goal.json", goal)
                write(
                    folder / "campaign/goals/b/protocol.json",
                    {"admission": admission, "targets": [{"id": task["id"], "task_sha256": digest(task)}]},
                )
            else:
                write(
                    folder / "terminal_summary_v1.json",
                    {
                        "campaign_sha256": csha,
                        "audit_id": aid,
                        "audit_sha256": asha,
                        "usage": charge,
                        "status": campaign["status"],
                    },
                )
    tsha = write(study / "tasks.json", tasks)
    psha = write(
        study / "final_protocol.json",
        {
            "task_bindings": {t["id"]: digest(t) for t in tasks},
            "fit_partition": "training",
            "score_partition": "future",
            "hyperparameter_search": False,
        },
    )
    design = {
        "units": units,
        "repository": str(repository),
        "snapshot_sha256": digest(snapshot),
        "condition_policies": policies,
        "free_model_only": model,
        "budget_per_unit": budget,
        "matched_blocks": 6,
        "tasks_per_block": 1,
        "source_projects": ["fixture"],
        "scientific_scopes": 6,
        "allocated_model_attempts": 18,
        "final_protocol_sha256": psha,
        "tasks_sha256": tsha,
    }
    dsha = write(study / "study_design.json", design)
    lsha = write(study / "launch_basis.json", {"study_design_sha256": dsha, "final_protocol_sha256": psha})
    return tmp_path, study, lsha, failed


def test_closed_failure_retains_full_allocation_without_rewriting_state(study_fixture):
    root, study, launch, failed = study_fixture
    original = (failed / "campaign/campaign.json").read_bytes()
    report = module.prepare(study, launch, repository=root)
    assert len(report["units"]) == 18 and report["allocated_model_attempts"] == 18
    row = report["units"][failed.name]
    assert row["classification"] == "unpublished_failure" and row["published_models"] == 0
    assert len(row["allocated_tasks"]) == 1 and row["accepted_checkpoint_targets_not_counted_as_publications"] == 1
    assert row["charged_usage"]["model_calls"] == 3 and row["goal_checkpoint_usage"]["model_calls"] == 2
    assert (failed / "campaign/campaign.json").read_bytes() == original
    with pytest.raises(ValueError, match="every acquisition batch"):
        require_terminal(read(failed / "campaign/campaign.json"), runner_live=False)
    path = study / "closeout.json"
    pin = write(path, report)
    facts, _, _, contexts = module.verify(study, {"path": str(path), "sha256": pin}, repository=root)
    assert facts["allocated_model_attempts"] == 18 and contexts[failed.name][2]["unpublished_failure"]


@pytest.mark.parametrize(
    "relative",
    ["index/orphan.json", "packets/orphan.json", "origins/x/orphan.json", "publication_attempts/orphan.json"],
)
def test_partial_publication_even_without_an_index_blocks_closeout(study_fixture, relative):
    root, study, launch, failed = study_fixture
    write(failed / "model_library" / relative, {})
    with pytest.raises(ValueError, match="partial publication"):
        module.prepare(study, launch, repository=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "owner",
        "inflight",
        "active",
        "queued",
        "partial_publication",
        "mixed_attempt",
        "negative_charge",
        "charge_mismatch",
    ],
)
def test_incomplete_or_inconsistent_failure_cannot_unlock_scoring(study_fixture, mutation):
    root, study, launch, failed = study_fixture
    path = failed / "campaign/campaign.json"
    state = read(path)
    attempt = state["batches"]["b"]["attempts"][0]
    if mutation == "owner":
        state["owner"] = {"pid": 2147483647}
    elif mutation == "inflight":
        state["inflight"] = {"batch": "b"}
    elif mutation == "active":
        state["status"] = "active"
    elif mutation == "queued":
        state["batches"]["b"]["status"] = "queued"
    elif mutation == "partial_publication":
        attempt["publication"] = {"outcomes": []}
    elif mutation == "mixed_attempt":
        attempt["status"] = "completed"
    elif mutation == "negative_charge":
        attempt["charged"]["model_calls"] = state["usage"]["model_calls"] = -1
    else:
        state["usage"]["model_calls"] += 1
    write(path, state)
    with pytest.raises(ValueError):
        module.prepare(study, launch, repository=root)


def test_any_live_study_runner_blocks_before_inspecting_units(study_fixture, monkeypatch):
    root, study, launch, _ = study_fixture
    monkeypatch.setattr(module, "process_live", lambda _: True)
    with pytest.raises(ValueError, match="runner is live"):
        module.prepare(study, launch, repository=root)


def test_absent_library_cannot_be_confused_with_a_later_created_library(study_fixture):
    root, study, launch, failed = study_fixture
    library = failed / "model_library"
    library.rmdir()
    report = module.prepare(study, launch, repository=root)
    assert report["units"][failed.name]["library_inventory"] == [{"path": ".", "kind": "absent"}]
    path = study / "closeout.json"
    pin = write(path, report)
    module.verify(study, {"path": str(path), "sha256": pin}, repository=root)
    library.mkdir()
    with pytest.raises(ValueError, match="evidence changed"):
        module.verify(study, {"path": str(path), "sha256": pin}, repository=root)


def test_symlink_in_failed_library_cannot_hide_a_publication(study_fixture):
    root, study, launch, failed = study_fixture
    (failed / "model_library/foreign").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        module.prepare(study, launch, repository=root)


@pytest.mark.parametrize(
    "mutation", ["goal_inflight", "goal_active", "goal_tasks", "library_binding", "scope", "runtime"]
)
def test_goal_scope_and_frozen_implementation_are_bound(study_fixture, mutation):
    root, study, launch, failed = study_fixture
    if mutation.startswith("goal_"):
        path = failed / ("campaign/goals/b/protocol.json" if mutation == "goal_tasks" else "campaign/goals/b/goal.json")
        value = read(path)
        if mutation == "goal_tasks":
            value["targets"][0]["task_sha256"] = "other"
        elif mutation == "goal_active":
            value["status"] = "active"
        else:
            value["inflight"] = {"cycle": 1}
        write(path, value)
    elif mutation == "library_binding":
        path = failed / "campaign/campaign.json"
        value = read(path)
        value["definition"]["library_root"] = str(root / "another_empty_library")
        write(path, value)
    elif mutation == "scope":
        path = failed / "plan.json"
        value = read(path)
        value["batches"][0]["tasks"][0]["scope"] = {"different": True}
        write(path, value)
    else:
        (study / "repository/src/runtime_fixture.py").write_text("changed")
    with pytest.raises(ValueError):
        module.prepare(study, launch, repository=root)


@pytest.mark.parametrize("mutation", ["late_packet", "late_runner", "goal_usage"])
def test_certificate_is_revalidated_after_creation(study_fixture, mutation):
    root, study, launch, failed = study_fixture
    report = module.prepare(study, launch, repository=root)
    path = study / "closeout.json"
    pin = write(path, report)
    if mutation == "late_packet":
        write(failed / "model_library/packets/late.json", {})
    elif mutation == "late_runner":
        write(study / "runner_identity_v3.json", {"pid": 2147483647, "start_ticks": "0", "boot_id": "no-such-boot"})
    else:
        goal = failed / "campaign/goals/b/goal.json"
        value = read(goal)
        value["usage"]["model_calls"] += 1
        write(goal, value)
    with pytest.raises(ValueError):
        module.verify(study, {"path": str(path), "sha256": pin}, repository=root)


@pytest.mark.parametrize("encoding", [str, int])
def test_current_live_process_cannot_pass_barrier_as_integer_ticks(encoding):
    from agent.rsi.campaigns.scientist_model_campaign import process_identity

    identity = process_identity()
    identity["start_ticks"] = encoding(identity["start_ticks"])
    assert module.process_live(identity) is True


@pytest.mark.parametrize(
    "identity",
    [
        {"pid": True},
        {"pid": "1"},
        {"pid": -1},
        {"pid": 1, "start_ticks": True, "boot_id": "fixture"},
        {"pid": 1, "start_ticks": "x", "boot_id": "fixture"},
    ],
)
def test_malformed_process_identity_is_not_treated_as_dead(identity):
    with pytest.raises(ValueError):
        module.process_live(identity)
