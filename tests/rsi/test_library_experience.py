"""Transfer contracts and real MCP transport; fixtures never publish models."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from test_generated_models import CODE, SCOPE, TRAIN, VALID

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import library_experience as transfer
from agent.rsi import scientist_model_library
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
from run_paired_generated import make_bundle
from run_science_goal import GeneratedModelGoal, prepare


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    donor = tmp_path / "donor"
    data = {
        "training": copy.deepcopy(TRAIN),
        "validation": copy.deepcopy(VALID),
        "future": [{"id": "future", "sequence": "HI", "target": 917.123}],
    }
    for phase, rows in data.items():
        atomic_json(donor / "data" / (phase + ".json"), rows)
    atomic_json(donor / "protocol.json", {"data_hashes": {phase: digest(rows) for phase, rows in data.items()}})
    identity = "a" * 64
    model = {
        "plugin_id": "generated_python.v1",
        "scope": SCOPE,
        "source_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
        "recipe": {"source_code": CODE},
        "model_state": {"private_fitted_weight": 917.123},
    }
    verified = {
        "entry_id": identity,
        "key_id": "b" * 64,
        "origin_goal": str(donor),
        "origin": {"synthetic": True},
        "packet": {"model": model},
    }
    monkeypatch.setattr(
        transfer, "verified_entries", lambda root, identities: [copy.deepcopy(verified) for _ in identities]
    )
    parts = {
        "training": [{"id": f"t{i}", "sequence": "K" + aa, "target": float(i)} for i, aa in enumerate("ACDEFGHIKL")],
        "validation": [{"id": "v0", "sequence": "PP", "target": 1.0}, {"id": "v1", "sequence": "PQ", "target": 2.0}],
        "future": [{"id": "f0", "sequence": "QQ", "target": 3.0}, {"id": "f1", "sequence": "QR", "target": 4.0}],
    }
    bindings = {}
    for phase, rows in parts.items():
        path = tmp_path / "target" / (phase + ".json")
        atomic_json(path, rows)
        bindings[phase] = {"path": str(path), "sha256": file_hash(path)}
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic unit-test source")
    task = {
        "schema": 1,
        "id": "synthetic_transfer",
        "stage": "activity",
        "plugin_id": "generated_python.v1",
        "scope": {**SCOPE, "protein": "Receiving synthetic fixture"},
        "paper": {"title": "Synthetic test", "doi": "test:fixture"},
        "references": [{"path": str(reference), "sha256": file_hash(reference)}],
        "data": bindings,
        "minimum_skill": 0.0,
    }
    request = {"condition": "source", "library_root": str(tmp_path / "library"), "entry_ids": [identity]}
    return request, task, verified, data, parts


@pytest.mark.parametrize("condition", ["none", "summary", "source"])
def test_conditions_transfer_only_admitted_context(fixture, condition):
    request, task, _, _, _ = fixture
    result = transfer.snapshot({**request, "condition": condition}, task)
    assert len(result["manifest"]["donors"]) == 1
    entries = result["projection"]["entries"]
    assert len(entries) == (0 if condition == "none" else 1)
    if entries:
        assert ("source_code" in entries[0]) == (condition == "source")
        assert "numpy" in entries[0]["method_summary"]
    public = json.dumps(result["projection"])
    assert "917.123" not in public and "private_fitted_weight" not in public
    assert request["library_root"] not in public


def test_method_summary_describes_code_structure_without_literal_values():
    source = "import numpy as np\ndef fit(sequences, targets, seed):\n return float(np.mean(targets)) + 917.123\ndef predict(model, sequences):\n return [model for _ in sequences]\n"
    summary = transfer.describe_source(source)
    assert "numpy.mean" in summary and "fit" in summary and "predict" in summary
    assert "917.123" not in summary and "return" not in summary


@pytest.mark.parametrize("phase", ["training", "validation", "future"])
def test_any_donor_cohort_overlap_with_target_holdouts_is_rejected(fixture, phase):
    request, task, verified, data, _ = fixture
    root = Path(verified["origin_goal"])
    data[phase][0] = {**data[phase][0], "sequence": "QQ"}
    atomic_json(root / "data" / (phase + ".json"), data[phase])
    atomic_json(root / "protocol.json", {"data_hashes": {key: digest(rows) for key, rows in data.items()}})
    with pytest.raises(transfer.IneligibleDonor, match="overlap"):
        transfer.snapshot(request, task)


def test_changed_donor_cohort_fails_even_after_resolver_returned(fixture):
    request, task, verified, _, _ = fixture
    atomic_json(Path(verified["origin_goal"]) / "data/training.json", [])
    with pytest.raises(ValueError, match="cohort changed"):
        transfer.snapshot(request, task)


def test_transitive_ancestry_is_screened_without_using_unrelated_cached_donors(tmp_path, fixture):
    request, task, verified, data, _ = fixture
    ancestor_root = tmp_path / "ancestor"
    ancestor_data = copy.deepcopy(data)
    ancestor_data["training"][0]["sequence"] = "QQ"
    for phase, rows in ancestor_data.items():
        atomic_json(ancestor_root / "data" / (phase + ".json"), rows)
    atomic_json(
        ancestor_root / "protocol.json", {"data_hashes": {phase: digest(rows) for phase, rows in ancestor_data.items()}}
    )
    direct_key = (request["library_root"], request["entry_ids"][0])
    ancestor_key = (request["library_root"], "d" * 64)
    with transfer.audit_session() as audit:
        audit["verified"][ancestor_key] = {
            **verified,
            "origin_goal": str(ancestor_root),
            "origin": {"ancestor_fixture": True},
        }
        audit["dependencies"][direct_key] = [ancestor_key]
        with pytest.raises(transfer.IneligibleDonor, match="ancestry cohorts overlap"):
            transfer.snapshot(request, task)
        audit["dependencies"][direct_key] = []
        result = transfer.snapshot(request, task)
        assert len(result["manifest"]["donors"][0]["ancestry"]) == 1


@pytest.mark.parametrize("attack", [None, "missing_result", "failed_result", "wrong_call", "reuse"])
def test_publication_requires_successful_read_of_frozen_context(tmp_path, fixture, attack):
    request, task, _, _, _ = fixture
    value = transfer.snapshot(request, task)
    root = tmp_path / "episode_goal"
    root.mkdir()
    binding = transfer.freeze(root, value)
    state, protocol = {"definition": {"library_experience": binding}}, {"library_experience": binding}
    folder = root / "episodes/cycle_000"
    caps = folder / "capability/bundles/versions/fixture"
    atomic_json(caps / "task.json", {"phase": "develop", "library_experience_sha256": digest(value["projection"])})
    atomic_json(caps / "library_experience.json", value["projection"])
    atomic_json(folder / "execution/profile.lock.json", {"task": {"tools": [transfer.TOOL]}})
    events = [
        {"kind": "tool_call_start", "tool_call_id": "read-1", "tool_name": "mcp__venusfactory__" + transfer.TOOL},
        {"kind": "tool_result", "tool_call_id": "read-1", "is_error": False},
    ]
    if attack == "missing_result":
        events.pop()
    elif attack == "failed_result":
        events[-1]["is_error"] = True
    elif attack == "wrong_call":
        events[-1]["tool_call_id"] = "other-call"
    (folder / "execution/events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    cycle = {"index": 0, "phase": "reuse" if attack == "reuse" else "develop", "status": "completed"}
    if attack:
        with pytest.raises(ValueError):
            transfer.verify_episode(root, state, protocol, cycle)
    else:
        transfer.verify_episode(root, state, protocol, cycle)


def test_frozen_projection_and_goal_admission_cannot_be_replaced(tmp_path, fixture):
    request, task, _, _, _ = fixture
    value = transfer.snapshot(request, task)
    root = tmp_path / "receiving_goal"
    root.mkdir()
    binding = transfer.freeze(root, value)
    atomic_json(root / "model_task.json", task)
    state = {"definition": {"library_experience": binding}}
    protocol = {"library_experience": binding}
    assert transfer.verify_goal(root, state, protocol) == value["projection"]
    changed = copy.deepcopy(state)
    changed["definition"].pop("library_experience")
    with pytest.raises(ValueError, match="admission changed"):
        transfer.verify_goal(root, changed, protocol)
    atomic_json(root / "library_experience/projection.json", {**value["projection"], "entries": []})
    with pytest.raises(ValueError, match="Frozen library experience changed"):
        transfer.verify_goal(root, state, protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["none", "summary", "source"])
async def test_real_mcp_keeps_local_lineage_and_library_context_separate(tmp_path, fixture, condition):
    request, task, _, _, parts = fixture
    local = transfer.snapshot(request, task)["projection"]
    external = transfer.snapshot({**request, "condition": condition}, task)["projection"]
    versions, version, caps = make_bundle(
        tmp_path / "capabilities",
        "develop",
        {"training.json": parts["training"]},
        task["scope"],
        {"provider": "test_fixture"},
        experience=local,
        library_experience=external,
    )
    versions.verify(version)
    work = tmp_path / "workspace"
    atomic_json(work / "training.json", parts["training"])
    config = {
        "mcpServers": {
            "test": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }
    async with Client(config) as client:
        for name, expected in [("read_model_experience", local), (transfer.TOOL, external)]:
            actual = (await client.call_tool(name, {})).data
            validate_generated_result(name, actual, workspace=work, capability=caps)
            assert actual == expected
    with pytest.raises(ValueError, match="only during development"):
        make_bundle(tmp_path / "bad_reuse", "reuse", {}, task["scope"], {}, library_experience=external)


def test_goal_preparation_freezes_transfer_before_any_execution(tmp_path, fixture):
    request, task, _, _, _ = fixture
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    )
    for name, value in [("profile", profile.public()), ("task", task), ("selection", request)]:
        atomic_json(tmp_path / (name + ".json"), value)
    store = prepare(
        SimpleNamespace(
            output=tmp_path / "prepared",
            profile=tmp_path / "profile.json",
            objective="Synthetic transfer admission",
            task_definition=tmp_path / "task.json",
            experience_selection=tmp_path / "selection.json",
            max_cycles=4,
            model_calls=12,
            tool_calls=8,
            wall_seconds=60,
        )
    )
    driver = GeneratedModelGoal(store, "synthetic-no-network")
    assert driver.library_experience["entries"][0]["source_code"] == CODE
    assert store.read()["usage"]["model_calls"] == 0
    assert driver.development_experience(store.read()) is None


def test_automatic_selection_skips_ineligible_but_never_invalid_evidence(fixture, monkeypatch):
    request, task, _, _, _ = fixture
    original = transfer.snapshot
    bad, good = "c" * 64, request["entry_ids"][0]

    def eligible(value, target):
        if value["entry_ids"] == [bad]:
            raise transfer.IneligibleDonor("overlap")
        return original(value, target)

    monkeypatch.setattr(transfer, "snapshot", eligible)
    policy = {"condition": "source", "selection": "prior_campaign", "max_entries": 1}
    selected, review = transfer.select_for_campaign(policy, request["library_root"], [bad, good], task)
    assert selected["entry_ids"] == [good] and review[0]["status"] == "ineligible"

    def changed(*args):
        raise ValueError("Changed source")

    monkeypatch.setattr(transfer, "snapshot", changed)
    with pytest.raises(ValueError, match="Changed source"):
        transfer.select_for_campaign(policy, request["library_root"], [bad, good], task)


def test_thousand_donor_dependency_chain_is_iterative_and_cycles_fail(tmp_path, monkeypatch):
    count, calls = 1005, []
    for index in range(count):
        atomic_json(tmp_path / f"goals/g{index}/goal.json", {"definition": {}})
        atomic_json(tmp_path / f"goals/g{index}/protocol.json", {"dependency": index - 1})

    class Library:
        def __init__(self, root):
            self.owner = tmp_path

        def _object(self, area, identity):
            index = int(identity, 16)
            return (
                {"key": {"id": index}, "canonical_origin": identity}
                if area == "entries"
                else {"goal": f"goals/g{index}"}
            )

        def goal(self, path):
            return path

        def resolve(self, identity, *, replay_training):
            assert replay_training
            index = int(identity, 16)
            if index:
                transfer.verified_entries(str(tmp_path / "library"), [f"{index - 1:064x}"])
            calls.append(index)
            return {"synthetic_index": index}

    def frozen(root, state, protocol):
        child = protocol["dependency"]
        return (
            {"manifest": {"selection": {"library_root": str(tmp_path / "library"), "entry_ids": [f"{child:064x}"]}}}
            if child >= 0
            else None
        )

    monkeypatch.setattr(scientist_model_library, "ScientistModelLibrary", Library)
    monkeypatch.setattr(transfer, "read_frozen", frozen)
    result = transfer.verified_entries(str(tmp_path / "library"), [f"{count - 1:064x}"])
    assert result == [{"synthetic_index": count - 1}] and calls == list(range(count))
    atomic_json(tmp_path / "goals/g0/protocol.json", {"dependency": count - 1})
    with pytest.raises(ValueError, match="Cyclic"):
        transfer.verified_entries(str(tmp_path / "library"), [f"{count - 1:064x}"])
    assert transfer._AUDIT.get() is None
