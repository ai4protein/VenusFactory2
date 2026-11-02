"""Study-bound contextual source reuse, with isolated synthetic numerical refits.

No fixture is a real publication or external-model acquisition.
"""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from test_generated_context import CODE, ROWS, SCOPE

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.rsi import context_transfer as policy
from agent.rsi import generated_context as api
from agent.rsi import library_experience as transfer
from agent.rsi import model_tasks
from agent.rsi import scientist_model_campaign as campaign
from agent.rsi.datasets.library_source import REFIT_TOOL
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.storage import atomic_json
from agent.rsi.engine.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
from run_paired_generated import make_bundle
from run_science_goal import GeneratedModelGoal, prepare


def task(root, study="10.1234/receiver", protein="Receiver", *, slope=-0.02):
    scope = deepcopy(SCOPE)
    scope["protein"] = protein
    rows = deepcopy(ROWS)
    for row in rows:
        row.update(protein=protein, target=4.0 + slope * row["values"][0])
        row["id"] = protein + "_" + row["id"]
    data = {"training": rows[:40], "validation": rows[40:50], "future": rows[50:]}
    source = root / "source.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("Synthetic grouping review for " + study)
    source_binding = {"path": str(source), "sha256": file_hash(source)}
    evidence = {
        "schema": "context_groups.v1",
        "level": "biological_replicate",
        "study_doi": study,
        "sources": [source_binding],
        "observations": {
            row["id"]: {"group": row["group"], "phase": phase} for phase, values in data.items() for row in values
        },
    }
    evidence_path = root / "groups.json"
    atomic_json(evidence_path, evidence)
    scope["grouping"]["evidence_sha256"] = file_hash(evidence_path)
    bindings = {}
    for phase, values in data.items():
        path = root / "data" / (phase + ".json")
        atomic_json(path, values)
        bindings[phase] = {"path": str(path), "sha256": file_hash(path)}
    definition = {
        "schema": 1,
        "id": "synthetic_" + protein.lower(),
        "stage": "stability",
        "plugin_id": api.PLUGIN_ID,
        "scope": scope,
        "paper": {"title": "Synthetic context transfer fixture", "doi": study},
        "references": [source_binding, {"path": str(evidence_path), "sha256": file_hash(evidence_path)}],
        "data": bindings,
        "minimum_skill": 0.0,
    }
    atomic_json(root / "model_task.json", definition)
    atomic_json(root / "protocol.json", {"data_hashes": {phase: digest(values) for phase, values in data.items()}})
    assert model_tasks.load_task(definition) == data
    return definition, data


def donor(root, study="10.1234/donor", identity="a" * 64):
    definition, data = task(root, study, "Donor", slope=-0.01)
    return (
        {
            "entry_id": identity,
            "key_id": "b" * 64,
            "origin_goal": str(root),
            "origin": {"synthetic_fixture": True, "study": study},
            "packet": {
                "model": {
                    "plugin_id": api.PLUGIN_ID,
                    "scope": definition["scope"],
                    "recipe": {"source_code": CODE},
                    "source_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
                    "model_state": {"private_fitted_weight": 917.123},
                }
            },
        },
        definition,
        data,
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    value, definition, data = donor(tmp_path / "donor")
    target, parts = task(tmp_path / "target")
    request = {
        "condition": "source",
        "library_root": str(tmp_path / "library"),
        "entry_ids": [value["entry_id"]],
        "overlap_policy": policy.POLICY,
    }
    monkeypatch.setattr(transfer, "verified_entries", lambda root, ids: [deepcopy(value) for _ in ids])
    return request, target, value, data, parts


@pytest.mark.parametrize("condition", ["none", "summary", "source"])
def test_distinct_reviewed_studies_freeze_groups_without_transferring_measurements(fixture, condition):
    request, target, value, _, _ = fixture
    result = transfer.snapshot({**request, "condition": condition}, target)
    assert result["manifest"]["context_admission"]["study_doi"] == "10.1234/receiver"
    ancestor = result["manifest"]["donors"][0]["ancestry"][0]
    assert ancestor["context_admission"]["study_doi"] == "10.1234/donor"
    assert ancestor["context_admission"]["sample_groups"]["future"] == ["g5"]
    public = json.dumps(result["projection"])
    assert "917.123" not in public and "private_fitted_weight" not in public
    assert value["origin_goal"] not in public and "sample_groups" not in public
    entries = result["projection"]["entries"]
    assert len(entries) == (condition != "none")
    if entries:
        assert ("source_code" in entries[0]) == (condition == "source")


@pytest.mark.parametrize(
    "alias",
    [
        "10.1234/Donor",
        "https://doi.org/10.1234/donor",
        "http://dx.doi.org/10.1234/donor",
        "doi: 10.1234/donor",
        "10.1234%2Fdonor",
    ],
)
def test_same_study_aliases_are_rejected_even_for_different_protein_and_group_names(tmp_path, fixture, alias):
    request, _, _, _, _ = fixture
    target, _ = task(tmp_path / "alias_target", alias, "DifferentProtein")
    with pytest.raises(transfer.IneligibleDonor, match="study overlaps"):
        transfer.snapshot(request, target)


def test_explicit_policy_required_and_unknown_or_wrong_interface_policies_rejected(fixture):
    request, target, _, _, _ = fixture
    missing = {k: v for k, v in request.items() if k != "overlap_policy"}
    with pytest.raises(transfer.IneligibleDonor, match="separately reviewed"):
        transfer.snapshot(missing, target)
    with pytest.raises(ValueError, match="Unsupported"):
        transfer.selection({**request, "overlap_policy": "allow_any_study"})
    with pytest.raises(ValueError, match="context model interface"):
        policy.validate_declaration(request, plugin_id="generated_python.v1")
    historical = transfer.snapshot({**missing, "entry_ids": []}, target)
    assert "context_admission" not in historical["manifest"]
    assert "overlap_policy" not in historical["manifest"]["selection"]


@pytest.mark.parametrize("field", ["assay", "target", "unit", "name", "covariate_unit", "order"])
def test_incompatible_numeric_interfaces_are_rejected(fixture, field):
    request, target, _, _, _ = fixture
    other = deepcopy(target)
    scope = other["scope"]
    if field in {"assay", "target", "unit"}:
        scope[field] = "different"
    elif field == "name":
        scope["covariates"][0]["name"] = "pH"
    elif field == "covariate_unit":
        scope["covariates"][0]["unit"] = "kelvin"
    else:
        left = deepcopy(scope)
        left["covariates"].append({"name": "pH", "unit": "pH", "minimum": 0.0, "maximum": 14.0})
        right = deepcopy(left)
        right["covariates"].reverse()
        assert not transfer.compatible(api.PLUGIN_ID, api.PLUGIN_ID, left, right)
        return
    with pytest.raises(transfer.IneligibleDonor, match="Incompatible donor"):
        transfer.snapshot(request, other)


def test_ranges_and_protein_context_may_differ_because_recipients_refit(fixture):
    request, target, _, _, _ = fixture
    other = deepcopy(target)
    other["scope"]["covariates"][0]["minimum"] = 0.0
    other["scope"]["context"]["material"] = "different permitted preparation"
    assert transfer.snapshot(request, other)["projection"]["entries"]


def test_matching_ancestor_study_rejected_but_unrelated_cached_entries_do_not_block(tmp_path, fixture):
    request, target, direct, _, _ = fixture
    ancestor, _, _ = donor(tmp_path / "ancestor", "10.1234/receiver", "c" * 64)
    direct_key = (request["library_root"], direct["entry_id"])
    ancestor_key = (request["library_root"], ancestor["entry_id"])
    with transfer.audit_session() as audit:
        audit["verified"][ancestor_key] = ancestor
        audit["dependencies"][direct_key] = [ancestor_key]
        with pytest.raises(transfer.IneligibleDonor, match="ancestry contextual study"):
            transfer.snapshot(request, target)
        audit["dependencies"][direct_key] = []
        assert len(transfer.snapshot(request, target)["manifest"]["donors"][0]["ancestry"]) == 1


def test_changed_donor_groups_fail_as_invalid_evidence_not_skippable_ineligibility(fixture):
    request, target, value, _, _ = fixture
    (Path(value["origin_goal"]) / "groups.json").write_text("{}")
    with pytest.raises(ValueError, match="Task asset changed") as caught:
        transfer.snapshot(request, target)
    assert not isinstance(caught.value, transfer.IneligibleDonor)


def test_qualified_sample_ids_cannot_be_relabelled_under_another_study(tmp_path):
    target, parts = task(tmp_path / "target")
    path = Path(target["references"][-1]["path"])
    evidence = json.loads(path.read_text())
    for rows in parts.values():
        for row in rows:
            row["group"] = "10.1234/donor:" + row["group"]
            evidence["observations"][row["id"]]["group"] = row["group"]
    atomic_json(path, evidence)
    target["references"][-1]["sha256"] = file_hash(path)
    target["scope"]["grouping"]["evidence_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="sample study differs"):
        policy.describe_task(target, parts)


@pytest.mark.parametrize("selection", ["explicit", "prior_campaign"])
def test_campaign_selection_preserves_policy_without_reading_target_scores(fixture, selection):
    request, target, _, _, _ = fixture
    declaration = {"condition": "source", "selection": selection, "overlap_policy": policy.POLICY}
    declaration.update({"entry_ids": request["entry_ids"]} if selection == "explicit" else {"max_entries": 1})
    chosen, _ = transfer.select_for_campaign(declaration, request["library_root"], request["entry_ids"], target)
    assert chosen == request


def test_frozen_context_provenance_is_recomputed(tmp_path, fixture):
    request, target, _, _, _ = fixture
    frozen = transfer.snapshot(request, target)
    root = tmp_path / "goal"
    root.mkdir()
    binding = transfer.freeze(root, frozen)
    atomic_json(root / "model_task.json", target)
    state, protocol = {"definition": {"library_experience": binding}}, {"library_experience": binding}
    assert transfer.verify_goal(root, state, protocol) == frozen["projection"]
    changed = deepcopy(frozen["manifest"])
    changed["context_admission"]["study_doi"] = "10.1234/forged"
    atomic_json(root / "library_experience/manifest.json", changed)
    binding["manifest"]["sha256"] = file_hash(root / "library_experience/manifest.json")
    with pytest.raises(ValueError, match="provenance changed"):
        transfer.verify_goal(root, state, protocol)


@pytest.mark.parametrize("adaptation", [False, True])
def test_production_goal_and_campaign_bind_the_declared_context_policy(tmp_path, monkeypatch, adaptation):
    target, _ = task(tmp_path / "target")
    profile = Profile(
        "science_expert",
        "expert-kimi-code",
        ModelBinding("test_fixture", "model", "https://example.org/v1"),
        Budget(wall_seconds=10, request_seconds=5, model_calls=3, tool_calls=2),
    )
    allocation = {
        "id": "recipient",
        "task": target,
        "profile": profile.public(),
        "generations": 1,
        "goal_budget": {"max_cycles": 4, "model_calls": 12, "tool_calls": 8, "wall_seconds": 60},
        "experience": {
            "condition": "source",
            "selection": "explicit",
            "entry_ids": [],
            "overlap_policy": policy.POLICY,
        },
    }
    repository = tmp_path / "repo"
    if adaptation:
        allocation["experience"]["adaptation_policy"] = policy.SOURCE_ADAPTATION
    repository.mkdir()
    (repository / "fixture.py").write_text("# synthetic\n")
    store = campaign.ModelCampaignStore.create(
        tmp_path / "campaign",
        objective="Synthetic contextual transfer",
        target_models=1,
        library_root=tmp_path / "library",
        repository=repository,
        implementation={"fixture.py": file_hash(repository / "fixture.py")},
        allocations=[allocation],
        resources={k: allocation["goal_budget"][k] for k in campaign.RESOURCES},
        counting="program_entry.v1",
    )
    request, _ = transfer.select_for_campaign(allocation["experience"], tmp_path / "library", [], target)
    for name, value in [("profile", profile.public()), ("task", target), ("selection", request)]:
        atomic_json(tmp_path / (name + ".json"), value)
    goal = prepare(
        SimpleNamespace(
            output=store.goal("recipient"),
            profile=tmp_path / "profile.json",
            task_definition=tmp_path / "task.json",
            experience_selection=tmp_path / "selection.json",
            objective="Synthetic transfer",
            **allocation["goal_budget"],
        )
    )
    driver = GeneratedModelGoal(goal, "synthetic-no-network")
    assert driver.library_experience["condition"] == "source"
    assert campaign.bind_goal(allocation, goal.root)["usage"]["model_calls"] == 0
    state = goal.read()
    protocol = json.loads((goal.root / "protocol.json").read_text())
    frozen = deepcopy(transfer.read_frozen(goal.root, state, protocol))
    frozen["manifest"]["selection"].pop("adaptation_policy" if adaptation else "overlap_policy")
    monkeypatch.setattr(transfer, "read_frozen", lambda *args: frozen)
    with pytest.raises(ValueError, match="adaptation policy changed" if adaptation else "overlap policy changed"):
        campaign.bind_goal(allocation, goal.root)


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback", ["none", "group_baselines.v1"])
async def test_real_mcp_refit_uses_recipient_targets_and_passes_independent_lifecycle(tmp_path, fixture, feedback):
    request, target, _, donor_data, parts = fixture
    value = transfer.snapshot(request, target)
    work = tmp_path / "workspace"
    atomic_json(work / "training.json", parts["training"])
    versions, version, caps = make_bundle(
        tmp_path / "capability",
        "develop",
        {"training.json": parts["training"]},
        target["scope"],
        {"provider": "synthetic"},
        plugin_id=api.PLUGIN_ID,
        library_experience=value["projection"],
        training_baseline_feedback=feedback,
    )
    config = {
        "mcpServers": {
            "context": {
                "command": sys.executable,
                "args": ["-u", str(caps / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }
    async with Client(config) as client:

        async def checked(name, args):
            result = (await client.call_tool(name, args)).data
            validate_generated_result(name, result, workspace=work, capability=caps)
            return result

        await checked(transfer.TOOL, {})
        identity = value["projection"]["entries"][0]["experience_id"]
        result = await checked(REFIT_TOOL, {"experience_id": identity, "rationale": "Refit synthetic recipient"})
        assert result["attempts_remaining"] == 1
        assert ("baseline_comparison" in result["training_only_feedback"]) == (feedback != "none")
        await checked("submit_model_code", {"source_sha256": result["source_sha256"], "rationale": "Select refit"})
    model = json.loads((work / "candidate.json").read_text())
    assert model == api.fit_candidate(parts["training"], target["scope"], {"source_code": CODE, "seed": 314159})
    assert (
        model["model_state"]
        != api.fit_candidate(
            donor_data["training"],
            value["projection"]["entries"][0]["donor_scope"],
            {"source_code": CODE, "seed": 314159},
        )["model_state"]
    )
    lc = ModelLifecycle(tmp_path / "lifecycle", model_tasks.policy_for_task(target, parts))
    mid = lc.submit(work / "candidate.json", session_id="develop", mode="science_expert")
    assert lc.assess(mid, training=parts["training"], validation=parts["validation"])["passed"]
    lc.export(mid, tmp_path / "registered")
    inputs = api.inference_inputs(parts["future"])
    predictions = api.predict(model, inputs, target["scope"])
    assert (
        lc.record_reuse(
            mid, session_id="fresh", mode="science_agent", inputs=inputs, predictions=predictions, scope=target["scope"]
        )["rows"]
        == 10
    )
    versions.verify(version)
