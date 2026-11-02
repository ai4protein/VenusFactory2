"""Optional numerical method transfer; fixtures never count as scientific models."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client
from test_context_transfer import donor, task  # noqa: F401
from test_context_transfer import fixture as context_fixture  # noqa: F401 - registered by pytest
from test_generated_context import CODE

from agent.harness.contracts import digest, file_hash
from agent.rsi import context_interfaces, context_transfer, library_experience, model_tasks
from agent.rsi.engine.campaign_evaluation import paired_design
from agent.rsi.datasets.library_source import REFIT_TOOL
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.storage import atomic_json
from agent.rsi.engine.tool_contracts import validate_generated_result

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_paired_generated import make_bundle


def receiver(definition, parts, plugin_id="generated_context.v2"):
    """Different measurements and dimensions, retaining source-bound sample groups."""
    definition, parts = deepcopy(definition), deepcopy(parts)
    definition["plugin_id"] = plugin_id
    scope = definition["scope"]
    scope.update(assay="synthetic alternative assay", target="alternative endpoint", unit="scaled_units")
    if plugin_id.endswith("v2"):
        scope["interface_version"] = plugin_id
    count = 19 if plugin_id.endswith("v2") else 3
    scope["covariates"] = [
        {"name": f"new_measurement_{i}", "unit": "scaled_units", "minimum": -1000, "maximum": 1000}
        for i in range(count)
    ]
    for phase, rows in parts.items():
        for row in rows:
            row["values"] = [row["values"][0] / 100] + [float(i) for i in range(1, count)]
        path = Path(definition["data"][phase]["path"])
        atomic_json(path, rows)
        definition["data"][phase]["sha256"] = file_hash(path)
    assert model_tasks.load_task(definition) == parts
    return definition, parts


def adapted(request):
    return {**request, "adaptation_policy": context_transfer.SOURCE_ADAPTATION}


@pytest.mark.parametrize("plugin_id", ["generated_context.v1", "generated_context.v2"])
@pytest.mark.parametrize("condition", ["none", "summary", "source"])
def test_explicit_adaptation_admits_method_across_inputs_and_endpoints(request, plugin_id, condition):
    request, definition, donor_value, _, parts = request.getfixturevalue("context_fixture")
    definition, _ = receiver(definition, parts, plugin_id)
    with pytest.raises(library_experience.IneligibleDonor, match="Incompatible donor"):
        library_experience.snapshot(request, definition)
    packet = library_experience.snapshot(adapted({**request, "condition": condition}), definition)
    assert packet["manifest"]["selection"]["adaptation_policy"] == context_transfer.SOURCE_ADAPTATION
    entries = packet["projection"]["entries"]
    assert len(entries) == (condition != "none")
    if entries:
        assert entries[0]["donor_scope"] == donor_value["packet"]["model"]["scope"]
        assert "Numeric source adaptation precedent" in entries[0]["method_summary"]
        assert ("source_code" in entries[0]) == (condition == "source")
    assert "private_fitted_weight" not in json.dumps(packet["projection"])
    assert "917.123" not in json.dumps(packet["projection"])


def test_adaptation_requires_known_policy_and_contextual_study_boundary(request):
    request, _, _, _, _ = request.getfixturevalue("context_fixture")
    with pytest.raises(ValueError, match="study-disjoint"):
        library_experience.selection({k: v for k, v in adapted(request).items() if k != "overlap_policy"})
    with pytest.raises(ValueError, match="Unsupported contextual source adaptation"):
        library_experience.selection({**request, "adaptation_policy": "infer_anything"})
    with pytest.raises(ValueError, match="context model interface"):
        context_transfer.validate_declaration(adapted(request), plugin_id="generated_sequence.v1")


def test_same_study_and_ancestral_study_remain_ineligible(tmp_path, request):
    request, definition, direct, _, parts = request.getfixturevalue("context_fixture")
    definition, _ = receiver(definition, parts)
    same, same_parts = task(tmp_path / "same", "https://doi.org/10.1234/donor", "OtherProtein")
    same, _ = receiver(same, same_parts)
    with pytest.raises(library_experience.IneligibleDonor, match="study overlaps"):
        library_experience.snapshot(adapted(request), same)
    ancestor, _, _ = donor(tmp_path / "ancestor", "10.1234/receiver", "c" * 64)
    parent = (request["library_root"], direct["entry_id"])
    grandparent = (request["library_root"], ancestor["entry_id"])
    with library_experience.audit_session() as audit:
        audit["verified"][grandparent] = ancestor
        audit["dependencies"][parent] = [grandparent]
        with pytest.raises(library_experience.IneligibleDonor, match="study overlaps"):
            library_experience.snapshot(adapted(request), definition)


@pytest.mark.parametrize("selection", ["explicit", "prior_campaign"])
def test_campaign_retrieval_preserves_opt_in(request, selection):
    request, definition, _, _, parts = request.getfixturevalue("context_fixture")
    definition, _ = receiver(definition, parts)
    declaration = {
        "condition": "source",
        "selection": selection,
        "overlap_policy": context_transfer.POLICY,
        "adaptation_policy": context_transfer.SOURCE_ADAPTATION,
    }
    declaration.update({"entry_ids": request["entry_ids"]} if selection == "explicit" else {"max_entries": 1})
    chosen, _ = library_experience.select_for_campaign(
        declaration, request["library_root"], request["entry_ids"], definition
    )
    assert chosen == adapted(request)


def test_matched_design_requires_identical_adaptation_rules():
    experience = {
        "selection": "explicit",
        "entry_ids": ["a" * 64],
        "overlap_policy": context_transfer.POLICY,
        "adaptation_policy": context_transfer.SOURCE_ADAPTATION,
    }
    allocations = {
        "reset": {"id": "reset", "experience": {**experience, "condition": "none"}},
        "retained": {"id": "retained", "experience": {**experience, "condition": "source"}},
    }
    pairs = [{"id": "matched", "left": "reset", "right": "retained", "factor": "library_experience"}]
    assert paired_design(allocations, pairs) == pairs
    allocations["reset"]["experience"].pop("adaptation_policy")
    with pytest.raises(ValueError, match="differ within comparison"):
        paired_design(allocations, pairs)


def test_frozen_adaptation_cannot_be_removed_after_goal_admission(tmp_path, request):
    request, definition, _, _, parts = request.getfixturevalue("context_fixture")
    definition, _ = receiver(definition, parts)
    packet = library_experience.snapshot(adapted(request), definition)
    root = tmp_path / "goal"
    root.mkdir()
    binding = library_experience.freeze(root, packet)
    atomic_json(root / "model_task.json", definition)
    state = {"definition": {"library_experience": binding}}
    protocol = {"library_experience": binding}
    assert library_experience.verify_goal(root, state, protocol) == packet["projection"]
    packet["manifest"]["selection"].pop("adaptation_policy")
    atomic_json(root / "library_experience/manifest.json", packet["manifest"])
    binding["manifest"]["sha256"] = file_hash(root / "library_experience/manifest.json")
    with pytest.raises(library_experience.IneligibleDonor, match="Incompatible"):
        library_experience.verify_goal(root, state, protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin_id", ["generated_context.v1", "generated_context.v2"])
async def test_real_mcp_refits_different_dimension_receiver_and_retains_validation(tmp_path, request, plugin_id):
    request, definition, _, donor_parts, parts = request.getfixturevalue("context_fixture")
    definition, parts = receiver(definition, parts, plugin_id)
    packet = library_experience.snapshot(adapted(request), definition)
    work = tmp_path / "workspace"
    atomic_json(work / "training.json", parts["training"])
    versions, version, caps = make_bundle(
        tmp_path / "capability",
        "develop",
        {"training.json": parts["training"]},
        definition["scope"],
        {"provider": "synthetic"},
        plugin_id=plugin_id,
        library_experience=packet["projection"],
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

        async def checked(name, arguments):
            value = (await client.call_tool(name, arguments)).data
            validate_generated_result(name, value, workspace=work, capability=caps)
            return value

        await checked(library_experience.TOOL, {})
        result = await checked(
            REFIT_TOOL,
            {
                "experience_id": packet["projection"]["entries"][0]["experience_id"],
                "rationale": "Refit the method to recipient measurements and units",
            },
        )
        assert result["attempts_remaining"] == 1
        await checked(
            "submit_model_code", {"source_sha256": result["source_sha256"], "rationale": "Select recipient fit"}
        )
    model = json.loads((work / "candidate.json").read_text())
    api = context_interfaces.module_for(plugin_id)
    assert model == api.fit_candidate(parts["training"], definition["scope"], {"source_code": CODE, "seed": 314159})
    donor_api = context_interfaces.module_for("generated_context.v1")
    old = donor_api.fit_candidate(
        donor_parts["training"],
        packet["projection"]["entries"][0]["donor_scope"],
        {"source_code": CODE, "seed": 314159},
    )
    assert model["model_state"] != old["model_state"]
    lc = ModelLifecycle(tmp_path / "lifecycle", model_tasks.policy_for_task(definition, parts))
    mid = lc.submit(work / "candidate.json", session_id="development", mode="science_expert")
    assert lc.assess(mid, training=parts["training"], validation=parts["validation"])["passed"]
    inputs = model_tasks.inference_inputs(parts["future"], plugin_id)
    predictions = api.predict(model, inputs, definition["scope"])
    receipt = lc.record_reuse(
        mid, session_id="fresh", mode="science_agent", inputs=inputs, predictions=predictions, scope=definition["scope"]
    )
    assert receipt["rows"] == len(inputs)
    assert model["training_sha256"] == digest(parts["training"])
    versions.verify(version)
