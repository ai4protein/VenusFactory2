"""Synthetic 19-feature computations and full MCP lifecycle; no scientific models."""

import copy
from pathlib import Path

import numpy as np
import pytest
from test_generated_context import CODE

from agent.rsi import context_feedback_dispatch as feedback
from agent.rsi import context_interfaces, generated_context, model_tasks
from agent.rsi import generated_context_v2 as api
from agent.rsi.context_baselines import run_controls
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.prediction_metrics import measure_context
from agent.rsi.research_project_catalog import build_catalog
from agent.rsi.storage import atomic_json, sha256_file


def scope(columns=19):
    return {
        "interface_version": api.PLUGIN_ID,
        "protein": "Synthetic laboratory yeast protein",
        "assay": "synthetic_context",
        "target": "synthetic_abundance",
        "unit": "synthetic_units",
        "context": {"material": "synthetic laboratory yeast fixture"},
        "covariates": [
            {"name": f"concentration_{i}", "unit": "mM", "minimum": 0.0, "maximum": 5.0} for i in range(columns)
        ],
        "grouping": {"level": "deleted_gene", "evidence_sha256": "a" * 64},
    }


def rows():
    x = np.random.default_rng(421).uniform(0.1, 4.0, (60, 19))
    y = 2 + x @ (np.arange(19) + 1) / 19
    return [
        {
            "id": f"row_{i}",
            "protein": scope()["protein"],
            "group": f"deletion_{i // 10}",
            "values": value.tolist(),
            "target": float(y[i]),
        }
        for i, value in enumerate(x)
    ]


def save(path, value):
    atomic_json(path, value)
    return {"path": str(path), "sha256": sha256_file(path)}


def make_task(folder, protein=None):
    folder.mkdir(parents=True)
    s = scope()
    if protein:
        s["protein"] = protein
    source = save(folder / "source.json", {"kind": "synthetic deletion-group measurements"})
    measurements, assets, assignments = {}, {}, {}
    for phase, start, stop in [("training", 0, 40), ("validation", 40, 50), ("future", 50, 60)]:
        measurements[phase] = rows()[start:stop]
        for row in measurements[phase]:
            row["protein"] = s["protein"]
            row["id"] = s["protein"] + "_" + row["id"]
            assignments[row["id"]] = {"phase": phase, "group": row["group"]}
        assets[phase] = save(folder / (phase + ".json"), measurements[phase])
    evidence = save(
        folder / "groups.json",
        {
            "schema": "context_groups.v1",
            "level": "deleted_gene",
            "study_doi": "10.5555/synthetic-context-v2",
            "sources": [source],
            "observations": assignments,
        },
    )
    s["grouping"]["evidence_sha256"] = evidence["sha256"]
    definition = {
        "schema": 1,
        "id": "synthetic_wide_context",
        "stage": "expression",
        "plugin_id": api.PLUGIN_ID,
        "scope": s,
        "paper": {"title": "Synthetic context fixture", "doi": "10.5555/synthetic-context-v2"},
        "references": [source, evidence],
        "data": assets,
        "minimum_skill": 0.0,
    }
    return definition, measurements, assignments


@pytest.fixture
def contextual(tmp_path):
    return make_task(tmp_path / "assets")


@pytest.mark.parametrize("columns", [1, 16, 19, 64])
def test_explicit_version_and_measured_dimensions(columns):
    s = scope(columns)
    api.validate_scope(s)
    context_interfaces.validate_scope(s)
    assert context_interfaces.module_for(api.PLUGIN_ID) is api
    plugin = get_model_plugin(api.PLUGIN_ID)
    assert plugin.source_sha256 != get_model_plugin(generated_context.PLUGIN_ID).source_sha256


@pytest.mark.parametrize("attack", ["no_version", "wrong_version", "zero", "too_wide", "duplicate", "unbound_group"])
def test_wide_contract_rejects_implicit_or_invalid_inputs(attack):
    s = scope()
    if attack == "no_version":
        del s["interface_version"]
    elif attack == "wrong_version":
        s["interface_version"] = generated_context.PLUGIN_ID
    elif attack in {"zero", "too_wide"}:
        s = scope(0 if attack == "zero" else 65)
    elif attack == "duplicate":
        s["covariates"][1] = copy.deepcopy(s["covariates"][0])
    else:
        s["grouping"]["evidence_sha256"] = "unbound"
    with pytest.raises(ValueError):
        api.validate_scope(s)


def test_legacy_scope_cannot_silently_upgrade():
    s = scope()
    with pytest.raises(ValueError):
        generated_context.validate_scope(s)
    del s["interface_version"]
    s["grouping"]["level"] = "biological_replicate"
    with pytest.raises(ValueError, match="1..16"):
        generated_context.validate_scope(s)
    with pytest.raises(ValueError):
        context_interfaces.validate_scope({**s, "interface_version": "unknown"})
    with pytest.raises(ValueError):
        context_interfaces.module_for("generated_context.v999")


def test_real_worker_uses_all_19_covariates_without_identifiers(monkeypatch):
    data, s = rows(), scope()
    real = api.run_code
    observed = []

    def checked(code, payload):
        assert all(len(v) == 19 for v in payload["features"])
        assert not {"id", "group", "protein", "scope"} & payload.keys()
        assert ("targets" in payload) == (payload["operation"] == "fit")
        observed.append(payload["operation"])
        return real(code, payload)

    monkeypatch.setattr(api, "run_code", checked)
    model = api.fit_candidate(data[:40], s, {"source_code": CODE, "seed": 0})
    query = api.inference_inputs(data[50:])
    actual = api.predict(model, query, s)
    expected = [2 + sum((i + 1) * v / 19 for i, v in enumerate(r["values"])) for r in query]
    np.testing.assert_allclose([r["prediction"] for r in actual], expected, rtol=1e-12, atol=1e-12)
    changed = copy.deepcopy(query)
    changed[0]["values"][-1] += 0.1
    assert api.predict(model, changed, s)[0]["prediction"] - actual[0]["prediction"] == pytest.approx(0.1)
    metrics = measure_context(data[:40], data[50:], actual, scope=s)
    assert metrics["rmse"] < 1e-11 and metrics["grouping_level"] == "deleted_gene"
    assert observed == ["fit", "predict", "predict"]


def test_task_admission_source_grouping_and_catalog(contextual, tmp_path):
    from test_context_integration import catalog_manifest

    definition, data, _ = contextual
    assert model_tasks.load_task(definition) == data
    manifest = catalog_manifest(tmp_path, [contextual])
    manifest["projects"][0]["study_doi"] = definition["paper"]["doi"]
    manifest["candidates"][0]["group_basis"] = "deleted_gene"
    report = build_catalog(manifest)
    assert report["summary"]["statuses"] == {"admitted": 1}
    assert report["summary"]["new_published_models"] == 0
    assert len(report["entries"][0]["baseline_plan"]["implemented"]) == 4
    manifest["candidates"][0]["group_basis"] = "genetic_population"
    assert build_catalog(manifest)["entries"][0]["status"] == "invalid"
    data["future"][0]["group"] = data["training"][0]["group"]
    definition["data"]["future"] = save(Path(definition["data"]["future"]["path"]), data["future"])
    with pytest.raises(ValueError, match="independent group"):
        model_tasks.load_task(definition)


def test_wide_feedback_is_explicit_and_legacy_packets_are_unchanged():
    from test_generated_context import ROWS, SCOPE

    from agent.rsi import context_feedback_profiles as legacy

    for plugin, policy in [(api.PLUGIN_ID, legacy.POLICY), (generated_context.PLUGIN_ID, feedback.WIDE_POLICY)]:
        with pytest.raises(ValueError):
            feedback.policy_for({feedback.FIELD: policy}, plugin_id=plugin)
    assert feedback.policy_for({feedback.FIELD: feedback.WIDE_POLICY}, plugin_id=api.PLUGIN_ID) == feedback.WIDE_POLICY
    assert feedback.policy_for({}, plugin_id=api.PLUGIN_ID) == "none"
    for module, data, s, policy in [
        (generated_context, ROWS, SCOPE, legacy.POLICY),
        (api, rows(), scope(), feedback.WIDE_POLICY),
    ]:
        fitting, scoring = module.development_split(data[:40])
        model = module.fit_candidate(fitting, s, {"source_code": CODE, "seed": 0})
        metrics = module.evaluate(model, scoring, s)
        packet = feedback.comparison(data[:40], s, metrics, policy=policy)
        if module is generated_context:
            assert packet == legacy.comparison(data[:40], s, metrics, policy=policy)
        else:
            assert packet["completed_controls"] == 4
            assert packet["schema"] == feedback.WIDE_POLICY
            assert "generated_context_v2.py" in packet["implementation"]
            assert set(packet["split"]["fit_groups"]).isdisjoint(packet["split"]["score_groups"])


def test_future_controls_do_not_fit_outer_labels(contextual):
    definition, data, _ = contextual
    first = run_controls(definition, score_partition="future")
    assert len(first["outcomes"]) == 4 and all(x["status"] == "completed" for x in first["outcomes"])
    for phase in ("validation", "future"):
        for row in data[phase]:
            row["target"] += 1000
        definition["data"][phase] = save(Path(definition["data"][phase]["path"]), data[phase])
    second = run_controls(definition, score_partition="future")
    assert [r["predictions"] for r in first["outcomes"]] == [r["predictions"] for r in second["outcomes"]]
    assert first["outcomes"][0]["metrics"] != second["outcomes"][0]["metrics"]


@pytest.mark.asyncio
async def test_real_mcp_development_feedback_acceptance_registration_reuse_and_replay(
    contextual, tmp_path, monkeypatch
):
    import test_context_integration as integration

    make_bundle = integration.make_bundle

    def wide_bundle(*args, **kwargs):
        if args[1] == "develop":
            kwargs["training_baseline_feedback"] = feedback.WIDE_POLICY
        result = make_bundle(*args, **kwargs)
        assert (result[2] / "generated_context_v2.py").is_file()
        return result

    monkeypatch.setattr(integration, "api", api)
    monkeypatch.setattr(integration, "make_bundle", wide_bundle)
    await integration.test_real_context_mcp_fit_register_reuse_and_publication_replay(contextual, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("evolving", [False, True])
async def test_scientist_goal_and_evolving_goal_reach_host_with_wide_bundle(
    contextual, tmp_path, monkeypatch, evolving
):
    import test_context_integration as integration

    monkeypatch.setattr(integration, "api", api)
    await integration.test_context_goal_reaches_host_with_production_bundle(contextual, tmp_path, monkeypatch, evolving)
