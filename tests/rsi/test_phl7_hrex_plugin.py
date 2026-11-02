"""Public HREX integration remains a bounded, source-bound plugin allocation."""

import copy
from pathlib import Path

import pytest

from agent.rsi.case_evidence import canonical_claims, render_evidence_report
from agent.rsi.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.phl7_case import ARCHIVE_SHA256
from agent.rsi.phl7_hrex import PLUGIN_ID, STAGES, canonical_hrex


def report():
    return {
        "status": "completed",
        "execution_origin": "new_computation",
        "compute_backend": "cpu",
        "archive_sha256": ARCHIVE_SHA256,
        "new_molecular_dynamics_performed": True,
        "full_paper_reproduced": False,
        "stages": [{"id": name, "status": "completed"} for name in STAGES],
        "summary": {
            "replicas": 8,
            "ps_per_replica": 4,
            "frames_per_replica": 21,
            "exchange_schedule_events": 19,
            "pair_attempts": 67,
            "accepted_swaps": 42,
            "energy_identities": 24,
            "atoms_per_replica": 40557,
        },
    }


def test_plugin_exposes_only_registered_backend_and_bounded_scope(tmp_path):
    entry = next(x for x in list_case_plugins() if x["id"] == PLUGIN_ID)
    assert entry["parameters"]["properties"]["backend"]["enum"] == ["cpu", "cuda"]
    options = {"root": tmp_path, "output": tmp_path / "research_runs/rsi/new", "controller_python": Path("/python")}
    assert invocation(PLUGIN_ID, {"backend": "cuda"}, **options)[-2:] == ["--backend", "cuda"]
    assert resource_for(PLUGIN_ID, {"backend": "cuda"}) == "gpu"
    assert resource_for(PLUGIN_ID, {}) == "cpu"
    for values in [{"steps": 250000000}, {"source": "/tmp/other"}, {"backend": "auto"}, {"replicas": 2}]:
        with pytest.raises(ValueError):
            invocation(PLUGIN_ID, values, **options)


def test_new_simulation_is_distinct_from_full_paper_or_convergence_claims():
    claims = canonical_claims(PLUGIN_ID, report())
    assert claims["new_molecular_dynamics_performed"] and not claims["full_paper_reproduced"]
    assert not claims["convergence_established"] and not claims["production_stability_established"]
    assert "four ps" in render_evidence_report("fixture", claims)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_run",
        "failed_stage",
        "missing_stage",
        "duplicate_stage",
        "long_duration",
        "bool_count",
        "nonfinite_swaps",
        "full_paper",
        "no_md",
    ],
)
def test_unsupported_or_incomplete_evidence_is_rejected(mutation):
    value = copy.deepcopy(report())
    if mutation == "old_run":
        value["execution_origin"] = "verified_existing"
    elif mutation == "failed_stage":
        value["stages"][1]["status"] = "failed"
    elif mutation == "missing_stage":
        value["stages"].pop()
    elif mutation == "duplicate_stage":
        value["stages"][1]["id"] = "preparation"
    elif mutation == "long_duration":
        value["summary"]["ps_per_replica"] = 500000
    elif mutation == "bool_count":
        value["summary"]["replicas"] = True
    elif mutation == "nonfinite_swaps":
        value["summary"]["accepted_swaps"] = float("nan")
    elif mutation == "full_paper":
        value["full_paper_reproduced"] = True
    else:
        value["new_molecular_dynamics_performed"] = False
    with pytest.raises(ValueError):
        canonical_hrex(value)


@pytest.mark.parametrize(
    "mutation", [None, "unverified", "binary_changed", "backend_changed", "escape", "cuda_from_cpu"]
)
def test_runtime_manifest_rejects_unverified_or_changed_dependencies(tmp_path, mutation):
    import json

    from agent.rsi.phl7_hrex import runtime_configuration
    from agent.rsi.storage import sha256_file

    config_path = tmp_path / "research_runs/rsi/runtime_configs/phl7_hrex/cpu.json"
    config_path.parent.mkdir(parents=True)
    prefix = tmp_path / "research_runs/rsi/fixture"
    prefix.mkdir()
    build = prefix / "build"
    build.mkdir()
    (build / "build_result.json").write_text(
        json.dumps({"status": "failed" if mutation == "unverified" else "completed"})
    )
    for name in ["plumed", "gmx_double"]:
        (prefix / name).write_text("binary fixture")
    config = {
        "schema": 1,
        "backend": "cpu",
        "build": str(build.relative_to(tmp_path)),
        "build_report_sha256": sha256_file(build / "build_result.json"),
    }
    for name in ["plumed", "gmx_double"]:
        config[name] = str((prefix / name).relative_to(tmp_path))
        config[name + "_sha256"] = sha256_file(prefix / name)
    if mutation == "binary_changed":
        (prefix / "plumed").write_text("changed executable")
    elif mutation == "backend_changed":
        config["backend"] = "cuda"
    elif mutation == "escape":
        config["build"] = "../outside"
    elif mutation == "cuda_from_cpu":
        config["backend"] = "cuda"
        config_path = config_path.with_name("cuda.json")
    config_path.write_text(json.dumps(config))
    if mutation is None:
        received, paths = runtime_configuration(tmp_path, "cpu")
        assert received == config and paths["build"] == build
    else:
        with pytest.raises(ValueError):
            runtime_configuration(tmp_path, "cuda" if mutation == "cuda_from_cpu" else "cpu")


@pytest.mark.parametrize("child_state", ["absent", None, True, False])
def test_failed_dynamics_preserves_known_progress_or_uncertainty(tmp_path, monkeypatch, child_state):
    import importlib.util
    import json
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pipeline_failure_test", scripts / "run_phl7_hrex_case.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "src/agent/rsi/phl7_hrex.py"
    source.parent.mkdir(parents=True)
    source.write_text("# isolated source fixture")
    build = tmp_path / "build"
    build.mkdir()
    (build / "build_result.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(
        module,
        "runtime_configuration",
        lambda *args: ({}, {"build": build, "plumed": tmp_path / "plumed", "gmx_double": tmp_path / "gmx"}),
    )

    def stage(argv, **kwargs):
        path = Path(argv[argv.index("--output") + 1])
        path.mkdir()
        if path.name == "dynamics":
            if child_state != "absent":
                (path / "report.json").write_text(
                    json.dumps({"status": "failed", "new_molecular_dynamics_performed": child_state})
                )
            return 13
        (path / "report.json").write_text('{"status":"completed"}')
        return 0

    monkeypatch.setattr(module, "run_in_group", stage)
    output = tmp_path / "research_runs/rsi/failure"
    result = module.run(output, "cpu")
    assert result["status"] == "failed" and result["stages"][-1]["id"] == "dynamics"
    assert result["stages"][-1]["status"] == "failed"
    assert result["new_molecular_dynamics_performed"] is (None if child_state == "absent" else child_state)
    assert json.loads((output / "report.json").read_text()) == result
