import copy
from pathlib import Path

import pytest

from agent.rsi.case_jobs import CaseJobBroker
from agent.rsi.case_plugins import invocation, list_case_plugins
from agent.rsi.rfantibody_benchmark import (
    CONDITIONS,
    SCALE,
    STAGES,
    build_stages,
    canonical_benchmark,
    verify_author_source,
)
from agent.rsi.rfantibody_evidence import COMMIT
from agent.rsi.storage import atomic_json, sha256_file


def report():
    return {
        "status": "completed",
        "computational_chain_completed": True,
        "settings": {"target": "ha", "scale": dict(SCALE), "source_commit": COMMIT},
        "stages": [{"status": "completed", "binding": {"id": name}} for name in STAGES],
        "conditions": {
            p: {
                "allocation": {
                    "backbones": 200,
                    "sequences_per_backbone": 8,
                    "total_sequences": 1600,
                    "hotspot_proportion": float(p),
                },
                "independent_rmsd_count": 1600,
                "sequence_passes": 100,
                "selected_backbone_passes": 20,
                "any_passing_sequence_backbones": 30,
                "selected_backbone_pass_percent": 10.0,
                "any_passing_sequence_backbone_percent": 15.0,
                "max_rmsd_absolute_difference": 0.001,
            }
            for p in CONDITIONS
        },
    }


def test_plugin_never_turns_a_small_allocation_into_a_full_benchmark(tmp_path):
    catalog = next(p for p in list_case_plugins() if p["id"] == "rfantibody.benchmark.v1")
    assert set(catalog["parameters"]["properties"]) == {"target"}

    def call(parameters):
        return invocation(
            "rfantibody.benchmark.v1",
            parameters,
            root=tmp_path,
            output=tmp_path / "research_runs/rsi/run",
            controller_python=Path("/python"),
        )

    assert call({"target": "tcdb"})[-2:] == ["--target", "tcdb"]
    for parameters in [
        {},
        {"target": "unknown"},
        {"target": "ha", "backbones": 2},
        {"target": "ha", "conditions": [0]},
    ]:
        with pytest.raises(ValueError):
            call(parameters)


@pytest.mark.parametrize("backbones,sequences", [(8, 4), (199, 8), (200, 7)])
def test_broker_checks_implicit_full_allocation_before_launch(tmp_path, backbones, sequences):
    broker = CaseJobBroker(
        root=tmp_path,
        workspace=tmp_path / "workspace",
        output=tmp_path / "research_runs/rsi/jobs",
        allowed_plugins={"rfantibody.benchmark.v1"},
        maximum_backbones=backbones,
        maximum_sequences_per_backbone=sequences,
    )
    with pytest.raises(ValueError, match="design budget"):
        broker.submit("rfantibody.benchmark.v1", {"target": "ha"}, "test")
    assert not broker.jobs and not broker.tasks


def test_three_conditions_share_sequence_input_and_preserve_full_counts(tmp_path):
    record = {
        "path": "research_runs/rsi/input.pdb",
        "loops": "H1:7,H2:6,H3:5-13",
        "hotspots": [{"chain": "T", "number": 305}],
    }
    out = tmp_path / "research_runs/rsi/run"
    stages = build_stages(tmp_path, out, record, {"path": "research_runs/rsi/framework.pdb"})
    assert [s.id for s in stages] == list(STAGES)
    assert stages[0].argv[stages[0].argv.index("--num-designs") + 1] == "200"
    assert stages[1].argv[stages[1].argv.index("--seqs-per-struct") + 1] == "8"
    for stage, condition in zip(stages[2:], CONDITIONS, strict=True):
        assert stage.argv[stage.argv.index("--input-quiver") + 1] == str(stages[1].output)
        assert stage.argv[stage.argv.index("--num-recycles") + 1] == "10"
        assert stage.argv[stage.argv.index("--hotspot-show-prop") + 1] == condition
        assert stage.argv[2] == str(out / "author_entry.py")
        assert stage.timeout_s >= 86400


def test_sequence_and_assessment_denominators_are_not_pooled():
    claims = canonical_benchmark(report(), {"target": "ha"})
    assert claims["sequence_design_records"] == 1600 and claims["refinement_assessments"] == 4800
    assert claims["backbones"] == 200 and not claims["experimental_binding_verified"]
    with pytest.raises(ValueError, match="request"):
        canonical_benchmark(report(), {"target": "rbd"})


@pytest.mark.parametrize(
    "mutation", ["missing_condition", "wrong_count", "missing_stage", "overclaimed_backbones", "wrong_percentage"]
)
def test_partial_or_misaggregated_results_cannot_complete(mutation):
    data = copy.deepcopy(report())
    if mutation == "missing_condition":
        del data["conditions"]["0.1"]
    if mutation == "wrong_count":
        data["conditions"]["0.0"]["independent_rmsd_count"] = 1599
    if mutation == "missing_stage":
        data["stages"].pop()
    if mutation == "overclaimed_backbones":
        data["conditions"]["1.0"]["selected_backbone_passes"] = 201
    if mutation == "wrong_percentage":
        data["conditions"]["0.0"]["selected_backbone_pass_percent"] = 5.0
    with pytest.raises(ValueError):
        canonical_benchmark(data)


def test_author_code_tampering_and_manifest_replacement_are_detected(tmp_path):
    code = tmp_path / "model.py"
    code.write_text("original")
    manifest = tmp_path / "author_manifest.json"
    atomic_json(manifest, {"model.py": sha256_file(code)})
    bound = sha256_file(manifest)
    verify_author_source(tmp_path, bound)
    code.write_text("modified")
    with pytest.raises(ValueError, match="implementation"):
        verify_author_source(tmp_path, bound)
    atomic_json(manifest, {"model.py": sha256_file(code)})
    with pytest.raises(ValueError, match="manifest"):
        verify_author_source(tmp_path, bound)


def test_running_stage_cannot_be_promoted_by_a_completed_top_level_status():
    data = report()
    data["stages"][2]["status"] = "running"
    with pytest.raises(ValueError, match="completed"):
        canonical_benchmark(data)


def test_frozen_repository_resource_alias_does_not_change_stage_recipes(tmp_path):
    live = tmp_path / "live"
    data = live / "research_runs/rsi"
    data.mkdir(parents=True)
    frozen = tmp_path / "frozen"
    (frozen / "research_runs").mkdir(parents=True)
    (frozen / "research_runs/rsi").symlink_to(data, target_is_directory=True)
    record = {
        "path": "research_runs/rsi/input.pdb",
        "loops": "H1:7,H2:6,H3:5-13",
        "hotspots": [{"chain": "T", "number": 305}],
    }
    first = build_stages(live, data / "run", record, {"path": "research_runs/rsi/framework.pdb"})
    second = build_stages(frozen, data / "run", record, {"path": "research_runs/rsi/framework.pdb"})
    assert [s.argv for s in first] == [s.argv for s in second]
    assert [s.inputs for s in first] == [s.inputs for s in second]


def test_complete_benchmark_has_explicit_long_deadline_without_expanding_legacy_jobs():
    from agent.rsi.case_plugins import antibody_allocation, maximum_case_job_seconds

    assert maximum_case_job_seconds("rfantibody.benchmark.v1") == 604800
    assert maximum_case_job_seconds("rfantibody.targets.v1") == 86400
    assert antibody_allocation("rfantibody.benchmark.v1", {"target": "ha"}) == (200, 8)
    assert antibody_allocation("rfantibody.benchmark.v1", {"target": "ha", "backbones": 1}) == (200, 8)
    assert antibody_allocation("rfantibody.targets.v1", {"backbones": 64, "sequences": 4}) == (64, 4)
