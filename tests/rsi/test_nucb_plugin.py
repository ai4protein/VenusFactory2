"""Fail-closed contracts for a complete declared NucB computational allocation."""

import copy
from pathlib import Path

import pytest

from agent.rsi import nucb_reproduction as n
from agent.rsi.case_evidence import canonical_claims
from agent.rsi.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.paper_campaign import campaign_cases, paper_case


def report():
    return {
        "status": "completed",
        "source_commit": n.COMMIT,
        "execution_origin": "new_raw_count_and_analysis_computation",
        "limits": list(n.LIMITS),
        "evidence": {
            "generation_rows": dict(n.ROWS),
            "processed_and_fiducial_tables": 16,
            "unique_landscape_sequences": 55760,
            "hit_rate_rows": 44,
            "purified_timepoints": 8428,
            "initial_rate_fits": 28,
            "reference_tables_passed": 17,
            "independent_analysis_replay_passed": True,
        },
    }


def test_nucb_catalog_and_fixed_allocation(tmp_path):
    p = paper_case("nucb_raw_analysis")
    assert p["paper_id"] == "nucb" and p["reproduction_level"] == "partial_computational"
    assert p["allocation"]["parameters"] == {} and p["excluded_steps"]
    assert len(campaign_cases("eight_papers")) == 8
    assert len(campaign_cases("seven_papers")) == 7
    assert [p["paper_id"] for p in campaign_cases("nuclease")] == ["nucb"]
    plugin = next(p for p in list_case_plugins() if p["id"] == "nucb.raw_analysis.v1")
    assert plugin["parameters"]["properties"] == {}
    assert plugin["parameters"]["additionalProperties"] is False
    assert resource_for(plugin["id"], {}) == "cpu"
    argv = invocation(
        plugin["id"], {}, root=tmp_path, output=tmp_path / "research_runs/rsi/new", controller_python="python"
    )
    assert "nuclease/bin/python" in argv[0] and argv[-2] == "--output"
    for parameters in [{"generations": ["g4"]}, {"source": "/tmp/other"}, {"operation": "verify_existing"}]:
        with pytest.raises(ValueError):
            invocation(
                plugin["id"],
                parameters,
                root=tmp_path,
                output=tmp_path / "research_runs/rsi/new",
                controller_python="python",
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation_rows", {"g4": 15404}),
        ("processed_and_fiducial_tables", 15),
        ("unique_landscape_sequences", 55759),
        ("hit_rate_rows", 43),
        ("purified_timepoints", 8427),
        ("initial_rate_fits", 27),
        ("reference_tables_passed", 16),
        ("independent_analysis_replay_passed", False),
    ],
)
def test_claims_reject_partial_or_failed_evidence(field, value):
    r = report()
    r["evidence"][field] = value
    with pytest.raises(ValueError):
        canonical_claims("nucb.raw_analysis.v1", r)


def test_claims_preserve_partial_scope_and_computation_origin():
    r = report()
    claims = canonical_claims("nucb.raw_analysis.v1", r)
    assert not claims["full_paper_reproduced"] and not claims["wet_lab_validation"]
    assert not claims["new_model_developed"]
    for key, value in [
        ("status", "running"),
        ("source_commit", "other"),
        ("limits", []),
        ("execution_origin", "verified_existing"),
    ]:
        bad = copy.deepcopy(r)
        bad[key] = value
        with pytest.raises(ValueError):
            canonical_claims("nucb.raw_analysis.v1", bad)


def test_artifacts_reject_traversal_symlinks_and_changed_bytes(tmp_path):
    inside = tmp_path / "evidence"
    inside.mkdir()
    (inside / "data").write_text("original")
    sha = n.sha256_file(inside / "data")
    assert n.checked(inside, "data", sha) == inside / "data"
    for name in ["../data", str((inside / "data").resolve())]:
        with pytest.raises(ValueError):
            n.safe_file(inside, name)
    (inside / "link").symlink_to(inside / "data")
    with pytest.raises(ValueError):
        n.safe_file(inside, "link")
    (inside / "data").write_text("modified")
    with pytest.raises(ValueError):
        n.checked(inside, "data", sha)


def test_native_validator_is_required_at_broker_boundary(monkeypatch, tmp_path):
    from agent.rsi.case_jobs import verify_job_report

    calls = []

    def reject(*args):
        calls.append(args)
        raise ValueError("forged successful report")

    monkeypatch.setattr(n, "verify_artifacts", reject)
    with pytest.raises(ValueError, match="forged"):
        verify_job_report(
            tmp_path, {"plugin_id": "nucb.raw_analysis.v1", "output": str(tmp_path), "parameters": {}}, report()
        )
    assert len(calls) == 1


def test_table_comparison_rejects_forged_numeric_result(tmp_path):
    root = Path(__file__).resolve().parents[2]
    actual, ref = tmp_path / "actual.csv", tmp_path / "reference.csv"
    ref.write_text('mutations,value\n"[]",1.0\n')
    actual.write_bytes(ref.read_bytes())
    assert n._compare_tables(root, actual, ref)["passed"]
    actual.write_text('mutations,value\n"[]",2.0\n')
    assert not n._compare_tables(root, actual, ref)["passed"]


def test_direct_runner_rejects_parent_escape_before_creating_output(tmp_path, monkeypatch):
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("nucb_runner_fixture", root / "scripts/rsi/run_nucb_case.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        runner.run(tmp_path / "research_runs/rsi/../../outside")
    assert not (tmp_path / "outside").exists()
