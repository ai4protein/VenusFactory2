"""CLEAN contract tests use synthetic evidence, never scientific measurements."""

import copy
import json

import pytest

from agent.rsi import clean_annotation as c
from agent.rsi.case_evidence import canonical_claims
from agent.rsi.case_jobs import verify_job_report
from agent.rsi.case_plugins import invocation, list_case_plugins, resource_for
from agent.rsi.paper_campaign import campaign_cases, paper_case


def report():
    results, replay = {}, {}
    for model in c.MODELS:
        for name, (_, n, _) in c.COHORTS.items():
            metrics = c.metrics([["A"]] * n, [["A"]] * n)
            key = model + "." + name
            results[key] = {"rows": n, "full_union_metrics": metrics}
            replay[key] = {"rows": n, "metrics": metrics, "max_distance_error": 1e-6}
    return {
        "status": "completed",
        "allocation": copy.deepcopy(c.ALLOCATION),
        "completed_embedding_records": 578,
        "source_commit": c.COMMIT,
        "asset_manifest_sha256": c.MANIFEST_SHA256,
        "execution_origin": "new_sequence_embedding_and_author_inference",
        "full_paper_reproduced": False,
        "original_clean_training_reproduced": False,
        "new_model_developed": False,
        "results": results,
        "independent_replay": {
            "status": "passed",
            "source_commit": c.COMMIT,
            "transformer_rerun_by_auditor": False,
            "allocations": replay,
        },
    }


def test_clean_scope_and_new_computation_only(tmp_path):
    paper = paper_case("clean_all_cohorts")
    assert paper["paper_id"] == "clean" and paper["excluded_steps"]
    assert paper["reproduction_level"] == "partial_computational"
    assert [p["paper_id"] for p in campaign_cases("enzyme_annotation")] == ["clean"]
    # Published previous suites retain their exact allocations.
    assert len(campaign_cases("eight_papers")) == 8
    plugin = next(p for p in list_case_plugins() if p["id"] == "clean.annotation.v1")
    assert plugin["parameters"]["properties"] == {}
    assert plugin["parameters"]["additionalProperties"] is False
    assert resource_for(plugin["id"], {}) == "cpu"
    kwargs = {"root": tmp_path, "output": tmp_path / "research_runs/rsi/new", "controller_python": "python"}
    argv = invocation(plugin["id"], {}, **kwargs)
    assert argv[0].endswith("envs/clean/bin/python")
    assert argv[2].endswith("run_clean_annotation.py")
    for parameters in [
        {"cohort": "price"},
        {"heads": ["split70"]},
        {"operation": "verify_existing"},
        {"source": "/tmp"},
    ]:
        with pytest.raises(ValueError, match="all 578"):
            invocation(plugin["id"], parameters, **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("completed_embedding_records", 577),
        ("source_commit", "other"),
        ("asset_manifest_sha256", "other"),
        ("execution_origin", "cached"),
        ("full_paper_reproduced", True),
        ("new_model_developed", True),
        ("original_clean_training_reproduced", True),
    ],
)
def test_rejects_incomplete_or_overclaimed_evidence(field, value):
    r = report()
    r[field] = value
    with pytest.raises(ValueError):
        canonical_claims("clean.annotation.v1", r)


def test_claims_bind_complete_independent_replay_and_partial_scope():
    r = report()
    claims = canonical_claims("clean.annotation.v1", r)
    assert claims["prediction_records"] == 1156
    assert not claims["full_paper_reproduced"] and not claims["wet_lab_validation"]
    assert not claims["new_model_developed"] and not claims["recursive_improvement_established"]
    r["independent_replay"]["allocations"].pop("split70.price")
    with pytest.raises(ValueError, match="independent replay"):
        canonical_claims("clean.annotation.v1", r)
    r = report()
    r["results"]["split70.price"]["full_union_metrics"]["micro_f1"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        canonical_claims("clean.annotation.v1", r)


def test_native_numerical_gate_cannot_be_skipped(monkeypatch, tmp_path):
    calls = []

    def reject(*args):
        calls.append(args)
        raise ValueError("forged completed report")

    monkeypatch.setattr(c, "verify_artifacts", reject)
    with pytest.raises(ValueError, match="forged"):
        verify_job_report(
            tmp_path, {"plugin_id": "clean.annotation.v1", "parameters": {}, "output": str(tmp_path)}, report()
        )
    assert len(calls) == 1


def test_artifact_omission_and_protocol_change_fail_before_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "verify_assets", lambda _: {"encoder": {}})
    r = report()
    r["artifact_sha256"] = {}
    with pytest.raises(ValueError, match="artifact coverage"):
        c.verify_artifacts(r, tmp_path, tmp_path)
    for name in c.artifact_names():
        (tmp_path / name).write_text("{}")
    r["artifact_sha256"] = {n: c.sha(tmp_path / n) for n in c.artifact_names()}
    with pytest.raises(ValueError, match="protocol"):
        c.verify_artifacts(r, tmp_path, tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps({"allocation": c.ALLOCATION}))
    with pytest.raises(ValueError, match="output changed"):
        c.verify_artifacts(r, tmp_path, tmp_path)


def test_runner_rejects_escape_before_making_directory(tmp_path, monkeypatch):
    import importlib.util
    import os
    from pathlib import Path

    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("clean_runner_fixture", root / "scripts/rsi/run_clean_annotation.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        runner.run(tmp_path / "research_runs/rsi/../../outside")
    assert not (tmp_path / "outside").exists()
