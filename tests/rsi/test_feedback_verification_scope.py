"""Reuse exact feedback replays within an audit, with fresh source checks."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from test_context_baseline_refit import fixture

from agent.rsi.engine import context_feedback_dispatch as dispatch


@pytest.fixture(params=[1, 2, 3])
def sample(request, monkeypatch):
    version = request.param
    scope, training, scoring = fixture("numeric")
    if version > 1:
        scope["interface_version"] = f"generated_context.v{version}"
    rows = training + scoring
    suffix = "" if version == 1 else f"_v{version}"
    module = importlib.import_module(f"agent.rsi.engine.context{suffix}_feedback")
    api = importlib.import_module(f"agent.rsi.engine.generated_context{suffix}")
    _, heldout = api.development_split(rows)
    feedback = {
        "rows": len(heldout),
        "independent_groups": len({row["group"] for row in heldout}),
        "rmse": 1.0,
    }
    feedback["baseline_comparison"] = module.comparison(rows, scope, feedback)
    task = {"plugin_id": api.PLUGIN_ID, "scope": scope, dispatch.FIELD: module.POLICY}
    original = module.comparison
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "comparison", count)
    return task, rows, feedback, module, calls


def test_nested_audit_reuses_exact_feedback_but_next_audit_replays(sample):
    task, rows, feedback, _, calls = sample
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
        with dispatch.verification_session():
            dispatch.verify_feedback(deepcopy(task), deepcopy(rows), deepcopy(feedback))
        assert len(calls) == 1
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
    dispatch.verify_feedback(task, rows, feedback)
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["feedback", "rows", "policy"])
def test_warm_replay_cannot_accept_changed_feedback_data_or_policy(sample, change):
    task, rows, feedback, _, _ = sample
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
        if change == "feedback":
            feedback["baseline_comparison"]["controls"][0]["metrics"]["rmse"] += 1
        elif change == "rows":
            rows[0]["target"] += 100
        else:
            task[dispatch.FIELD] = "none"
        with pytest.raises(ValueError, match="numerical replay|Undeclared"):
            dispatch.verify_feedback(task, rows, feedback)


def test_warm_replay_rechecks_source_bytes(sample, tmp_path, monkeypatch):
    task, rows, feedback, module, calls = sample
    source_root = Path(dispatch.__file__).parent
    for name in feedback["baseline_comparison"]["implementation"]:
        (tmp_path / name).write_bytes((source_root / name).read_bytes())
    monkeypatch.setattr(dispatch, "__file__", str(tmp_path / "context_feedback_dispatch.py"))
    controls = next(value for name, value in vars(module).items() if name.endswith("_controls"))
    original_path = Path(controls.__file__)
    copied = tmp_path / original_path.name
    monkeypatch.setattr(controls, "__file__", str(copied))
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
        dispatch.verify_feedback(task, rows, feedback)
        assert len(calls) == 1
        copied.write_bytes(copied.read_bytes() + b"\n# changed source\n")
        with pytest.raises(ValueError, match="numerical replay"):
            dispatch.verify_feedback(task, rows, feedback)
    assert len(calls) == 2


def test_failed_replay_is_not_cached_and_scope_cleans_up_on_error(sample):
    task, rows, feedback, _, calls = sample
    bad = deepcopy(feedback)
    bad["baseline_comparison"]["candidate_rmse"] += 1
    with dispatch.verification_session():
        for _ in range(2):
            with pytest.raises(ValueError, match="numerical replay"):
                dispatch.verify_feedback(task, rows, bad)
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="interrupted"):
        with dispatch.verification_session():
            dispatch.verify_feedback(task, rows, feedback)
            raise RuntimeError("interrupted")
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
    assert len(calls) == 4


def test_bounded_cache_evicts_old_results_without_skipping_verification(sample, monkeypatch):
    task, rows, feedback, _, calls = sample
    monkeypatch.setattr(dispatch, "_MAX_VERIFIED_FEEDBACK", 2)
    with dispatch.verification_session():
        for marker in range(3):
            dispatch.verify_feedback({**task, "audit_fixture": marker}, rows, feedback)
        dispatch.verify_feedback({**task, "audit_fixture": 0}, rows, feedback)
    assert len(calls) == 4


def test_concurrent_audits_do_not_share_verification_state(sample):
    task, rows, feedback, _, calls = sample

    def audit(_):
        with dispatch.verification_session():
            dispatch.verify_feedback(task, rows, feedback)
            dispatch.verify_feedback(task, rows, feedback)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(audit, range(2)))
    assert len(calls) == 2


@pytest.mark.parametrize("packet", [{}, {"baseline_comparison": None}])
def test_undeclared_feedback_retains_existing_none_semantics(packet):
    task = {"plugin_id": "generated_context.v2", dispatch.FIELD: "none"}
    with dispatch.verification_session():
        dispatch.verify_feedback(task, [], packet)
        dispatch.verify_feedback(task, [], packet)


@pytest.mark.parametrize("change", ["tuple_rows", "tuple_values"])
def test_serialization_cannot_bypass_original_input_types(sample, change):
    task, rows, feedback, module, _ = sample
    altered = deepcopy(rows)
    if change == "tuple_rows":
        altered = tuple(altered)
    else:
        altered[0]["values"] = tuple(altered[0]["values"])
    with pytest.raises(ValueError):
        module.verify_feedback(task, altered, feedback)
    with dispatch.verification_session():
        dispatch.verify_feedback(task, rows, feedback)
        with pytest.raises(ValueError):
            dispatch.verify_feedback(task, altered, feedback)
