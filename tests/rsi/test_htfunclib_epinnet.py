import numpy as np
import pandas as pd
import pytest

from agent.rsi import htfunclib_epinnet as case


def test_sequence_overlap_is_distinct_from_row_overlap_and_label_conflict():
    data = pd.DataFrame({"sequence": ["AAA", "BBB", "AAA", "CCC"], "y": [1, 0, 0, 1]})
    result = case.overlap_diagnostics(data, {"train": [0, 1], "test": [2, 3]})
    assert result["shared_sequence_identities"] == 1
    assert result["shared_source_row_ids"] == 0
    assert result["test_records_with_training_sequence"] == 1
    assert result["test_records_with_unseen_sequence"] == 1
    assert result["labeled_sequences_with_conflicting_energy_labels"] == 1


def test_native_probability_replay_and_schema_reject_corruption():
    weights = {"w0": np.zeros((91, 26)), "b0": np.zeros(26), "w1": np.zeros((26, 1)), "b1": np.array([0.0])}
    np.testing.assert_array_equal(case.forward(weights, np.eye(91)), np.full((91, 2), 0.5))
    weights["b1"][0] = np.log(3.0)
    np.testing.assert_allclose(case.forward(weights, np.zeros((2, 91)))[:, 1], 0.75)
    weights["w0"][0, 0] = np.nan
    with pytest.raises(ValueError, match="weight shape or values"):
        case.forward(weights, np.zeros((1, 91)))


def test_fixed_split_hash_preserves_row_order():
    assert case.order_digest([1, 2, 3]) != case.order_digest([3, 2, 1])
    assert case.order_digest([1, 2]) != case.order_digest([1, 2, 2])


@pytest.mark.parametrize(
    "claim",
    [
        {"status": "running"},
        {"status": "completed", "agent_generated_model": True},
        {"status": "completed", "new_rosetta_jobs_executed": 20000},
    ],
)
def test_incomplete_or_overclaimed_epinnet_rejected(claim):
    with pytest.raises(ValueError, match="Incomplete or unsupported"):
        case.canonical_epinnet(claim)


def test_epinnet_plugin_rejects_test_tuning_or_partial_scope(tmp_path):
    from agent.rsi.case_plugins import invocation, list_case_plugins, resource_for

    def call(parameters):
        return invocation(
            case.CASE_ID,
            parameters,
            root=tmp_path,
            output=tmp_path / "research_runs/rsi/run",
            controller_python=tmp_path / "python",
        )

    argv = call({})
    assert argv[0].endswith("envs/htfunclib/bin/python")
    assert argv[2].endswith("run_htfunclib_epinnet.py")
    assert resource_for(case.CASE_ID, {}) == "cpu"
    assert next(p for p in list_case_plugins() if p["id"] == case.CASE_ID)["parameters"]["properties"] == {}
    for parameters in [{"top_k": 41}, {"seed": 7}, {"max_iter": 3}, {"source": "/tmp/source"}]:
        with pytest.raises(ValueError, match="complete pinned"):
            call(parameters)
