import copy

import pytest

from agent.rsi.yeast_population_controls import controls, verify_metrics


def example():
    scope = {
        "protein": "YAL001C",
        "assay": "population_protein_expression",
        "target": "qc_relative_protein_abundance",
        "unit": "log2_ratio_to_pooled_qc",
        "context": {"organism": "Saccharomyces cerevisiae"},
        "grouping": {"level": "genetic_population", "evidence_sha256": "0" * 64},
        "covariates": [
            {"name": name, "unit": "test", "minimum": 0.0, "maximum": 100.0}
            for name in ["author_log2_tpm", "od_rna", "od_protein", "ploidy"]
        ],
    }
    rows = [
        {
            "id": f"r{i}",
            "protein": "YAL001C",
            "group": "fit" if i < 30 else "scoring",
            "values": [float(i), 0.5, 1.0, 2.0],
            "target": 2.5 * i - 1,
        }
        for i in range(40)
    ]
    return scope, rows[:30], rows[30:]


def test_transcript_linear_control_recovers_known_function():
    scope, fit, score = example()
    outcomes = controls(fit, score, scope)
    assert len(outcomes) == 6
    verify_metrics(fit, score, outcomes)
    transcript = next(o for o in outcomes if o["method"] == "transcript_only_ols")
    assert transcript["metrics"]["rmse"] < 1e-10
    corrupted = copy.deepcopy(outcomes)
    corrupted[0]["metrics"]["rmse"] += 1
    with pytest.raises(ValueError, match="independent arithmetic"):
        verify_metrics(fit, score, corrupted)


def test_scoring_targets_never_affect_fitted_predictions():
    scope, fit, score = example()
    first = controls(fit, score, scope)
    changed = [{**r, "target": -12345.0} for r in score]
    second = controls(fit, changed, scope)
    assert [o["predictions"] for o in first] == [o["predictions"] for o in second]
    with pytest.raises(ValueError, match="independent"):
        controls(fit, [{**r, "group": "fit"} for r in score], scope)
