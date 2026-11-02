import hashlib
import itertools
import json

import numpy as np
import pytest

from agent.rsi.classifier_transfer import LOGISTIC_REFERENCE, refit_recipe, registered_source
from agent.rsi.generated_classifier_large import fit_candidate, predict
from agent.rsi.storage import sha256_file


@pytest.fixture
def donor(tmp_path):
    run = tmp_path / "donor"
    folder = run / "arm_000"
    workspace = folder / "develop/workspace"
    workspace.mkdir(parents=True)
    artifact = folder / "controller/artifacts/versions/model_id"
    artifact.mkdir(parents=True)
    source = "def fit(sequences, targets, seed): return {'x': 1}\ndef predict(model, sequences): return [[0.5,0.5] for s in sequences]\n"
    sha = hashlib.sha256(source.encode()).hexdigest()
    model = {
        "recipe": {"source_code": source, "seed": 3},
        "source_sha256": sha,
        "scope": {"protein": "GFP"},
        "model_state": {"private_donor_weights": [99]},
    }
    (workspace / "candidate.json").write_text(json.dumps(model))
    (artifact / "model.json").write_text(json.dumps(model))
    state = {"published": ["model_id"], "candidates": {"model_id": {"status": "registered"}}}
    (folder / "controller/state.json").write_text(json.dumps(state))
    report = {
        "status": "campaign_completed",
        "arms": [{"id": 0, "status": "completed", "model_id": "model_id", "assessment": {"passed": True}}],
    }
    (run / "report.json").write_text(json.dumps(report))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "source_report_sha256": sha256_file(run / "report.json"),
                "arms": [
                    {
                        "id": 0,
                        "status": "generation_acceptance_registration_and_fresh_session_reuse_verified",
                        "source_sha256": sha,
                    }
                ],
            }
        )
    )
    return run, audit


def test_transfer_recipe_contains_source_and_new_seed_but_no_fitted_donor_weights(donor):
    source, record, model = registered_source(*donor)
    recipe = refit_recipe(source, seed=17)
    assert set(recipe) == {"source_code", "seed"}
    assert recipe["seed"] == 17 and recipe["source_code"] == source
    assert "model_state" not in recipe and model["model_state"]["private_donor_weights"] == [99]
    assert record["transfer_unit"] == "exact Python source only; target weights are fitted afresh"


@pytest.mark.parametrize("mutation", ["source", "registry", "audit", "artifact"])
def test_changed_or_unregistered_donor_is_rejected(donor, mutation):
    run, audit = donor
    if mutation == "source":
        path = run / "arm_000/develop/workspace/candidate.json"
        model = json.loads(path.read_text())
        model["recipe"]["source_code"] += "# altered"
        path.write_text(json.dumps(model))
    elif mutation == "registry":
        path = run / "arm_000/controller/state.json"
        value = json.loads(path.read_text())
        value["published"] = []
        path.write_text(json.dumps(value))
    elif mutation == "audit":
        value = json.loads(audit.read_text())
        value["source_report_sha256"] = "stale"
        audit.write_text(json.dumps(value))
    else:
        path = run / "arm_000/controller/artifacts/versions/model_id/model.json"
        value = json.loads(path.read_text())
        value["model_state"] = {}
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        registered_source(run, audit)


def test_fixed_reference_numeric_export_matches_sklearn_and_unknown_category_rule():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    tails = ["".join(x) for x in itertools.islice(itertools.product("ACDEFG", repeat=3), 20)]
    sequences = [head + tail for head in "AC" for tail in tails]
    targets = [int(s[0] == "C") for s in sequences]
    rows = [{"id": str(i), "sequence": s, "target": y} for i, (s, y) in enumerate(zip(sequences, targets, strict=True))]
    scope = {
        "protein": "synthetic",
        "assay": "numeric parity",
        "target": "binary",
        "unit": "class probability",
        "sequence_length": 4,
        "class_labels": ["zero", "one"],
    }
    model = fit_candidate(rows, scope, refit_recipe(LOGISTIC_REFERENCE))
    test = [{"id": "heldout_0", "sequence": "GGGG"}, {"id": "heldout_1", "sequence": "CFFF"}]
    actual = predict(model, test, scope)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    x = encoder.fit_transform(np.array([list(s) for s in sequences]))
    fitted = LogisticRegression(C=1.0, solver="liblinear", max_iter=500, tol=1e-4, random_state=314159).fit(x, targets)
    expected = fitted.predict_proba(encoder.transform(np.array([list(r["sequence"]) for r in test])))
    assert np.allclose([r["probabilities"] for r in actual], expected, rtol=0, atol=1e-12)
