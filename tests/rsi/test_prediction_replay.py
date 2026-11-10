"""Host-owned same-input replay evidence and admission checks."""
import copy
import json

import pytest
from test_model_lifecycle import setup

from agent.rsi.engine.numerical_evidence import archive_assessment, verify_assessment
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json


def accepted(root):
    lc, mid, model, training, validation, reuse = setup(root)
    report = lc.assess(mid, training=training, validation=validation)
    return lc, mid, model, report, validation, reuse


def test_replay_compares_probabilities_even_when_the_predicted_class_matches():
    from agent.rsi.engine.numerical_evidence import prediction_replay_summary
    first = [{"id": "x", "prediction": 0, "probabilities": [0.6, 0.4]}]
    second = [{"id": "x", "prediction": 0, "probabilities": [0.7, 0.3]}]
    assert not prediction_replay_summary(first, second)["exact_match"]
    assert prediction_replay_summary(first, copy.deepcopy(first))["exact_match"]


def test_previous_per_row_evidence_remains_readable_without_claiming_a_replay(tmp_path):
    lc, mid, model, report, rows, _ = accepted(tmp_path)
    verified = verify_assessment(lc.root, mid, model, lc.policy, report)
    legacy = copy.deepcopy(report)
    legacy.pop("prediction_replay")
    legacy["numerical_evidence"] = archive_assessment(
        lc.root, mid, model, lc.policy, report["metrics"], verified["predictions"], rows, [], {}
    )
    old = verify_assessment(lc.root, mid, model, lc.policy, legacy)
    assert old["evidence"]["schema"] == 1
    assert "prediction_replay" not in old["evidence"]
    legacy["prediction_replay"] = report["prediction_replay"]
    with pytest.raises(ValueError, match="Legacy assessment"):
        verify_assessment(lc.root, mid, model, lc.policy, legacy)


@pytest.mark.parametrize("damage", ["file", "summary", "removed_summary"])
def test_prediction_replay_cannot_be_changed_or_removed_before_export(tmp_path, damage):
    lc, mid, model, report, _, _ = accepted(tmp_path)
    if damage == "file":
        folder = lc.root / "numerical_evidence/versions" / report["numerical_evidence"]["version_id"]
        folder.chmod(0o755)
        p = folder / "repeated_predictions.json"
        rows = json.loads(p.read_text())
        rows[0]["prediction"] += 1
        atomic_json(p, rows)
    else:
        if damage == "summary":
            report["prediction_replay"]["replayed_sha256"] = "0" * 64
        else:
            report.pop("prediction_replay")
        atomic_json(lc.root / f"assessment_{mid}.json", report)
        state = lc._state()
        state["candidates"][mid]["assessment_sha256"] = digest(report)
        atomic_json(lc.root / "state.json", state)
    with pytest.raises(ValueError):
        lc.export(mid, tmp_path / "unverified")
    assert not (tmp_path / "unverified").exists()
