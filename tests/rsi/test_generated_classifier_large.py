import hashlib
from pathlib import Path

import pytest

from agent.rsi import generated_classifier as original
from agent.rsi import generated_classifier_large as large
from agent.rsi.model_plugins import get_model_plugin


def scope(length=4):
    return {
        "protein": "test",
        "assay": "test",
        "target": "class",
        "unit": "probability",
        "sequence_length": length,
        "class_labels": ["zero", "one"],
    }


def test_large_plugin_is_separate_and_keeps_legacy_identity():
    root = Path(large.__file__).parent
    expected = hashlib.sha256(
        b"".join(
            (root / n).read_bytes()
            for n in [
                "generated_classifier.py",
                "code_sandbox.py",
                "code_worker.py",
                "protein_references.py",
                "numeric_runtime.py",
            ]
        )
    ).hexdigest()
    assert original.implementation_hash() == expected
    plugin = get_model_plugin(large.PLUGIN_ID)
    assert plugin.id != original.PLUGIN_ID
    assert plugin.source_sha256 != expected
    assert plugin.acceptance_metric == "brier_skill_vs_training_prior"


def test_duplicate_sequence_across_validation_chunks_is_rejected(monkeypatch):
    monkeypatch.setattr(large, "MAX_PREDICT_ROWS", 2)
    rows = [{"id": str(i), "sequence": s, "target": i % 2} for i, s in enumerate(["AAAA", "AAAC", "AAAA"])]
    with pytest.raises(ValueError, match="across validation chunks"):
        large.validate_rows(rows, scope(), labels=True)


def test_residue_budget_and_probability_call_budget_fail_before_execution(monkeypatch):
    monkeypatch.setattr(large, "MAX_TOTAL_RESIDUES", 7)
    with pytest.raises(ValueError, match="residue budget"):
        large.validate_rows([{}, {}], scope(), labels=True)
    model = {"plugin_id": large.PLUGIN_ID, "scope": scope()}
    with pytest.raises(ValueError, match="60000"):
        large.predict(model, [{}] * 60001, scope())


def test_probability_view_does_not_mutate_registered_artifact():
    model = {"plugin_id": large.PLUGIN_ID, "scope": scope(), "model_state": {"p": [0.4, 0.6]}}
    viewed = large._probability_view(model, [{}], scope())
    assert viewed["plugin_id"] == original.PLUGIN_ID
    assert model["plugin_id"] == large.PLUGIN_ID
    with pytest.raises(ValueError, match="applicability"):
        large._probability_view({**model, "plugin_id": original.PLUGIN_ID}, [{}], scope())
