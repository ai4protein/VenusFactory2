"""Versioned MCP formatting preserves complete feedback and legacy envelopes."""

import json

import pytest
from test_shared_model_display import restored_inspection

from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.shared_model_display import (
    LEGACY_POLICY,
    MAX_DISPLAY_CHARACTERS,
    POLICY,
    VECTOR_POLICY,
    envelope,
    wire_text,
)


def dense_feedback():
    return {
        "source_sha256": "a" * 64,
        "attempts_remaining": 1,
        "status": "tested_not_published",
        "results": [
            {
                "task_id": f"target_{i}",
                "ledger_sha256": str(i),
                "result": {
                    "training_only_feedback": {
                        "rmse": i / 11,
                        "control_table": [[i / 13 + j / 17 + k / 19 for k in range(8)] for j in range(11)],
                    }
                },
            }
            for i in range(32)
        ],
    }


def test_dense_32_target_feedback_fits_actual_v3_wire_without_losing_values():
    full = dense_feedback()
    before = digest(full)
    packet = envelope("evaluate_model_code", full)
    text = wire_text(packet)
    assert len(text) <= MAX_DISPLAY_CHARACTERS < len(json.dumps(packet, indent=2))
    assert json.loads(text) == packet and digest(full) == before
    assert [row["task_id"] for row in packet["content"]["results"]] == [row["task_id"] for row in full["results"]]
    for row, original in zip(packet["content"]["results"], full["results"]):
        assert row["result"]["training_only_feedback"] == original["result"]["training_only_feedback"]
    with pytest.raises(ValueError, match="90000-character"):
        envelope("evaluate_model_code", full, policy=VECTOR_POLICY)


@pytest.mark.parametrize("policy", [VECTOR_POLICY, POLICY])
def test_vector_pool_survives_policy_upgrade(policy):
    rows = [
        {
            "task_id": f"p_{i}",
            "contract": {"training": {"examples": [{"id": str(i), "values": [i, 0.3], "target": i / 7}]}},
        }
        for i in range(3)
    ]
    assert digest(restored_inspection(envelope("inspect_model_code_task", {"targets": rows}, policy=policy))) == digest(
        rows
    )


@pytest.mark.parametrize("policy", [LEGACY_POLICY, VECTOR_POLICY])
def test_legacy_wire_format_remains_pretty_json(policy):
    packet = envelope("inspect_model_code_task", {"targets": []}, policy=policy)
    assert wire_text(packet) == json.dumps(packet, ensure_ascii=True, indent=2)


def test_v3_still_rejects_payload_over_real_wire_budget():
    full = {"targets": [{"task_id": "p", "contract": {"diagnostic": "x" * MAX_DISPLAY_CHARACTERS}}]}
    with pytest.raises(ValueError, match="90000-character"):
        envelope("inspect_model_code_task", full)
