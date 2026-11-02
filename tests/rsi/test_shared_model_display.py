"""Compact displays preserve task coverage, full contracts and feedback values."""

import copy
import json

import pytest
from test_shared_history_projection import feedback

from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.shared_model_display import LEGACY_POLICY, MAX_DISPLAY_CHARACTERS, POLICY, envelope


def merged(common, specific):
    result = copy.deepcopy(common)
    for key, value in specific.items():
        result[key] = (
            merged(result[key], value)
            if key in result and isinstance(result[key], dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def test_full_inspection_contracts_round_trip_without_repeated_instructions():
    rows = [
        {
            "task_id": f"plant_{index}",
            "contract": {
                "abi": {"fit": "shared fit contract" * 100},
                "scope": {"protein": f"plant_{index}", "covariates": ["value"]},
                "training": {"rows": 82, "examples": [{"values": [index], "target": index / 3}]},
                "empty": {},
            },
        }
        for index in range(32)
    ]
    full = {"schema": "shared_model_program.v1", "targets": rows}
    before = copy.deepcopy(full)
    packet = envelope("inspect_model_code_task", full, policy=LEGACY_POLICY)
    view = packet["content"]
    assert [merged(view["shared_defaults"], row) for row in view["targets"]] == rows
    assert [row["task_id"] for row in view["targets"]] == [row["task_id"] for row in rows]
    assert len(json.dumps(packet, indent=2)) < len(json.dumps(full, indent=2)) / 2
    assert full == before


def test_shared_defaults_do_not_conflate_json_value_types():
    rows = [{"task_id": f"plant_{index}", "contract": {"number": value}} for index, value in enumerate([1, 1.0, True])]
    full = {"targets": rows}
    view = envelope("inspect_model_code_task", full)["content"]
    restored = [merged(view["shared_defaults"], row) for row in view["targets"]]
    assert digest(restored) == digest(rows)


def test_evaluation_display_keeps_every_target_and_numeric_feedback():
    full = {
        "schema": "shared_model_program.v1",
        "source_sha256": "a" * 64,
        "attempts_remaining": 1,
        "status": "tested_not_published",
        "results": [
            {"task_id": f"plant_{index}", "ledger_sha256": str(index), "result": feedback()} for index in range(32)
        ],
    }
    packet = envelope("evaluate_model_code", full)
    view = packet["content"]
    restored = [merged(view["shared_defaults"], row) for row in view["results"]]
    assert len(restored) == 32 and packet["source_sha256"] == full["source_sha256"]
    assert all(row["result"]["training_only_feedback"]["rmse"] == 1.23456789012345 for row in restored)
    assert all(
        row["result"]["training_only_feedback"]["baseline_comparison"]["rows"][-1][-1]
        == {"error": "fixture diagnostic"}
        for row in restored
    )
    assert len(json.dumps(packet, indent=2)) <= MAX_DISPLAY_CHARACTERS


def test_discovery_keeps_independent_models_and_scopes():
    rows = [
        {
            "task_id": f"plant_{index}",
            "result": {
                "models": [
                    {
                        "model_id": str(index),
                        "scope": {"protein": f"plant_{index}", "covariates": ["value"]},
                        "assessment": {"rmse": index / 7},
                    }
                ]
            },
        }
        for index in range(32)
    ]
    packet = envelope("discover_generated_models", {"schema": "shared_model_program.v1", "targets": rows})
    view = packet["content"]
    restored = [merged(view["shared_defaults"], row) for row in view["targets"]]
    assert [row["model"] for row in restored] == [row["result"]["models"][0] for row in rows]


def test_oversized_display_is_rejected_without_truncation():
    full = {"targets": [{"task_id": "plant_0", "contract": {"sequence": "A" * MAX_DISPLAY_CHARACTERS}}]}
    with pytest.raises(ValueError, match="90000-character budget"):
        envelope("inspect_model_code_task", full)
    assert len(full["targets"][0]["contract"]["sequence"]) == MAX_DISPLAY_CHARACTERS
    with pytest.raises(ValueError, match="Unsupported"):
        envelope("unbound_operation", {})


def restored_inspection(packet):
    view = packet["content"]
    rows = [merged(view["shared_defaults"], row) for row in view["targets"]]
    for row in rows:
        training = row.get("contract", {}).get("training", {})
        if "example_rows" in training and "examples" not in training:
            examples = training.pop("example_rows")
            for example in examples:
                index = example.pop("values_ref")
                example["values"] = copy.deepcopy(view["training_example_values"][index])
            training["examples"] = examples
    return rows


def test_mixed_32_protein_examples_reconstruct_every_value_and_fit_display_limit():
    rows = []
    for index in range(32):
        examples = []
        for group in range(3):
            examples.append(
                {
                    "id": f"public_protein_{index}_group_{group}",
                    "protein": f"protein_{index}",
                    "group": f"published_group_{group}",
                    "values": [(index % 2 + group + column / 7) / 3.141592653589793 for column in range(19)],
                    "target": index / 11 + group / 13,
                }
            )
        rows.append(
            {
                "task_id": f"protein_{index}",
                "contract": {"training": {"rows": 82, "examples": examples}, "condition": index % 2},
            }
        )
    full = {"schema": "shared_model_program.v1", "targets": rows}
    before = digest(full)
    packet = envelope("inspect_model_code_task", full)
    assert packet["schema"] == POLICY
    assert len(packet["content"]["training_example_values"]) == 4
    assert digest(restored_inspection(packet)) == digest(rows)
    assert digest(full) == before
    assert len(json.dumps(packet, ensure_ascii=True, indent=2)) < 50000
    with pytest.raises(ValueError, match="90000-character"):
        envelope("inspect_model_code_task", full, policy=LEGACY_POLICY)


def test_vector_pool_does_not_merge_integer_float_and_boolean_or_reorder_rows():
    rows = [
        {
            "task_id": f"plant_{i}",
            "contract": {"training": {"examples": [{"id": str(i), "values": [value], "target": value}]}},
        }
        for i, value in enumerate([1, 1.0, True, 1])
    ]
    packet = envelope("inspect_model_code_task", {"targets": rows})
    assert len(packet["content"]["training_example_values"]) == 3
    assert digest(restored_inspection(packet)) == digest(rows)


def test_noncontext_examples_and_reserved_field_collisions_are_not_rewritten():
    rows = [
        {"task_id": "sequence", "contract": {"training": {"examples": [{"sequence": "AAAA", "target": 1}]}}},
        {"task_id": "empty", "contract": {"training": {"examples": []}}},
        {"task_id": "reserved", "contract": {"training": {"examples": [{"values": [1], "values_ref": "literal"}]}}},
        {
            "task_id": "reserved_training",
            "contract": {"training": {"examples": [{"values": [1]}], "example_rows": "literal"}},
        },
    ]
    packet = envelope("inspect_model_code_task", {"targets": rows})
    assert packet["content"]["training_example_values"] == []
    assert digest(restored_inspection(packet)) == digest(rows)
    with pytest.raises(ValueError, match="Unsupported shared tool presentation policy"):
        envelope("inspect_model_code_task", {"targets": rows}, policy="unrecognized")
