"""Paired allocations represent one selected task, without admitting duplicate arms."""

import copy

import pytest
from test_yeast_population_controls import example

from agent.rsi import context_feedback_profiles as feedback
from agent.rsi.yeast_population_evaluation import verify_selected_tasks


def design():
    scope, _, _ = example()
    allocations = {}
    pairs = []
    for task in ("first", "second"):
        left, right = task + "_group", task + "_domain"
        for name, policy in ((left, feedback.POLICY), (right, feedback.EXPRESSION_POLICY)):
            allocations[name] = {
                "id": name, "task": {"id": task, "plugin_id": "generated_context.v1", "scope": scope},
                feedback.FIELD: policy,
            }
        pairs.append({"id": task, "left": left, "right": right, "factor": feedback.FIELD})
    return {"allocations": allocations, "pairs": pairs}, {"selection": {"selected_tasks": ["first", "second"]}}


def test_two_arms_count_as_one_selected_task():
    definition, protocol = design()
    verify_selected_tasks(definition, protocol)
    protocol["selection"]["selected_tasks"].reverse()
    verify_selected_tasks(definition, protocol)


@pytest.mark.parametrize("attack", ["unpaired", "repeated_pair", "duplicate_selection", "missing_selection", "different_task", "different_data"])
def test_paired_membership_rejects_changed_design(attack):
    definition, protocol = copy.deepcopy(design())
    if attack == "unpaired":
        definition["pairs"].pop()
    elif attack == "repeated_pair":
        definition["pairs"].append({**definition["pairs"][0], "id": "duplicate"})
    elif attack == "duplicate_selection":
        protocol["selection"]["selected_tasks"].append("first")
    elif attack == "missing_selection":
        protocol["selection"]["selected_tasks"].pop()
    elif attack == "different_task":
        definition["allocations"]["first_domain"]["task"] = {
            **definition["allocations"]["first_domain"]["task"], "id": "third"
        }
    else:
        definition["allocations"]["first_domain"]["task"] = {
            **definition["allocations"]["first_domain"]["task"], "data": "changed"
        }
    with pytest.raises(ValueError):
        verify_selected_tasks(definition, protocol)


def test_original_single_arm_protocol_remains_exact():
    definition, protocol = design()
    definition["pairs"] = []
    definition["allocations"] = {k: v for k, v in definition["allocations"].items() if k.endswith("_group")}
    verify_selected_tasks(definition, protocol)
    definition["allocations"]["duplicate"] = definition["allocations"]["first_group"]
    with pytest.raises(ValueError, match="membership"):
        verify_selected_tasks(definition, protocol)
