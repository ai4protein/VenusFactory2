import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
from run_experience_campaign import allocation_order, task_prompt


def test_all_mode_condition_case_allocations_are_retained_and_counterordered():
    allocations = allocation_order()
    keys = {(a["case"], a["condition"], a["mode"]) for a in allocations}
    assert len(allocations) == len(keys) == 12
    first = [(a["condition"], a["mode"]) for a in allocations[:6]]
    second = [(a["condition"], a["mode"]) for a in allocations[6:]]
    assert first == second[::-1]


def test_task_message_has_no_condition_specific_prompt_advantage():
    scope = {"protein": "test", "target": "fitness", "assay": "measured", "unit": "AU"}
    prompt = task_prompt(scope)
    assert "read_model_experience" in prompt
    assert "No target acceptance or later labels" in prompt
    assert "two source versions" in prompt
