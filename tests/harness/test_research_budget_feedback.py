"""Host scheduling hints must not become persistent research evidence."""
import copy

import pytest

from agent.harness.research_execution_feedback import (
    _failure_message,
    feedback_state,
    repair_source_context,
)

NORMAL = "Save a checkpoint when the current step is ready."
URGENT = "Save submit_research_checkpoint now with actual artifacts and only completed steps."


def footer(remaining):
    instruction = URGENT if remaining <= 2 else NORMAL
    return (f"Execution budget: {remaining} tool calls remain; 8 read-only discovery calls remain. "
            f"{instruction} Checkpoint saves have a separate allowance: 2 remain.")


@pytest.mark.parametrize("remaining", [0, 1, 2, 4])
@pytest.mark.parametrize("separator", ["", "\n"])
def test_complete_host_footer_is_not_a_persistent_error(tmp_path, remaining, separator):
    error = "ValueError: model interface is missing"
    assert _failure_message(error + separator + footer(remaining), tmp_path) == error


@pytest.mark.parametrize("text", [
    "Execution budget: 0 tool calls remain; 8 read-only discovery calls remain.",
    footer(0) + " More error details follow.",
    footer(0).replace(URGENT, "Save a different checkpoint now."),
    footer(0).replace(URGENT, NORMAL),
    footer(4).replace(NORMAL, URGENT),
    footer(0).replace("allowance: 2 remain.", "allowance: unknown remain."),
    footer(0).replace("budget: 0", "budget: -1"),
])
def test_similar_or_nonterminal_text_remains_evidence(tmp_path, text):
    diagnostic = "ValueError: " + text
    assert _failure_message(diagnostic, tmp_path) == diagnostic


def test_recorded_r3_zero_budget_error_reaches_next_repair_without_stale_allowance(tmp_path):
    # Exact diagnostic and concatenated footer from 40183 cycle_004/events[85].
    # Attempt identity is a public receipt reference; no measurements are copied.
    error = (
        "TrainingAttemptError: Training attempt 0a80e87ee82942df8e79384cce4ba4c6 failed: "
        "Model source check failed: Implement module-level fit(training, seed). "
        "The module has no callable fit entrypoint. It may call a helper class, but must return "
        "the fitted estimator or one dict containing the learner and its fitted transformations. "
        "Read rows from training['train'].\n"
        "Implement module-level predict(model, inputs). The module has no callable predict entrypoint. "
        "Recover the learner and fitted transformations from the single model argument. "
        "inputs is one query dict. Return a dict with the declared prediction key "
        "(scalar or multilabel probabilities); do not refit on query targets."
    )
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/model.py").write_text("# Diagnostic fixture, never executed.\n")
    events = [
        {"kind": "tool_call_start", "tool_call_id": "r3-failure",
         "tool_name": "mcp__venusfactory__train_research_model",
         "args": {"source_file": "src/model.py"}},
        {"kind": "tool_result", "tool_call_id": "r3-failure", "is_error": True,
         "output": error + footer(0)},
    ]
    original = copy.deepcopy(events)
    current = feedback_state(events, workspace)
    assert current["pending"][0]["error"] == error
    assert current["pending"][0]["training_attempt_id"] == "0a80e87ee82942df8e79384cce4ba4c6"
    following = feedback_state([], workspace, current)
    repair = repair_source_context(workspace, following)
    assert repair["error"] == error
    assert "Execution budget:" not in str(repair)
    assert events == original  # Immutable historical events retain the original receipt text.
    legacy = copy.deepcopy(current)
    legacy["pending"][0]["error"] = error + footer(0)
    assert feedback_state([], workspace, legacy)["pending"][0]["error"] == error
