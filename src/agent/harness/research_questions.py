"""Validate explicit user questions without turning template text into a pause."""
from __future__ import annotations

# These names are tool protocol fields, not requests for unavailable user data.
_INTERNAL_FIELDS = frozenset({
    "evaluation_file", "source_file", "parent_experiment_id", "prediction_key",
    "review_spec", "reproduction_script", "fitted_files",
})


def validate_input_questions(questions):
    message = (
        "needs_input must contain concrete questions about unavailable user data. "
        "Do not copy template text. Use [] when no user input is needed; "
        "code errors and unfinished work require another implementation step."
    )
    if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
        raise ValueError(message)
    for question in questions:
        normalized = " ".join(question.casefold().split())
        field = normalized.strip("\"'`?.: ").replace(" ", "_")
        if field in _INTERNAL_FIELDS:
            hint = (
                "Reuse evaluation_file from prepare_research_dataset or prepared_dataset context; "
                "prepare the supplied data first if no evaluation exists. "
                if field == "evaluation_file" else
                "Resolve tool arguments from previous tool results or authored workspace artifacts. "
            )
            raise ValueError(message + " " + hint)
        if (not normalized or normalized in {"...", "…"}
                or normalized.startswith("specific missing user information")):
            raise ValueError(message)
    return questions
