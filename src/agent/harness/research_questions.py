"""Validate explicit user questions without turning template text into a pause."""
from __future__ import annotations


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
        if (not normalized or normalized in {"...", "…"}
                or normalized.startswith("specific missing user information")):
            raise ValueError(message)
    return questions
