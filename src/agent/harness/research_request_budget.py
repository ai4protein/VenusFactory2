"""Role-aware retry ceilings within the existing Scientist episode budget."""

REVIEW_RESERVE = 3


def request_ceiling(role, model_calls):
    # Planning leaves the implementation/review reserve intact. Once independent
    # review begins, its bounded correction may use the remaining real budget.
    return model_calls if role in {"SC", "REVIEW"} else max(0, model_calls - REVIEW_RESERVE)
