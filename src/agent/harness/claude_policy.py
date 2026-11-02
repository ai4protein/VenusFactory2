"""Shared native Claude control admission; no inference or runtime imports."""


def validate_profile(profile):
    from .provider_config import local_route
    configured_defaults = profile.budget.thinking is None and local_route(
        profile.model.model, profile.model.base_url, profile.model.protocol
    ) is not None
    if (
        profile.model.protocol != "chat_completions"
        or (profile.budget.thinking != "disabled" and not configured_defaults)
        or profile.budget.reasoning_effort is not None
        or profile.budget.temperature is None
        or profile.request_recovery != "none"
    ):
        raise ValueError("Native Claude bridge requires explicit Chat controls, disabled thinking and no retry")
