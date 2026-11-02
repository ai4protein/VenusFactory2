"""Shared scientific modes; runtime implementation and language model are separate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchMode:
    id: str
    workflow: str
    scientific_name: str
    roles: tuple[str, ...]
    interaction: str
    completion_unit: str
    tools_optional: bool

    @property
    def engine(self) -> str:
        """Legacy web routing only; research profiles choose their adapter explicitly."""
        return LEGACY_DEPLOYMENT_ENGINES[self.id]


LEGACY_DEPLOYMENT_ENGINES = {"science_expert": "graph", "science_agent": "kimi-code"}

MODES = {
    "science_expert": ResearchMode(
        "science_expert",
        "role_collaboration",
        "goal-directed expert-team research",
        ("PI", "CB", "MLS", "SC"),
        "persistent_goal",
        "verified_goal",
        False,
    ),
    "science_agent": ResearchMode(
        "science_agent",
        "adaptive_execution",
        "interactive tool-using assistance",
        (),
        "conversation",
        "user_turn",
        True,
    ),
}


def get_research_mode(mode: str) -> ResearchMode:
    try:
        return MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown scientific mode: {mode}") from exc
