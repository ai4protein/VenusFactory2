"""Get interaction partners for protein(s). Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_get, string_request_get


def string_interaction_partners(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    limit: int = 10,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Get all interaction partners (TSV)."""
    params = {
        "identifiers": _identifiers_get(identifiers),
        "species": species,
        "required_score": required_score,
        "limit": limit,
        "caller_identity": caller_identity,
    }
    return string_request_get(f"{STRING_BASE_URL}/tsv/interaction_partners", params)
