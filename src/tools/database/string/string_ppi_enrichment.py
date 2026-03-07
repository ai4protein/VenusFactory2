"""PPI enrichment (test if network has more edges than expected). Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_get, string_request_get


def string_ppi_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Test if network is enriched. Returns JSON text."""
    params = {
        "identifiers": _identifiers_get(identifiers),
        "species": species,
        "required_score": required_score,
        "caller_identity": caller_identity,
    }
    return string_request_get(f"{STRING_BASE_URL}/json/ppi_enrichment", params)
