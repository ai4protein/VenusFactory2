"""Map protein names/IDs to STRING IDs. Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_post, string_request_post


def string_map_ids(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    limit: int = 1,
    echo_query: int = 1,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Map protein names/synonyms/IDs to STRING IDs. Returns TSV text."""
    data = {
        "identifiers": _identifiers_post(identifiers),
        "species": species,
        "limit": limit,
        "echo_query": echo_query,
        "caller_identity": caller_identity,
    }
    return string_request_post(f"{STRING_BASE_URL}/tsv/get_string_ids", data)
