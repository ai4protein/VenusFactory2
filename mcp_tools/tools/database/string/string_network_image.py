"""Get network visualization as PNG. Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_get, string_request_image_get


def string_network_image(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_flavor: str = "evidence",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
) -> bytes:
    """Get network as PNG image. Returns bytes; on error returns error message as bytes."""
    params = {
        "identifiers": _identifiers_get(identifiers),
        "species": species,
        "required_score": required_score,
        "network_flavor": network_flavor,
        "add_nodes": add_nodes,
        "caller_identity": caller_identity,
    }
    return string_request_image_get(f"{STRING_BASE_URL}/image/network", params)
