"""Get PPI network data (TSV). Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_get, string_request_get


def string_network(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    required_score: int = 400,
    network_type: str = "functional",
    add_nodes: int = 0,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Get protein-protein interaction network (TSV)."""
    params = {
        "identifiers": _identifiers_get(identifiers),
        "species": species,
        "required_score": required_score,
        "network_type": network_type,
        "add_nodes": add_nodes,
        "caller_identity": caller_identity,
    }
    return string_request_get(f"{STRING_BASE_URL}/tsv/network", params)


if __name__ == "__main__":
    print("Testing registered tool: string_network", flush=True)
    out = string_network("P43403", species=9606)
    print(f"  string_network(P43403, 9606) -> {len(out)} chars")
    if out.strip():
        print("  ", out.strip().split("\n")[0][:80] + "...")
    print("Done.")
