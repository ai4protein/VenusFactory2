"""Functional enrichment (GO, KEGG, Pfam, etc.). Skill: src/agent/skills/string_database/."""
from typing import List, Union

from .string_rest import STRING_BASE_URL, _identifiers_get, string_request_get


def string_enrichment(
    identifiers: Union[str, List[str]],
    species: int = 9606,
    caller_identity: str = "claude_scientific_skills",
) -> str:
    """Functional enrichment analysis. Returns TSV."""
    params = {
        "identifiers": _identifiers_get(identifiers),
        "species": species,
        "caller_identity": caller_identity,
    }
    return string_request_get(f"{STRING_BASE_URL}/tsv/enrichment", params)


if __name__ == "__main__":
    print("Testing registered tool: string_enrichment", flush=True)
    out = string_enrichment("P43403", species=9606)
    print(f"  string_enrichment(P43403, 9606) -> {len(out)} chars")
    if out.strip():
        print("  ", out.strip().split("\n")[0][:80] + "...")
    print("Done.")
