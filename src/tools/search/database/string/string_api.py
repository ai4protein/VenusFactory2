"""
STRING Database REST API — package entry point (re-exports).

Atomic modules live in the same package:
  string_rest, string_map_ids, string_network, string_network_image,
  string_interaction_partners, string_enrichment, string_ppi_enrichment,
  string_homology, string_version.

Skill (for Agent): src/agent/skills/string_database/
  - SKILL.md, references/string_reference.md

Usage:
  from src.tools.search.database.string import string_map_ids, string_network, ...
  from src.tools.search.database.string.string_api import string_version  # same
"""
from .string_map_ids import string_map_ids
from .string_network import string_network
from .string_network_image import string_network_image
from .string_interaction_partners import string_interaction_partners
from .string_enrichment import string_enrichment
from .string_ppi_enrichment import string_ppi_enrichment
from .string_homology import string_homology
from .string_version import string_version
from .string_rest import STRING_BASE_URL, BASE_URL

__all__ = [
    "STRING_BASE_URL",
    "BASE_URL",
    "string_map_ids",
    "string_network",
    "string_network_image",
    "string_interaction_partners",
    "string_enrichment",
    "string_ppi_enrichment",
    "string_homology",
    "string_version",
]
