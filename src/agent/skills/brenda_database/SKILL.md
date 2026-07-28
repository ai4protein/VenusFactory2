---
name: brenda_database
description: >-
  BRENDA enzyme kinetics via VenusFactory download tools (SOAP). Use for Km/kcat, reactions, organism comparison, environmental optima by EC number. Do NOT use for pathway maps alone (kegg_database) or protein sequence fetch (uniprot_database). Requires BRENDA_EMAIL and BRENDA_PASSWORD.
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# BRENDA Database

## Overview

Agent exposes **download-only** tools that write JSON/CSV to disk. Configure `BRENDA_EMAIL` + `BRENDA_PASSWORD` in the environment.

## Project Tools (VenusFactory2)

| Tool | Key args | Purpose |
|------|----------|---------|
| **download_brenda_km_values_by_ec_number** | `ec_number`, `out_path`, optional organism/substrate | Km values |
| **download_brenda_reactions_by_ec_number** | `ec_number`, `out_path` | Reactions |
| **download_brenda_enzymes_by_substrate** | `substrate`, `out_path`, `limit` | Enzymes by substrate |
| **download_brenda_compare_organisms_by_ec_number** | `ec_number`, `organisms`, `out_path` | Cross-organism |
| **download_brenda_environmental_parameters_by_ec_number** | `ec_number`, `out_path` | pH / temperature |
| **download_brenda_kinetic_data_by_ec_number** | `ec_number`, `out_path`, `format` | Kinetic export |
| **download_brenda_pathway_report** | `pathway` (dict), `out_path` | Pathway report file |

Returns rich JSON with `status` + `file_info` (not `{success, file_path}`). Tools live in `src/tools/database/tools_agent.py`.

## Workflow

1. Resolve EC number (user / UniProt / KEGG).
2. Pick the matching download tool; set session-scoped `out_path`.
3. Parse saved file; plot kinetics if useful (`nature_figure` for publication).

## When NOT to use

- Gene→pathway wiring without kinetics → `kegg_database` / `string_database`
- Empty files usually mean missing BRENDA credentials

## Common mistakes

- Importing `query_brenda_*` as hub tools (not registered — use `download_*`)
- Putting SOAP payloads in chat instead of reading `file_info.file_path`
- Wrong path docs pointing at `search/tools_agent.py`

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="brenda_database", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.
