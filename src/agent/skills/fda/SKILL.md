---
name: fda
description: >-
  Query openFDA via VenusFactory for drugs, devices, adverse events, recalls, and regulatory submissions (510k, PMA). Use when the user needs FDA pharmacovigilance, labeling, NDC/UNII, or openFDA analytics. Do NOT use for ChEMBL bioactivity (chembl_database) or general biomedical literature (pubmed).
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# FDA / openFDA

## Overview

**Hub tool:** `query_fda` — drug label search with adverse-event fallback (see `FdaDrugLabelSearchInput`). This tool does not cover every openFDA category. Use registered search/download tools for additional public evidence, then analyze downloaded files in the session sandbox.

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **query_fda** | `query`, `max_results`, `max_content_length` | Conversational drug label / event lookup |

## Workflow

1. Quick drug question → `query_fda`
2. Devices / foods / recalls / 510k → inspect the relevant reference, then search available tools; report unsupported structured queries rather than inventing a callable API.
3. API credentials are managed by the server. Do not put credentials into analysis source or session files.

## When NOT to use

- Potency / IC50 / SAR → `chembl_database` + `rdkit`
- Literature review → `pubmed` / `openalex`

## Common mistakes

- Inventing tools like `query_fda_device` / `query_drug_events` as hub names
- Treating `query_fda` as full openFDA coverage

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="fda", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.

### Session execution
Use `write_session_file(path, content)` to write actual Python source and `run_session_python(path)` to execute it. Paths are relative to the server-assigned session workspace. Installed Python packages are available; repository source paths in this guide are host implementation references, not guest imports. Networking uses registered database/search/download tools; the analysis sandbox has no network access. Verify exit_code and actual output files.
