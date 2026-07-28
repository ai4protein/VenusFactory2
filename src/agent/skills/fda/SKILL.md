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

**Hub tool:** `query_fda` — drug label search with adverse-event fallback (see `FdaDrugLabelSearchInput`). Broader openFDA categories (devices, foods, 510k, …) use `FDAQuery` via **`agent_generated_code`** and modules under `src/tools/search/deepsearch/fda/` (or legacy `references/`).

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **query_fda** | `query`, `max_results`, `max_content_length` | Conversational drug label / event lookup |

## Workflow

1. Quick drug question → `query_fda`
2. Devices / foods / recalls / 510k → load `references/devices.md` (etc.) + `agent_generated_code` with `FDAQuery`
3. Set `OPENFDA_API_KEY` when available to raise rate limits

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
