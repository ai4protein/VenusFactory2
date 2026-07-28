---
name: hpa_expression_context
description: >-
  Human Protein Atlas expression and localization via VenusFactory download tools. Use when the user needs tissue expression, subcellular location, single-cell type, blood expression, or protein summary by gene symbol for therapeutic/target context. Do NOT use for mouse/non-human expression atlases or PPI networks (string_database).
license: Unknown
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# HPA Expression Context

## Overview

Downloads Human Protein Atlas JSON/TSV-style payloads for a **human gene symbol** (e.g. `TP53`, `EGFR`). Always set a session-scoped `out_path`.

## Project Tools (VenusFactory2)

| Tool | Purpose |
|------|---------|
| **download_hpa_protein_by_gene** | Protein summary / atlas entry |
| **download_hpa_subcellular_location_by_gene** | Subcellular localization |
| **download_hpa_tissue_expression_by_gene** | Tissue expression ranks |
| **download_hpa_single_cell_type_by_gene** | Single-cell type expression |
| **download_hpa_blood_expression_by_gene** | Blood / immune context |

Args pattern: `gene_name`, `out_path`.

## Workflow

1. Confirm human gene symbol (map via UniProt / NCBI Gene if needed).
2. Fetch tissue + subcellular (minimum useful pair).
3. Optional single-cell / blood for immunology or circulating targets.
4. **Figure:** bar chart of top tissues (`nature_figure` if publication).

## When NOT to use

- Non-human organisms → other databases / literature
- Interaction partners → `string_database`
- Domain architecture → `interpro_domain_annotation`

## Common mistakes

- Using UniProt accession as `gene_name` without checking HPA expects a gene symbol
- Skipping tissue figure after download (CB policy expects a plot for expression tables)
