---
name: interpro_domain_annotation
description: >-
  InterPro domain/family annotation via VenusFactory download tools. Use when the user needs domain boundaries, family membership, or UniProt→InterPro annotations for engineering target selection. Do NOT use for pathway enrichment (string_database / kegg_database) or kinetic parameters (brenda_database).
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# InterPro Domain Annotation

## Overview

Retrieves InterPro entry metadata, UniProt-linked annotations, and protein lists. Prefer these hub tools over ad-hoc HTTP in `agent_generated_code`.

## Project Tools (VenusFactory2)

| Tool | Args | When |
|------|------|------|
| **download_interpro_metadata_by_id** | `interpro_id`, `out_dir` | Known IPR / entry id |
| **download_interpro_annotations_by_uniprot_id** | `uniprot_id`, `out_dir` | Domains on a protein |
| **download_interpro_proteins_by_id** | `interpro_id`, `out_dir`, pagination args | Members of a family |
| **download_interpro_uniprot_list_by_id** | `interpro_id`, `out_dir` | UniProt accessions for an entry |
| **download_uniprot_meta_by_id** | `uniprot_id`, `out_path` | Cross-check function/GO |

## Workflow

1. If user has UniProt accession → `download_interpro_annotations_by_uniprot_id`.
2. For a specific InterPro id → metadata + optional protein list.
3. Before proposing mutations in catalytic triads, confirm the domain via InterPro + UniProt meta.
4. Optional: map domains onto structure via `protein_structure_pipeline` + residue indices.

## Common mistakes

- Treating InterPro ids as UniProt accessions.
- Editing conserved catalytic residues without reading annotation evidence.
