---
name: chembl_database
description: >-
  ChEMBL bioactive molecules and drugs via VenusFactory download tools. Use for molecule/drug by ID, similarity/substructure by SMILES, SAR starting points. Do NOT use for openFDA regulatory data (fda) or RDKit-only local chemistry (rdkit).
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# ChEMBL Database

## Project Tools (VenusFactory2)

| Tool | Key args | Purpose |
|------|----------|---------|
| **download_chembl_molecule_by_id** | `mol_id`, **`out_path`** | Molecule record |
| **download_chembl_drug_by_id** | `chembl_id` / drug id, **`out_path`** | Drug record |
| **download_chembl_similarity_by_smiles** | `smiles`, threshold, **`out_path`** | Similarity neighbors |
| **download_chembl_substructure_by_smiles** | `smiles`, **`out_path`** | Substructure hits |

Returns rich JSON with `file_info`. Output path arg is **`out_path`** (not `out_dir`). Follow with `rdkit` for descriptors/filters if needed.

## Workflow

1. Resolve ChEMBL ID or start from SMILES.
2. Download → parse JSON/TSV on disk.
3. Optional RDKit property filter / figure of activity distributions.

## When NOT to use

- FDA labels / adverse events → `fda`
- Pure local fingerprinting without ChEMBL → `rdkit`

## References (progressive disclosure)

**Trust order:** `SKILL.md` (hub tools & envelopes) → topic refs → `references/legacy_guide.md` (archived; may be outdated).

```text
read_skill(skill_id="chembl_database", relative_path="references/legacy_guide.md")
```

Load legacy only after the hub workflow in this file is insufficient.
