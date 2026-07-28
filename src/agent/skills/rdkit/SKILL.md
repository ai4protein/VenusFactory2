---
name: rdkit
description: >-
  RDKit cheminformatics via VenusFactory bioinfo scripts and agent_generated_code. Use for SMILES/SDF, descriptors, fingerprints, substructure filters, similarity. Do NOT use for ChEMBL bioactivity download (chembl_database) or openFDA (fda). No dedicated rdkit_* LangChain tool — run scripts or import modules.
license: BSD-3-Clause
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# RDKit Cheminformatics

## Overview

Fine-grained molecular control in Python. Platform helpers live under `src/tools/bioinfo/rdkit/` (CLI + importable functions). Prefer them over re-implementing RDKit from scratch.

## VenusFactory execution

- No `rdkit_*` `@tool` in the hub — use **`agent_generated_code`** (or `python_repl`) and import the modules below.
- Extended tutorials: `read_skill("rdkit", relative_path="references/legacy_guide.md")`.
- Bioactivity from ChEMBL first → `chembl_database`, then RDKit for chemistry ops.

## Project scripts (src/tools/bioinfo/rdkit/)

| Script | Purpose |
|--------|---------|
| `molecular_properties.py` | MW, LogP, TPSA, Lipinski, QED → CSV |
| `substructure_filter.py` | SMARTS/SMILES include/exclude filters |
| `similarity_search.py` | Morgan/RDKit/MACCS fingerprints + Tanimoto |

```python
from src.tools.bioinfo.rdkit import calculate_properties, filter_molecules, similarity_search
# loaders: substructure_filter.load_molecules / similarity_search.load_molecules
```

```bash
python -m src.tools.bioinfo.rdkit.molecular_properties "CCO"
python -m src.tools.bioinfo.rdkit.similarity_search "CCO" database.smi --threshold 0.7 -o hits.csv
```

## When NOT to use

- Download ChEMBL molecules/activities → `chembl_database`
- Simple FDA label search → `fda` / `query_fda`

## Common mistakes

- Importing `load_molecules` from package root (use module-specific loaders)
- Assuming a LangChain tool named `rdkit_*` exists
- Sanitization failures on dirty SMILES — handle `None` mols

## References

- `references/api_reference.md`, `descriptors_reference.md`, `smarts_patterns.md`
- `references/legacy_guide.md` — full capability cookbook
