---
name: rdkit
description: >-
  RDKit cheminformatics via VenusFactory bioinfo scripts and session Python tools. Use for SMILES/SDF, descriptors, fingerprints, substructure filters, similarity. Do NOT use for ChEMBL bioactivity download (chembl_database) or openFDA (fda). No dedicated rdkit_* LangChain tool — run scripts or import modules.
license: BSD-3-Clause
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# RDKit Cheminformatics

## Overview

Fine-grained molecular analysis with the installed `rdkit` Python package in the session sandbox.

## VenusFactory execution

- No `rdkit_*` `@tool` in the hub — write source with **`write_session_file`**, then execute **`run_session_python`**. Import installed RDKit modules, not the host repository.
- Extended tutorials: `read_skill("rdkit", relative_path="references/legacy_guide.md")`.
- Bioactivity from ChEMBL first → `chembl_database`, then RDKit for chemistry ops.

## Python packages

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
# Read the user's SMILES/SDF from relative session files. Validate parsed molecules
# before computing descriptors, SMARTS matches or fingerprint similarities.
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

### Session execution
Use `write_session_file(path, content)` to write actual Python source and `run_session_python(path)` to execute it. Paths are relative to the server-assigned session workspace. Installed Python packages are available; repository source paths in this guide are host implementation references, not guest imports. Networking uses registered database/search/download tools; the analysis sandbox has no network access. Verify exit_code and actual output files.
