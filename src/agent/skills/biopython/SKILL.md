---
name: biopython
description: >-
  Biopython guidance for sequence I/O, alignments, Bio.PDB, and Entrez parsing. Use for custom bioinformatics code via agent_generated_code. Prefer VenusFactory download tools for NCBI/UniProt/AlphaFold bulk fetches so large payloads stay on disk. Do NOT use as a substitute for protein_sequence_similarity_search or alphafold_database hub tools.
license: Unknown
metadata:
  version: "1.2"
  skill-author: VenusFactory2
---

# Biopython

## Overview

Library skill for parsing and analysis. Prefer hub download tools when fetching remote data; use Biopython to parse the saved files.

## Project Tools to prefer first

| Goal | Hub tool / skill |
|------|------------------|
| Read FASTA | **read_fasta** |
| PDB → sequences | **pdb_chain_sequences**, **get_seq_from_pdb_chain_a**, **pdb_dir_to_fasta** |
| NCBI sequence/BLAST | **download_ncbi_sequence**, **download_ncbi_blast** (`ncbi_sequence`) |
| UniProt | `uniprot_database` tools |
| AlphaFold structure | `alphafold_database` tools |
| Custom parse/plot | **agent_generated_code** + this skill |

## Module map (on-demand references)

| Topic | Load with relative_path |
|-------|-------------------------|
| SeqIO / sequences | `references/sequence_io.md` |
| Alignments | `references/alignment.md` or `references/legacy_guide.md` |
| Entrez / NCBI | `references/databases.md` — set email + `NCBI_API_KEY` |
| Bio.PDB | `references/structure.md` or `references/legacy_guide.md` |
| Phylogenetics | `references/phylogenetics.md` |
| Full cookbook | `references/legacy_guide.md` |

```text
read_skill(skill_id="biopython", relative_path="references/databases.md")
```

## Entrez setup (when needed)

```python
from Bio import Entrez
import os
Entrez.email = "you@example.com"
if k := os.environ.get("NCBI_API_KEY"):
    Entrez.api_key = k
```

## Common mistakes

- Pulling huge GenBank XML into chat instead of `download_ncbi_*`
- Reimplementing AlphaFold download with Biopython instead of hub tools
- Missing Entrez email (NCBI requirement)

## References

- See Module map above; full cookbook: `references/legacy_guide.md`
