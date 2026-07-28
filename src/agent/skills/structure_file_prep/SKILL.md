---
name: structure_file_prep
description: >-
  Structure/sequence file preparation with VenusFactory file tools. Use for FASTA parsing, PDB chain extraction, PDB↔mmCIF conversion (MAXIT), apo checks, batch PDB→FASTA, and UniProt ID from RCSB metadata. Do NOT use for structure prediction (protein_structure_pipeline) or homology search.
license: Apache-2.0
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Structure & Sequence File Prep

## Project Tools (VenusFactory2)

| Tool | Purpose |
|------|---------|
| **read_fasta** | Parse multi-FASTA → headers/sequences |
| **extract_uids_from_fasta** | Pull UIDs from headers |
| **uid_file_to_chunks** | Chunk UID lists for batch jobs |
| **pdb_chain_sequences** | Per-chain sequences from PDB |
| **get_seq_from_pdb_chain_a** | Chain A sequence shortcut |
| **pdb_dir_to_fasta** | Directory of PDBs → FASTA |
| **check_pdb_apo** | Ligand-free / apo heuristic |
| **maxit_structure_convert** | `pdb2cif` / `cif2pdb` / `cif2mmcif` (needs MAXIT) |
| **extract_uniprot_id_from_rcsb_metadata** | UniProt from RCSB metadata JSON |
| **unzip_archive** / **ungzip_file** | Unpack downloads |

## Workflows

### PDB → design/mutation ready

1. `check_pdb_apo` (if apo backbone required)
2. `pdb_chain_sequences` → choose designed chains
3. Optional `maxit_structure_convert` for format mismatches

### RCSB metadata → UniProt → AlphaFold

1. `download_rcsb_entry_metadata_by_pdb_id`
2. `extract_uniprot_id_from_rcsb_metadata`
3. Hand off to `alphafold_database` / `protein_structure_pipeline`

## Common mistakes

- Hardcoding chain lengths without `pdb_chain_sequences`
- Calling MAXIT strategies other than `pdb2cif`/`cif2pdb`/`cif2mmcif`
- Feeding directories to tools that expect a single file path
