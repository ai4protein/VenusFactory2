---
name: ncbi_sequence
description: NCBI E-utilities for biological sequences — fetch protein/nucleotide FASTA by accession, run BLAST, translate CDS to protein, search NCBI Protein by gene+organism. Use when the user provides an NCBI accession (NP_, XP_, NM_, NR_, etc.), asks for a sequence by gene name + species, or needs to translate a coding sequence. Don't use for ClinVar variants (use ncbi_clinvar) or gene metadata lookup (use ncbi_gene).
license: Unknown
metadata:
    skill-author: VenusFactory2.
---

# NCBI Sequence Tools

## Overview

Wraps NCBI E-utilities (`efetch`, `esearch`) for sequence-centric workflows. Honors `NCBI_API_KEY` env var to raise the QPS limit from 3 → 10. Set `USER_EMAIL` env var to identify yourself to NCBI as a good citizen.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_ncbi_sequence** | `ncbi_id`, `out_dir`, `db` (default `protein`) | rich JSON envelope; FASTA file | Fetch a single sequence by accession from `protein` or `nuccore`. |
| **download_ncbi_metadata** | `ncbi_id`, `out_path`, `db` | rich JSON envelope; metadata JSON | Fetch GenBank-style metadata for an accession. |
| **download_ncbi_blast** | (see existing schema) | rich JSON envelope | NCBI-hosted BLAST. **Prefer `download_mmseqs2_homologs_by_sequence` (faster) or `download_blast_homologs_by_sequence` (EBI mirror) for protein-protein searches.** |
| **translate_ncbi_cds_to_protein** | `accession` (nuccore, e.g. `NM_000518` for HBB mRNA), `out_dir`, `target_length` (default `0` = longest), `timeout` | rich JSON envelope; FASTA at `<out_dir>/<accession>_protein.fasta`; `biological_metadata.method="fasta_cds_aa"` | Use `efetch(rettype=fasta_cds_aa)` to get the CDS-translated protein, pick the translation closest to `target_length` (or longest). |
| **search_ncbi_protein_by_gene_and_organism** | `gene` (e.g. `TP53`), `organism` (`"Homo sapiens"`), `out_dir`, `target_length` (default `0` = no length filter; non-zero filters to ±25 aa window), `retmax` (default `10`), `timeout` | rich JSON envelope; multi-FASTA + `<stem>.json` summary; `biological_metadata.summary_path` for the per-hit JSON | Search NCBI Protein DB with `<gene>[Gene Name] AND <organism>[Organism]`, fetch all hits as multi-FASTA. |

## Workflow: "Get the protein sequence for gene X in species Y"

1. **Try `search_ncbi_protein_by_gene_and_organism`** first — if it finds 1-5 hits, pick the canonical one.
2. **If you know the mRNA accession** (e.g. from a Gene record), use `translate_ncbi_cds_to_protein` for the canonical CDS-derived protein (no ambiguity from isoforms).
3. **Fallback**: keyword search via `download_ncbi_metadata` to find an accession, then `download_ncbi_sequence`.

## Workflow: "Translate this mRNA to protein"

- Direct: `translate_ncbi_cds_to_protein(accession=..., target_length=expected_aa_count)`. The tool uses NCBI's pre-translated CDS protein when available — no client-side translation needed, no codon-table issues.

## Common Mistakes

- **Confusing `protein` and `nuccore` DBs**: protein accessions (NP_, XP_, AAA-style) go to `db=protein`; mRNA/genomic (NM_, NC_, etc.) go to `db=nuccore`. `translate_ncbi_cds_to_protein` always uses `nuccore` internally.
- **Mismatched gene+organism**: NCBI is strict; `TP53 AND Homo sapiens` works, `tp53 AND human` may not.
- **Forgetting `NCBI_API_KEY`**: at 3 QPS you'll hit rate limits with batch operations. Set the env var to bump to 10 QPS.
- **target_length too narrow**: `search_ncbi_protein_by_gene_and_organism` applies ±25 aa filter; if your target_length is uncertain, pass 0 and pick from results.

## References

- [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Get an NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) (free)
- [Entrez query syntax](https://www.ncbi.nlm.nih.gov/books/NBK3837/)
