---
name: protein_sequence_similarity_search
description: Find homologous protein sequences from a query sequence using MMseqs2 (fast, ColabFold web API) or BLAST (comprehensive, EBI). Use when the user provides a protein sequence or FASTA file and wants homologs, function inference by sequence similarity, or input for an MSA. Do NOT use for structural similarity (use foldseek) or DNA/RNA queries.
license: Apache-2.0 (adapted from google-deepmind/science-skills)
metadata:
  version: "1.0"
  skill-author: VenusFactory2 (adapted from Google DeepMind)
---

# Protein Sequence Similarity Search

## Overview

Two complementary search engines, each wrapped as a submit-poll-download-parse pipeline:

- **MMseqs2 (ColabFold)** — fast, default. Searches UniRef (+ optional MGnify environmental). Best for quickly building an MSA-grade hit set.
- **EBI BLAST (NCBI BLAST hosted at EBI)** — slower but more exhaustive search across multiple UniProt/UniRef/PDB databases. Use when you need BLAST-specific scoring (e.g., comparing against a published BLAST result) or when MMseqs2 returns few hits.

Both tools save the full hit list to a JSON file on disk and return only a top-10 markdown preview in the response, keeping the agent's context window small.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_mmseqs2_homologs_by_sequence** | `sequence_or_fasta_path` (required, raw sequence or FASTA file path), `out_dir` (required), `include_mgnify` (default `False`), `poll_interval` (default `10.0` s), `timeout_secs` (default `900` s) | JSON: `{status, file_info {file_path -> mmseqs2_<ticket>.json, file_size, format: "json"}, content_preview (top-10 markdown table), biological_metadata {engine, ticket_id, query_length, hit_count, include_mgnify}}` | Fast UniRef homologue search via ColabFold MMseqs2 API. |
| **download_blast_homologs_by_sequence** | `sequence_or_fasta_path` (required), `out_dir` (required), `database` (default `"uniprotkb_swissprot"`; comma-separated list ok), `email` (optional, falls back to env `USER_EMAIL`), `poll_interval` (default `30.0` s), `timeout_secs` (default `900` s) | JSON: `{status, file_info {file_path -> blast_<jobid>.json}, content_preview, biological_metadata {engine, job_id, query_length, databases, hit_count, email}}` | Authoritative BLAST search against UniProt/UniRef/PDB at EBI. |

## When to Use This Skill

- The user gives a sequence and asks "find similar proteins" / "what's this protein's family" / "find homologs"
- You need to build an MSA from a single seed sequence: run MMseqs2 → write hits → `download_clustalo_msa_by_fasta`
- You need orthologs in a specific clade: BLAST with `database=uniprotkb_human` (or `_bacteria`, `_viruses`, etc.)
- You need PDB hits to seed structure-based analysis: BLAST with `database=pdb`

## Which Engine to Pick

| Situation | Engine |
|---|---|
| Default / unspecified | **MMseqs2** (faster, ~1-3 min) |
| User explicitly says "BLAST" | **BLAST** |
| MMseqs2 returned <5 hits | **BLAST** fallback |
| Need PDB-only hits | **BLAST** with `database=pdb` |
| Need MGnify environmental hits | **MMseqs2** with `include_mgnify=True` |
| Want comprehensive UniProtKB+TrEMBL coverage | **BLAST** with `database=uniprotkb` |

## Supported BLAST Databases

`uniprotkb` `uniprotkb_swissprot` `uniprotkb_swissprotsv` `uniprotkb_reference_proteomes` `uniprotkb_trembl` `uniprotkb_refprotswissprot` `uniprotkb_archaea` `uniprotkb_arthropoda` `uniprotkb_bacteria` `uniprotkb_complete_microbial_proteomes` `uniprotkb_eukaryota` `uniprotkb_fungi` `uniprotkb_human` `uniprotkb_mammals` `uniprotkb_nematoda` `uniprotkb_rodents` `uniprotkb_vertebrates` `uniprotkb_viridiplantae` `uniprotkb_viruses` `uniprotkb_enzyme` `uniprotkb_covid19` `uniref100` `uniref90` `uniref50` `pdb`

## Output JSON Schema (file_info.file_path)

```json
{
  "hits": [
    {"target_id": "...", "q_cov": 78.3, "e_value": 1.2e-50, "identity": 0.45, "aln_len": 240, ...},
    ...
  ],
  "metadata": {"engine": "MMseqs2 (ColabFold)", "ticket_id": "...", "query_length": 256, "hit_count": 178, ...}
}
```

## Rate Limiting

- ColabFold MMseqs2: 2 req/s, ~1-3 min wall clock per job. Polls every 10 s.
- EBI BLAST: 2 req/s submit; polls every 30 s. Jobs take 2-10 min.
- Both timeout at 15 min by default — increase `timeout_secs` for very large queries.

## Common Mistakes

- **Passing a FASTA file with multiple records** but expecting all to be searched: only the first record is used. Loop in the agent if you need multiple queries.
- **Skipping `out_dir`**: required. Tools error with `ValidationError: empty out_dir`.
- **Setting `database` to an unsupported value**: returns `ValidationError` with the full allowed list — pick from that list.
- **Treating MMseqs2 hits as the final answer when count is 0**: this is a soft failure; fall back to BLAST.

## References

- [ColabFold MMseqs2 API](https://github.com/sokrypton/ColabFold)
- [EBI BLAST web service](https://www.ebi.ac.uk/jdispatcher/sss/ncbiblast) and [terms of use](https://www.ebi.ac.uk/about/terms-of-use/)
- Adapted from `google-deepmind/science-skills:skills/protein_sequence_similarity_search/`
