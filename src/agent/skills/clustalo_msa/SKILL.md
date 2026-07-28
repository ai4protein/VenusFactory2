---
name: clustalo_msa
description: Multiple sequence alignment of proteins via EBI Clustal Omega web service. Use when you have ≥2 protein sequences in a FASTA file (≤4000 sequences, ≤4 MB) and need an alignment to assess conservation, residue importance, or domain structure. Do NOT use for: single sequences, homology search (use protein_sequence_similarity_search), structural alignment (use foldseek), or DNA/RNA alignment.
license: Apache-2.0 (adapted from google-deepmind/science-skills)
metadata:
  version: "1.0"
  skill-author: VenusFactory2 (adapted from Google DeepMind)
---

# Clustal Omega MSA (EBI)

## Overview

Submits a FASTA file with multiple protein sequences to the [EBI Clustal Omega REST service](https://www.ebi.ac.uk/jdispatcher/msa/clustalo), polls for completion, and downloads the resulting alignment in FASTA format. Pipeline is fully managed by the tool — the agent only provides the input FASTA and an output directory.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_clustalo_msa_by_fasta** | `fasta_path` (required, path to input FASTA), `out_dir` (required), `email` (optional; falls back to env `USER_EMAIL`, then `noreply@venusfactory.cn`), `poll_interval` (default `10.0` s), `timeout_secs` (default `900` s) | JSON: `{status, file_info {file_path, file_name, file_size, format: "fasta"}, content_preview, biological_metadata {input_sequences, aligned_sequences, job_id, email}, execution_context}` | Submit + poll + download MSA. Writes `<input_stem>_msa.fasta` to `out_dir`. |

## When to Use This Skill

- Compute MSA for a small/medium set of homologous proteins (UniProt search results, BLAST hits, manually curated set)
- Generate input for conservation scoring, phylogenetic analysis, or HMM profile training
- Identify conserved active-site residues from a small protein family

## When NOT to Use

- Single sequence input → use `protein_sequence_similarity_search` to first find homologs
- >4000 sequences or >4 MB FASTA → EBI rejects; split into chunks or run locally with `mafft --auto`
- DNA / RNA alignment → Clustal Omega is for proteins
- Structural alignment of 3D structures → use `download_foldseek_results_by_pdb_file`

## Pipeline

1. **Validate input**: file exists, size ≤ 4 MB, 2 ≤ sequence count ≤ 4000.
2. **Submit**: `POST https://www.ebi.ac.uk/Tools/services/rest/clustalo/run` with `email` + `title` + `sequence` form data.
3. **Poll**: `GET .../status/{job_id}` every `poll_interval` seconds until `FINISHED` (or `ERROR`/`FAILURE`/`NOT_FOUND` → fail fast).
4. **Download**: `GET .../result/{job_id}/fa` → FASTA alignment text.
5. **Save**: write to `<out_dir>/<input_stem>_msa.fasta`.

## Rate Limiting & Politeness

- The tool defaults to 10-second polls with a 15-minute wall-clock timeout.
- EBI requests a valid contact email — the default `noreply@venusfactory.cn` works but setting `USER_EMAIL` in your environment is preferred so EBI can contact you if your job affects service health.
- A single submit + many polls is the established contract; do not invoke the tool in a tight loop.

## Common Mistakes

- **Passing a single sequence**: the tool errors with `ValidationError: need ≥2 sequences`. Run a similarity search first.
- **Pasting raw sequences into a tool argument**: this tool only accepts a *file path*. Write the FASTA to disk first (use `read_fasta`, `extract_uids_from_fasta`, etc., to compose).
- **Mixing nucleotide and protein sequences**: EBI returns garbage. Filter the input FASTA before calling.

## References

- [EBI Clustal Omega REST docs](https://www.ebi.ac.uk/Tools/common/tools/help/index.html?tool=clustalo)
- [Terms of use](https://www.ebi.ac.uk/about/terms-of-use/)
- Adapted from `google-deepmind/science-skills:skills/protein_sequence_msa/scripts/msa_align.py`
