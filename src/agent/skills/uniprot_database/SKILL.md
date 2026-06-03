---
name: uniprot_database
description: UniProt — protein sequence, function, taxonomy, cross-references. Search proteins by query, retrieve a UniProt entry, map IDs between databases (PDB↔UniProt etc.), pull FASTA sequence, fetch metadata, run SPARQL against sparql.uniprot.org. Use whenever the user mentions a UniProt accession (e.g. P04637), asks for protein function/sequence/family info, or needs cross-DB ID mapping. Don't use for AlphaFold structures (use alphafold_database) or PDB structures (use rcsb_database).
license: Unknown
metadata:
    skill-author: VenusFactory2.
---

# UniProt Database

## Overview

UniProt KnowledgeBase (UniProtKB) is the central protein-knowledge resource. This skill exposes 6 tools spanning text search, single-entry retrieval, ID mapping, sequence/metadata download, and SPARQL queries.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_uniprot_search_by_query** | `query` (UniProt query syntax, e.g. `organism_id:9606 AND reviewed:true`), `out_path`, optional `format` | rich JSON envelope | Search UniProtKB; results saved to file. |
| **download_uniprot_retrieve_by_id** | `uniprot_id` (e.g. `P04637`), `out_path`, optional `frmt` (`fasta` default) | rich JSON envelope | Retrieve one entry in the chosen format. |
| **download_uniprot_mapping** | `fr` (from db, e.g. `PDB`), `to` (to db, e.g. `UniProtKB_AC-ID`), `query` (comma-separated IDs), `out_path` | rich JSON envelope | Cross-database ID mapping (PDB↔UniProt, gene name↔accession, etc.). |
| **download_uniprot_seq_by_id** | `uniprot_id`, `out_path` | rich JSON envelope; FASTA file | Sequence FASTA only. |
| **download_uniprot_meta_by_id** | `uniprot_id`, `out_path` | rich JSON envelope; JSON file | Full metadata (function, taxonomy, GO, xrefs, etc.). |
| **download_uniprot_sparql_by_query** | `query` (SPARQL string), `out_dir` (required), `timeout` (default `120`s) | rich JSON envelope; SPARQL JSON file at `file_info.file_path`; `biological_metadata.row_count` + `head_vars` | Run an arbitrary SPARQL query against https://sparql.uniprot.org/sparql. |

## When to Use Which Tool

| Goal | Tool |
|---|---|
| You have a UniProt accession, want sequence | `download_uniprot_seq_by_id` |
| You have a UniProt accession, want everything else | `download_uniprot_meta_by_id` |
| You have a gene name / PDB ID, want UniProt | `download_uniprot_mapping` |
| You have a free-text query, want a list | `download_uniprot_search_by_query` |
| You need a complex cross-resource graph query (e.g. all enzymes in pathway X catalyzing reaction Y) | `download_uniprot_sparql_by_query` |

## SPARQL Quick Examples

**Top 5 proteins by name:**
```sparql
PREFIX up: <http://purl.uniprot.org/core/>
SELECT ?p ?name WHERE {
  ?p a up:Protein .
  ?p up:recommendedName / up:fullName ?name .
} LIMIT 5
```

**Proteins in human, with a specific GO term:**
```sparql
PREFIX up: <http://purl.uniprot.org/core/>
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
SELECT ?protein ?name
WHERE {
  ?protein a up:Protein .
  ?protein up:organism taxon:9606 .
  ?protein up:classifiedWith <http://purl.obolibrary.org/obo/GO_0003700> .
  ?protein up:recommendedName / up:fullName ?name .
} LIMIT 20
```

## Common Mistakes

- **Passing an unreviewed entry's accession when you need curated info**: filter with `reviewed:true` in your search.
- **Mapping in the wrong direction**: `fr=PDB to=UniProtKB_AC-ID` for "PDB → UniProt"; flip for the reverse.
- **SPARQL timeout on broad queries**: add `LIMIT N`; default endpoint timeout is harsh. Bump `timeout` parameter if your query is legitimately heavy.

## References

- [UniProt REST API](https://www.uniprot.org/help/api_queries)
- [UniProt query syntax](https://www.uniprot.org/help/query-fields)
- [UniProt SPARQL endpoint](https://sparql.uniprot.org/) — interactive query editor with examples
- [ID mapping db list](https://www.uniprot.org/help/id_mapping)
