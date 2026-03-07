---
name: ncbi_gene
description: Query NCBI Gene via E-utilities/Datasets API. Search by symbol/ID, retrieve gene info (RefSeqs, GO, locations, phenotypes), batch lookups, for gene annotation and functional analysis.
license: Unknown
metadata:
    skill-author: VenusFactory.
---

# NCBI Gene Database

## Overview

NCBI Gene is a comprehensive database integrating gene information from diverse species. It provides nomenclature, reference sequences (RefSeqs), chromosomal maps, biological pathways, genetic variations, phenotypes, and cross-references to global genomic resources. **In this project the agent exposes 3 download tools**: fetch gene data by ID or symbol (Datasets API), and batch lookup by symbols. For finer control, `ncbi_operations.py` also provides E-utilities download/query functions (esearch, esummary, efetch) and additional batch operations. All download/query functions return rich JSON `{status, file_info/content, content_preview, biological_metadata, execution_context}`.

## When to Use This Skill

This skill should be used when working with gene data including searching by gene symbol or ID, retrieving gene sequences and metadata, analyzing gene functions and pathways, or performing batch gene lookups.

## Quick Start

The skill provides:
1. **Project modules** in `src/tools/database/ncbi/`: `fetch_gene_data.py` (Datasets API), `query_gene.py` (E-utilities), `batch_gene_lookup.py` (batch operations), `ncbi_operations.py` (query/download operations); gene download functions re-exported via package.
2. References: `references/api_reference.md`, `references/common_workflows.md`

NCBI provides two main APIs:
- **Datasets API** — Optimized for gene data retrieval; used by `download_ncbi_gene_by_id` and `download_ncbi_gene_by_symbol` tools
- **E-utilities** — Full-featured API for complex queries; used by the esearch/esummary/efetch programmatic functions

### Agent Tools (Download Only)

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_ncbi_gene_by_id` | `gene_id`, `out_path` | Download gene data by NCBI Gene ID (Datasets API) to JSON |
| `download_ncbi_gene_by_symbol` | `symbol`, `taxon`, `out_path` | Download gene data by symbol and organism (Datasets API) to JSON |
| `download_ncbi_batch_lookup_by_symbols` | `gene_symbols`, `organism`, `out_path` | Batch lookup multiple genes by symbols to JSON |

Also available as general NCBI tools (useful in gene workflows):

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_ncbi_sequence` | `ncbi_id`, `out_path`, `db` (optional) | Download NCBI sequence by accession (FASTA) |
| `download_ncbi_metadata` | `ncbi_id`, `out_path`, `db`, `rettype` (optional) | Download NCBI metadata (GenBank/XML) |
| `download_ncbi_blast` | `sequence`, `out_path`, `program`, `database`, etc. | Submit BLAST search and download XML |

### Project Modules (Programmatic Use)

| Capability | Function | Module | Purpose |
|------------|----------|--------|---------|
| Fetch by ID | `fetch_gene_by_id(gene_id, api_key)` | fetch_gene_data.py | Datasets API: gene data as dict |
| Fetch by symbol | `fetch_gene_by_symbol(symbol, taxon, api_key)` | fetch_gene_data.py | Datasets API: gene data as dict |
| Fetch multiple | `fetch_multiple_genes(gene_ids, api_key)` | fetch_gene_data.py | Datasets API: multiple genes at once |
| Taxon lookup | `get_taxon_id(taxon_name)` | fetch_gene_data.py | Convert name to NCBI taxon ID |
| E-util search | `esearch(query, retmax, api_key)` | query_gene.py | Search Gene DB, returns Gene IDs |
| E-util summary | `esummary(gene_ids, api_key)` | query_gene.py | Get document summaries by Gene IDs |
| E-util fetch | `efetch(gene_ids, retmode, api_key)` | query_gene.py | Fetch full gene records (XML/text) |
| Search+summarize | `search_and_summarize(query, organism, max_results, api_key)` | query_gene.py | Convenience: search + display |
| Batch search | `batch_esearch(queries, organism, api_key)` | batch_gene_lookup.py | Search multiple symbols → ID map |
| Batch summary | `batch_esummary(gene_ids, api_key, chunk_size)` | batch_gene_lookup.py | Summaries in chunks |
| Batch by IDs | `batch_lookup_by_ids(gene_ids, api_key)` | batch_gene_lookup.py | Structured gene data by IDs |
| Batch by symbols | `batch_lookup_by_symbols(gene_symbols, organism, api_key)` | batch_gene_lookup.py | Structured gene data by symbols |
| Query: by ID | `query_ncbi_gene_by_id(gene_id)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: by symbol | `query_ncbi_gene_by_symbol(symbol, taxon)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: esearch | `query_ncbi_gene_esearch(query, retmax)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: esummary | `query_ncbi_gene_esummary(gene_ids)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: efetch | `query_ncbi_gene_efetch(gene_ids, retmode)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: batch search | `query_ncbi_batch_esearch(queries, organism)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: batch by IDs | `query_ncbi_batch_lookup_by_ids(gene_ids)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: batch by symbols | `query_ncbi_batch_lookup_by_symbols(gene_symbols, organism)` | ncbi_operations.py | Returns rich JSON in memory |
| Download: by ID | `download_ncbi_gene_by_id(gene_id, out_path)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: by symbol | `download_ncbi_gene_by_symbol(symbol, taxon, out_path)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: esearch | `download_ncbi_gene_esearch(query, out_path, retmax)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: esummary | `download_ncbi_gene_esummary(gene_ids, out_path)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: efetch | `download_ncbi_gene_efetch(gene_ids, out_path, retmode)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: batch search | `download_ncbi_batch_esearch(queries, out_path, organism)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: batch by IDs | `download_ncbi_batch_lookup_by_ids(gene_ids, out_path)` | ncbi_operations.py | Save to file, return rich JSON |
| Download: batch by symbols | `download_ncbi_batch_lookup_by_symbols(gene_symbols, organism, out_path)` | ncbi_operations.py | Save to file, return rich JSON |

## Core Capabilities

### 1. Download Gene by ID (Datasets API)

```python
from src.tools.database.ncbi import download_ncbi_gene_by_id

result = download_ncbi_gene_by_id("672", "output/ncbi_gene_brca1.json")
# Returns rich JSON with gene metadata, RefSeqs, GO annotations, etc.
```

### 2. Download Gene by Symbol (Datasets API)

```python
from src.tools.database.ncbi import download_ncbi_gene_by_symbol

result = download_ncbi_gene_by_symbol("BRCA1", "human", "output/ncbi_gene_brca1_by_symbol.json")
result = download_ncbi_gene_by_symbol("TP53", "Homo sapiens", "output/ncbi_gene_tp53.json")
```

### 3. Batch Lookup by Symbols

```python
from src.tools.database.ncbi import download_ncbi_batch_lookup_by_symbols

result = download_ncbi_batch_lookup_by_symbols(
    ["BRCA1", "TP53", "EGFR"], "human", "output/ncbi_genes_batch.json"
)
```

### 4. E-utilities Workflow (programmatic)

```python
from src.tools.database.ncbi.ncbi_operations import (
    query_ncbi_gene_esearch,
    download_ncbi_gene_esummary,
    download_ncbi_gene_efetch,
)

# Step 1: Search for gene IDs
esearch_result = query_ncbi_gene_esearch("BRCA1[gene] AND human[organism]", retmax=10)

# Step 2: Download summaries or full records
download_ncbi_gene_esummary(["672", "7157"], "output/gene_summaries.json")
download_ncbi_gene_efetch(["672"], "output/gene_full_record.xml")
```

### 5. Low-Level Direct Access (programmatic)

```python
from src.tools.database.ncbi.fetch_gene_data import fetch_gene_by_id, fetch_gene_by_symbol
from src.tools.database.ncbi.query_gene import esearch, esummary, efetch
from src.tools.database.ncbi.batch_gene_lookup import batch_lookup_by_symbols

# Datasets API
gene_data = fetch_gene_by_id("672")
gene_data = fetch_gene_by_symbol("BRCA1", "human")

# E-utilities
gene_ids = esearch("insulin[gene] AND human[organism]")
summaries = esummary(gene_ids)
records = efetch(gene_ids, retmode="xml")

# Batch
results = batch_lookup_by_symbols(["BRCA1", "TP53"], "human")
```

## Common Workflows

### Workflow 1: Gene Annotation (download)

```python
from src.tools.database.ncbi import (
    download_ncbi_gene_by_symbol,
    download_ncbi_batch_lookup_by_symbols,
)

# Single gene
download_ncbi_gene_by_symbol("BRCA1", "human", "output/brca1_annotation.json")

# Gene panel
download_ncbi_batch_lookup_by_symbols(
    ["BRCA1", "BRCA2", "TP53", "PTEN", "ATM"], "human", "output/cancer_panel.json"
)
```

### Workflow 2: Gene Search and Retrieve (programmatic)

```python
from src.tools.database.ncbi.ncbi_operations import (
    query_ncbi_gene_esearch,
    download_ncbi_gene_by_id,
)
import json

# Search by complex query
result = query_ncbi_gene_esearch("p53 AND human[organism]", retmax=5)
parsed = json.loads(result)

# Download details for each hit
if parsed.get("status") == "success":
    content = json.loads(parsed["content"])
    for gene_id in content.get("gene_ids", []):
        download_ncbi_gene_by_id(str(gene_id), f"output/gene_{gene_id}.json")
```

### Workflow 3: Cross-Database Integration

```python
from src.tools.database.ncbi import (
    download_ncbi_gene_by_id,
    download_ncbi_sequence,
    download_ncbi_metadata,
)

# Step 1: Get comprehensive gene info
download_ncbi_gene_by_id("672", "output/brca1_gene_info.json")

# Step 2: Download protein sequence
download_ncbi_sequence("NP_009225.1", "output/brca1_protein.fasta", db="protein")

# Step 3: Download GenBank metadata
download_ncbi_metadata("NP_009225.1", "output/brca1_metadata.gb", db="protein")
```

## Search Query Patterns

Example E-utilities query patterns for NCBI Gene:

- Gene symbol: `insulin[gene name] AND human[organism]`
- Gene with disease: `dystrophin[gene name] AND muscular dystrophy[disease]`
- Chromosome location: `human[organism] AND 17q21[chromosome]`
- GO term: `GO:0006915[biological process]` (apoptosis)
- Phenotype: `diabetes[phenotype] AND mouse[organism]`
- Pathway: `insulin signaling pathway[pathway]`

## API Access

**Rate Limits:**
- Without API key: 3 requests/second for E-utilities, 5 requests/second for Datasets API
- With API key: 10 requests/second for both APIs

**Authentication:**
Register for a free NCBI API key at https://www.ncbi.nlm.nih.gov/account/ to increase rate limits.

## Data Formats

NCBI Gene data can be retrieved in multiple formats:

| Format | Use case |
|--------|----------|
| JSON | Modern applications, programmatic processing |
| XML | Legacy systems, detailed metadata |
| GenBank | Sequence data with annotations |
| FASTA | Sequence analysis workflows |

## Best Practices

1. **Always specify organism** when searching by gene symbol to avoid ambiguity
2. **Use Gene IDs** for precise lookups when available
3. **Batch requests** when working with multiple genes to minimize API calls
4. **Cache results** locally to reduce redundant queries
5. **Include API key** in environment for higher rate limits
6. **Handle errors gracefully** with retry logic for transient failures

## Resources

### Helper Scripts

- `fetch_gene_data.py` — NCBI Datasets API: `fetch_gene_by_id()`, `fetch_gene_by_symbol()`, `fetch_multiple_genes()`
- `query_gene.py` — E-utilities: `esearch()`, `esummary()`, `efetch()`, `search_and_summarize()`
- `batch_gene_lookup.py` — Batch: `batch_esearch()`, `batch_esummary()`, `batch_lookup_by_ids()`, `batch_lookup_by_symbols()`

### Reference Documentation

- **`references/api_reference.md`** — E-utilities and Datasets API documentation, endpoints, parameters, response formats
- **`references/common_workflows.md`** — Additional examples and use case patterns
