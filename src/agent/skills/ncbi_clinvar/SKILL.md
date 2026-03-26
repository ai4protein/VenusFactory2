---
name: ncbi_clinvar
description: Query NCBI ClinVar for variant clinical significance. Search by gene/condition/CLNSIG, interpret pathogenicity, use E-utilities or FTP; annotate VCFs. Use project tools in src.tools.database.ncbi.
license: Unknown
metadata:
    skill-author: VenusFactory2
---

# ClinVar Database (NCBI)

## Overview

ClinVar is NCBI's archive of relationships between human genetic variants and phenotypes, with supporting evidence. **In this project the agent exposes one compound download tool** (`download_ncbi_clinvar_variants`) that searches by term, retrieves variant IDs, and saves results to JSON. For finer control, `ncbi_operations.py` also provides atomic ClinVar E-utilities download/query functions (esearch, esummary, efetch), and the low-level module `ncbi_clinvar.py` provides term builders and FTP helpers. All download/query functions return rich JSON `{status, file_info/content, content_preview, biological_metadata, execution_context}`.

## When to Use This Skill

- Searching for variants by gene, condition, or clinical significance
- Interpreting clinical significance (pathogenic, benign, VUS); see `references/clinical_significance.md`
- Accessing ClinVar via E-utilities (esearch, esummary, efetch) or FTP
- Understanding review status (star ratings) and conflicting interpretations
- Annotating variant call sets with clinical significance; see `references/data_formats.md` for VCF/XML/tab formats

## Quick Start

The skill provides:
1. **Project modules** in `src/tools/database/ncbi/`: `ncbi_clinvar.py` (atomic E-utilities, term builder, FTP), `ncbi_operations.py` (query/download operations); ClinVar download functions re-exported via package.
2. References: `references/api_reference.md`, `references/clinical_significance.md`, `references/data_formats.md`

### Agent Tools (Download Only)

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_ncbi_clinvar_variants` | `term`, `out_path`, `retmax` (optional, default 20) | Search ClinVar variants by term and save results to JSON |

Also available as general NCBI tools (not ClinVar-specific but useful in ClinVar workflows):

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_ncbi_sequence` | `ncbi_id`, `out_path`, `db` (optional) | Download NCBI sequence by accession (FASTA) |
| `download_ncbi_metadata` | `ncbi_id`, `out_path`, `db`, `rettype` (optional) | Download NCBI metadata (GenBank/XML) |
| `download_ncbi_blast` | `sequence`, `out_path`, `program`, `database`, etc. | Submit BLAST search and download XML |

### Project Modules (Programmatic Use)

| Capability | Function | Module | Purpose |
|------------|----------|--------|---------|
| Build search term | `build_clinvar_term(gene, clinical_significance, condition, chr, variant_name, exclude_conflicting, raw_term)` | ncbi_clinvar.py | Build E-utilities query string |
| High-level search | `query_clinvar(term, retmax, retstart, retmode, api_key)` | ncbi_clinvar.py | Search ClinVar, return esearch response text |
| Get summaries | `get_clinvar_summary(variation_ids, retmode, api_key)` | ncbi_clinvar.py | Get variant summaries by IDs |
| Full records | `fetch_clinvar_records(variation_ids, rettype, retmode, api_key)` | ncbi_clinvar.py | Fetch full XML/VCV records |
| Parse IDs | `parse_esearch_ids(esearch_response_text)` | ncbi_clinvar.py | Extract Variation IDs from esearch JSON |
| Parse count | `parse_esearch_count(esearch_response_text)` | ncbi_clinvar.py | Extract total count from esearch JSON |
| FTP URL | `get_clinvar_ftp_url(key)` | ncbi_clinvar.py | Get FTP URL by key (variant_summary, vcf_grch38, xml_latest, etc.) |
| FTP download | `download_clinvar_ftp(url_or_key, out_dir, filename)` | ncbi_clinvar.py | Download file from ClinVar FTP |
| Atomic esearch | `clinvar_esearch(term, retmax, retstart, retmode, api_key, sort)` | ncbi_clinvar.py | Raw esearch response text |
| Atomic esummary | `clinvar_esummary(id_list, retmode, api_key)` | ncbi_clinvar.py | Raw esummary response text |
| Atomic efetch | `clinvar_efetch(id_list, rettype, retmode, api_key)` | ncbi_clinvar.py | Raw efetch response text |
| Query: esearch | `query_ncbi_clinvar_esearch(term, retmax, retstart, retmode)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: esummary | `query_ncbi_clinvar_esummary(id_list, retmode)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: efetch | `query_ncbi_clinvar_efetch(id_list, rettype, retmode)` | ncbi_operations.py | Returns rich JSON in memory |
| Query: variants | `query_ncbi_clinvar_variants(term, retmax)` | ncbi_operations.py | Compound: esearch → returns rich JSON |
| Download: esearch | `download_ncbi_clinvar_esearch(term, out_path, retmax, retstart, retmode)` | ncbi_operations.py | Save esearch results to file |
| Download: esummary | `download_ncbi_clinvar_esummary(id_list, out_path, retmode)` | ncbi_operations.py | Save esummary results to file |
| Download: efetch | `download_ncbi_clinvar_efetch(id_list, out_path, rettype, retmode)` | ncbi_operations.py | Save efetch results to file |
| Download: variants | `download_ncbi_clinvar_variants(term, out_path, retmax)` | ncbi_operations.py | Compound: esearch → save to file |

## Core Capabilities

### 1. Search ClinVar Variants (download)

```python
from src.tools.database.ncbi import download_ncbi_clinvar_variants

# Simple gene search
result = download_ncbi_clinvar_variants("BRCA1[gene]", "output/clinvar_brca1.json", retmax=50)

# Pathogenic variants for a gene
result = download_ncbi_clinvar_variants(
    "BRCA1[gene] AND pathogenic[CLNSIG]",
    "output/clinvar_brca1_pathogenic.json",
    retmax=100,
)
```

### 2. Build Search Terms (programmatic)

```python
from src.tools.database.ncbi.ncbi_clinvar import build_clinvar_term

# Build term with filters
term = build_clinvar_term(
    gene="BRCA1",
    clinical_significance="pathogenic",
    exclude_conflicting=True,
)
# → "BRCA1[gene] AND pathogenic[CLNSIG] NOT conflicting[RVSTAT]"

# Search by condition
term = build_clinvar_term(condition="breast cancer")
# → "breast cancer[disorder]"
```

### 3. Atomic E-utilities Workflow (programmatic)

```python
from src.tools.database.ncbi.ncbi_clinvar import (
    build_clinvar_term, query_clinvar, parse_esearch_ids,
    get_clinvar_summary, fetch_clinvar_records,
)

# Step 1: Build term and search
term = build_clinvar_term(gene="BRCA1", clinical_significance="pathogenic")
esearch_result = query_clinvar(term, retmax=100)

# Step 2: Parse IDs
variation_ids = parse_esearch_ids(esearch_result)

# Step 3: Get summaries or full records
summaries = get_clinvar_summary(variation_ids)
records = fetch_clinvar_records(variation_ids, rettype="vcv")
```

### 4. Download Atomic Results

```python
from src.tools.database.ncbi import (
    download_ncbi_clinvar_esearch,
    download_ncbi_clinvar_esummary,
    download_ncbi_clinvar_efetch,
)

# Save esearch results
download_ncbi_clinvar_esearch("BRCA1[gene]", "output/clinvar_esearch.json", retmax=50)

# Save summaries (need IDs from esearch first)
download_ncbi_clinvar_esummary(["12345", "67890"], "output/clinvar_esummary.json")

# Save full records
download_ncbi_clinvar_efetch(["12345", "67890"], "output/clinvar_efetch.xml")
```

### 5. FTP Bulk Downloads (programmatic)

```python
from src.tools.database.ncbi.ncbi_clinvar import get_clinvar_ftp_url, download_clinvar_ftp

# Get FTP URL
url = get_clinvar_ftp_url("variant_summary")  # Or: vcf_grch37, vcf_grch38, xml_latest, var_citations, cross_references

# Download
download_clinvar_ftp("variant_summary", "output/clinvar_ftp/")
download_clinvar_ftp("vcf_grch38", "output/clinvar_ftp/")
```

## Key Concepts

- **Search fields:** `[gene]`, `[CLNSIG]`, `[disorder]`, `[variant name]`, `[chr]`, `[RVSTAT]` (e.g. exclude conflicts: `NOT conflicting[RVSTAT]`)
- **Variation ID (VCV):** ClinVar variation accession; returned by esearch as UID and used in esummary/efetch
- **rettype:** `vcv` (variant-centric) or `rcv` (variant–condition pair) for efetch

## Important Notes

- ClinVar data is not for direct clinical diagnosis; involve a genetics professional
- Prefer high review status (e.g. expert panel, practice guideline) when interpreting classifications
- Use `exclude_conflicting=True` in `build_clinvar_term` to filter out conflicting interpretations
- **Rate limits:** 3 req/s without API key, 10 with key; the module applies delays automatically

## Reference Documentation

- **`references/api_reference.md`** – E-utilities (esearch, esummary, efetch, elink), parameters, rate limits, Biopython example
- **`references/clinical_significance.md`** – Germline/somatic classifications, ACMG/AMP terms, review status, conflict resolution
- **`references/data_formats.md`** – FTP layout, XML (VCV/RCV), VCF, tab-delimited files, processing examples
