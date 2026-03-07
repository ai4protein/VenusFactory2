---
name: kegg_database
description: Direct REST API access to KEGG (academic use only). Pathway analysis, gene-pathway mapping, metabolic pathways, drug interactions, ID conversion. Use this for direct HTTP/REST work or KEGG-specific control.
license: Non-academic use of KEGG requires a commercial license
metadata:
    skill-author: VenusFactory.
---

# KEGG Database

## Overview

KEGG (Kyoto Encyclopedia of Genes and Genomes) is a comprehensive bioinformatics resource for biological pathway analysis and molecular interaction networks. **In this project the agent exposes only download tools**: save database info, entry lists, search results, entry data, ID conversions, cross-references, and drug-drug interactions to files; each returns rich JSON `{status, file_info, content_preview, biological_metadata, execution_context}`. For programmatic use, the package also provides query-style APIs (see Project Modules).

**Important**: KEGG API is made available only for academic use by academic users.

## When to Use This Skill

This skill should be used when querying pathways, genes, compounds, enzymes, diseases, and drugs across multiple organisms using KEGG's REST API.

## Quick Start

The skill provides:
1. **Project modules** in `src/tools/database/kegg/`: `kegg_rest.py` (base HTTP client), `kegg_operations.py` (query/download operations), `kegg_api.py` (backward-compat re-exports); all download functions re-exported via package. For programmatic use, import e.g. `from src.tools.database.kegg import download_kegg_entry_by_id, ...`.
2. Reference: `references/kegg_reference.md`

### Agent Tools (Download Only)

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_kegg_info_by_database` | `database`, `out_path` | Download KEGG database info/statistics to file |
| `download_kegg_list_by_database` | `database`, `out_path`, `org_or_ids` (optional) | Download KEGG entry list by database to file |
| `download_kegg_find_by_database` | `database`, `query`, `out_path`, `option` (optional) | Download KEGG search results to file |
| `download_kegg_entry_by_id` | `entry_id`, `out_path`, `format` (optional) | Download KEGG entry data by entry ID to file |
| `download_kegg_conv_by_id` | `target_db`, `source_id`, `out_path` | Download KEGG ID conversion result to file |
| `download_kegg_link_by_id` | `target_db`, `source_id`, `out_path` | Download KEGG cross-reference links to file |
| `download_kegg_ddi_by_id` | `drug_id`, `out_path` | Download KEGG drug-drug interaction data to file |

All return rich JSON: `{status, file_info, content_preview, biological_metadata, execution_context}`. Academic use only.

### Project Modules (Programmatic Use)

| Capability | Function | Module | Purpose |
|------------|----------|--------|---------|
| HTTP client | `kegg_request(operation, *path_parts)` | kegg_rest.py | Base REST GET request, returns text |
| ID helper | `_join_ids(entry_id)` | kegg_rest.py | Format one or multiple IDs for URL (max 10) |
| Query: info | `query_kegg_info_by_database(database)` | kegg_operations.py | Returns rich JSON in memory |
| Query: list | `query_kegg_list_by_database(database, org_or_ids)` | kegg_operations.py | Returns rich JSON in memory |
| Query: find | `query_kegg_find_by_database(database, query, option)` | kegg_operations.py | Returns rich JSON in memory |
| Query: entry | `query_kegg_entry_by_id(entry_id, format)` | kegg_operations.py | Returns rich JSON in memory |
| Query: conv | `query_kegg_conv_by_id(target_db, source_id)` | kegg_operations.py | Returns rich JSON in memory |
| Query: link | `query_kegg_link_by_id(target_db, source_id)` | kegg_operations.py | Returns rich JSON in memory |
| Query: ddi | `query_kegg_ddi_by_id(drug_id)` | kegg_operations.py | Returns rich JSON in memory |
| Download: info | `download_kegg_info_by_database(database, out_path)` | kegg_operations.py | Save to file, return rich JSON |
| Download: list | `download_kegg_list_by_database(database, out_path, org_or_ids)` | kegg_operations.py | Save to file, return rich JSON |
| Download: find | `download_kegg_find_by_database(database, query, out_path, option)` | kegg_operations.py | Save to file, return rich JSON |
| Download: entry | `download_kegg_entry_by_id(entry_id, out_path, format)` | kegg_operations.py | Save to file, return rich JSON |
| Download: conv | `download_kegg_conv_by_id(target_db, source_id, out_path)` | kegg_operations.py | Save to file, return rich JSON |
| Download: link | `download_kegg_link_by_id(target_db, source_id, out_path)` | kegg_operations.py | Save to file, return rich JSON |
| Download: ddi | `download_kegg_ddi_by_id(drug_id, out_path)` | kegg_operations.py | Save to file, return rich JSON |
| Compat alias | `kegg_info`, `kegg_list`, `kegg_find`, `kegg_get`, `kegg_conv`, `kegg_link`, `kegg_ddi` | kegg_operations.py | Backward-compat aliases for query functions |

## Core Capabilities

### 1. Database Information

**Download database info:**
```python
from src.tools.database.kegg import download_kegg_info_by_database

result = download_kegg_info_by_database("pathway", "output/kegg_info_pathway.txt")
```

**Query in-memory:**
```python
from src.tools.database.kegg.kegg_operations import query_kegg_info_by_database

result = query_kegg_info_by_database("pathway")
```

**Common databases**: `kegg`, `pathway`, `module`, `brite`, `genes`, `genome`, `compound`, `glycan`, `reaction`, `enzyme`, `disease`, `drug`

### 2. Listing Entries

**Download entry list:**
```python
from src.tools.database.kegg import download_kegg_list_by_database

# List all reference pathways
result = download_kegg_list_by_database("pathway", "output/kegg_pathways.txt")

# List human-specific pathways
result = download_kegg_list_by_database("pathway", "output/kegg_hsa_pathways.txt", org_or_ids="hsa")
```

**Common organism codes**: `hsa` (human), `mmu` (mouse), `dme` (fruit fly), `sce` (yeast), `eco` (E. coli)

### 3. Searching

**Download search results:**
```python
from src.tools.database.kegg import download_kegg_find_by_database

# Keyword search
result = download_kegg_find_by_database("genes", "p53", "output/kegg_find_p53.txt")

# Chemical formula search (exact match)
result = download_kegg_find_by_database("compound", "C7H10N4O2", "output/kegg_find_formula.txt", option="formula")

# Molecular weight range search
result = download_kegg_find_by_database("drug", "300-310", "output/kegg_find_mass.txt", option="exact_mass")
```

**Search options**: `formula` (exact match), `exact_mass` (range), `mol_weight` (range)

### 4. Retrieving Entries

**Download entry data:**
```python
from src.tools.database.kegg import download_kegg_entry_by_id

# Get pathway entry
result = download_kegg_entry_by_id("hsa00010", "output/kegg_glycolysis.txt")

# Get protein sequence (FASTA)
result = download_kegg_entry_by_id("hsa:10458", "output/kegg_gene_aaseq.fasta", format="aaseq")

# Get compound structure
result = download_kegg_entry_by_id("cpd:C00002", "output/kegg_atp.mol", format="mol")

# Get pathway as JSON (single entry only)
result = download_kegg_entry_by_id("hsa05130", "output/kegg_pathway.json", format="json")
```

**Output formats**: `aaseq` (protein FASTA), `ntseq` (nucleotide FASTA), `mol` (MOL format), `kcf` (KCF format), `image` (PNG), `kgml` (XML), `json` (pathway JSON)

**Important**: Image, KGML, and JSON formats allow only one entry at a time.

### 5. ID Conversion

**Download ID conversion:**
```python
from src.tools.database.kegg import download_kegg_conv_by_id

# Convert KEGG gene to NCBI Gene ID
result = download_kegg_conv_by_id("ncbi-geneid", "hsa:10458", "output/kegg_conv.txt")

# Convert to UniProt
result = download_kegg_conv_by_id("uniprot", "hsa:10458", "output/kegg_conv_uniprot.txt")
```

**Supported conversions**: `ncbi-geneid`, `ncbi-proteinid`, `uniprot`, `pubchem`, `chebi`

### 6. Cross-Referencing

**Download cross-references:**
```python
from src.tools.database.kegg import download_kegg_link_by_id

# Get genes in a specific pathway
result = download_kegg_link_by_id("genes", "hsa00010", "output/kegg_link_glycolysis.txt")

# Find pathways containing a specific gene
result = download_kegg_link_by_id("pathway", "hsa:10458", "output/kegg_link_gene.txt")

# Find compounds in a pathway
result = download_kegg_link_by_id("compound", "hsa00010", "output/kegg_link_compound.txt")
```

**Common links**: genes ↔ pathway, pathway ↔ compound, pathway ↔ enzyme, genes ↔ ko (orthology)

### 7. Drug-Drug Interactions

**Download DDI data:**
```python
from src.tools.database.kegg import download_kegg_ddi_by_id

result = download_kegg_ddi_by_id("D00001", "output/kegg_ddi.txt")
```

## Common Workflows

### Workflow 1: Gene to Pathway Mapping (download)

```python
from src.tools.database.kegg import (
    download_kegg_find_by_database,
    download_kegg_link_by_id,
    download_kegg_entry_by_id,
)

# Step 1: Find gene by name
download_kegg_find_by_database("genes", "p53", "output/kegg_p53_genes.txt")

# Step 2: Link gene to pathways
download_kegg_link_by_id("pathway", "hsa:7157", "output/kegg_p53_pathways.txt")

# Step 3: Get pathway details
download_kegg_entry_by_id("hsa05200", "output/kegg_cancer_pathway.txt")
```

### Workflow 2: Compound to Pathway Analysis (download)

```python
from src.tools.database.kegg import (
    download_kegg_find_by_database,
    download_kegg_link_by_id,
    download_kegg_entry_by_id,
)

# Step 1: Search for compound
download_kegg_find_by_database("compound", "glucose", "output/kegg_glucose.txt")

# Step 2: Link compound to reactions/pathways
download_kegg_link_by_id("reaction", "cpd:C00031", "output/kegg_glucose_reactions.txt")
download_kegg_link_by_id("pathway", "rn:R00299", "output/kegg_reaction_pathways.txt")

# Step 3: Get pathway details
download_kegg_entry_by_id("map00010", "output/kegg_glycolysis.txt")
```

### Workflow 3: Cross-Database Integration (download)

```python
from src.tools.database.kegg import (
    download_kegg_conv_by_id,
    download_kegg_entry_by_id,
)

# Step 1: Convert KEGG gene IDs to external database IDs
download_kegg_conv_by_id("uniprot", "hsa:10458", "output/kegg_to_uniprot.txt")
download_kegg_conv_by_id("ncbi-geneid", "hsa:10458", "output/kegg_to_ncbi.txt")

# Step 2: Get sequences using KEGG
download_kegg_entry_by_id("hsa:10458", "output/kegg_gene_seq.fasta", format="aaseq")
```

### Workflow 4: Organism-Specific Pathway Analysis (programmatic)

```python
from src.tools.database.kegg.kegg_operations import (
    query_kegg_list_by_database,
    query_kegg_entry_by_id,
)

# List pathways for multiple organisms
human_pathways = query_kegg_list_by_database("pathway", "hsa")
mouse_pathways = query_kegg_list_by_database("pathway", "mmu")

# Get organism-specific pathway details
hsa_glycolysis = query_kegg_entry_by_id("hsa00010")
mmu_glycolysis = query_kegg_entry_by_id("mmu00010")
```

## Response Format

### Download Response (success)
```json
{
  "status": "success",
  "file_info": {
    "file_path": "/absolute/path/to/file.txt",
    "file_name": "file.txt",
    "file_size": 12345,
    "format": "txt"
  },
  "content_preview": "first 500 chars...",
  "biological_metadata": {"database": "pathway"},
  "execution_context": {"download_time_ms": 234, "source": "KEGG"}
}
```

### Query Response (success)
```json
{
  "status": "success",
  "content": "{...full JSON...}",
  "content_preview": "first 500 chars...",
  "biological_metadata": {"database": "pathway"},
  "execution_context": {"query_time_ms": 123, "source": "KEGG"}
}
```

### Error Response
```json
{
  "status": "error",
  "error": {"type": "QueryError", "message": "...", "suggestion": "..."},
  "file_info": null
}
```

## Pathway Categories

KEGG organizes pathways into seven major categories:

1. **Metabolism** (e.g., `map00010` - Glycolysis, `map00190` - Oxidative phosphorylation)
2. **Genetic Information Processing** (e.g., `map03010` - Ribosome, `map03040` - Spliceosome)
3. **Environmental Information Processing** (e.g., `map04010` - MAPK signaling, `map02010` - ABC transporters)
4. **Cellular Processes** (e.g., `map04140` - Autophagy, `map04210` - Apoptosis)
5. **Organismal Systems** (e.g., `map04610` - Complement cascade, `map04910` - Insulin signaling)
6. **Human Diseases** (e.g., `map05200` - Pathways in cancer, `map05010` - Alzheimer disease)
7. **Drug Development** (chronological and target-based classifications)

Reference `references/kegg_reference.md` for detailed pathway lists and classifications.

## Important Identifiers and Formats

| Type | Format | Example |
|------|--------|---------|
| Pathway (reference) | `map#####` | `map00010` |
| Pathway (human) | `hsa#####` | `hsa00010` |
| Gene | `organism:gene_number` | `hsa:10458` |
| Compound | `cpd:C#####` | `cpd:C00002` (ATP) |
| Drug | `dr:D#####` | `dr:D00001` |
| Enzyme | `ec:EC_number` | `ec:1.1.1.1` |
| KO (Orthology) | `ko:K#####` | `ko:K00001` |

## Helper Scripts

Scripts live in `src/tools/database/kegg/`. Import from package: `from src.tools.database.kegg import ...`

### kegg_operations.py

Central operations module providing both query and download functions:

- `query_kegg_info_by_database(database)` — returns rich JSON in memory
- `query_kegg_list_by_database(database, org_or_ids)` — returns rich JSON in memory
- `query_kegg_find_by_database(database, query, option)` — returns rich JSON in memory
- `query_kegg_entry_by_id(entry_id, format)` — returns rich JSON in memory
- `query_kegg_conv_by_id(target_db, source_id)` — returns rich JSON in memory
- `query_kegg_link_by_id(target_db, source_id)` — returns rich JSON in memory
- `query_kegg_ddi_by_id(drug_id)` — returns rich JSON in memory
- `download_kegg_info_by_database(database, out_path)` — save to file, return rich JSON
- `download_kegg_list_by_database(database, out_path, org_or_ids)` — save to file, return rich JSON
- `download_kegg_find_by_database(database, query, out_path, option)` — save to file, return rich JSON
- `download_kegg_entry_by_id(entry_id, out_path, format)` — save to file, return rich JSON
- `download_kegg_conv_by_id(target_db, source_id, out_path)` — save to file, return rich JSON
- `download_kegg_link_by_id(target_db, source_id, out_path)` — save to file, return rich JSON
- `download_kegg_ddi_by_id(drug_id, out_path)` — save to file, return rich JSON

Backward-compat aliases: `kegg_info`, `kegg_list`, `kegg_find`, `kegg_get`, `kegg_conv`, `kegg_link`, `kegg_ddi`.

**Test**: `bash script/tools/database/test_kegg.sh` — runs `kegg_operations.py --test`, outputs under `example/database/kegg/`.

### kegg_rest.py

Base HTTP client:
- `kegg_request(operation, *path_parts)` — GET request to `https://rest.kegg.jp/`, returns response text
- `_join_ids(entry_id)` — format one or multiple entry IDs for URL (max 10, `+` separated)

### kegg_api.py

Backward-compat entry point: re-exports all query/download functions and legacy aliases.

## API Limitations

1. **Entry limits**: Maximum 10 entries per operation (except image/kgml/json: 1 entry only)
2. **Academic use**: API is for academic use only; commercial use requires licensing
3. **HTTP status codes**: Check for 200 (success), 400 (bad request), 404 (not found)
4. **Rate limiting**: No explicit limit, but avoid rapid-fire requests

## Troubleshooting

**404 Not Found**: Entry or database doesn't exist; verify IDs and organism codes
**400 Bad Request**: Syntax error in API call; check parameter formatting
**Empty results**: Search term may not match entries; try broader keywords
**Image/KGML errors**: These formats only work with single entries; remove batch processing

## Resources

### references/kegg_reference.md

Comprehensive API documentation including complete database list, operation syntax, all organism codes, HTTP status codes, and integration with Biopython/R.

### External

- KEGG website: https://www.kegg.jp/
- KEGG Mapper: https://www.kegg.jp/kegg/mapper/
- BlastKOALA: Automated genome annotation
- GhostKOALA: Metagenome/metatranscriptome annotation
