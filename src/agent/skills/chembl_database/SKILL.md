---
name: chembl_database
description: Query ChEMBL bioactive molecules and drug discovery data. Search compounds by structure/properties, retrieve bioactivity data (IC50, Ki), find inhibitors, perform SAR studies, for medicinal chemistry.
license: Unknown
metadata:
    skill-author: VenusFactory.
---

# ChEMBL Database

## Overview

ChEMBL is a manually curated database of bioactive molecules maintained by the European Bioinformatics Institute (EBI), containing over 2 million compounds, 19 million bioactivity measurements, 13,000+ drug targets, and data on approved drugs and clinical candidates. **In this project the agent exposes only download tools**: save molecule data, similarity search results, substructure search results, and drug information to files; each returns rich JSON `{status, file_info, content_preview, biological_metadata, execution_context}`. For programmatic use, the package also provides query-style APIs (see Project Modules).

## When to Use This Skill

This skill should be used when:

- **Compound searches**: Finding molecules by ChEMBL ID or name
- **Target information**: Retrieving data about proteins, enzymes, or biological targets
- **Bioactivity data**: Querying IC50, Ki, EC50, or other activity measurements
- **Drug information**: Looking up approved drugs, mechanisms, or indications
- **Structure searches**: Performing similarity or substructure searches by SMILES
- **Cheminformatics**: Analyzing molecular properties and drug-likeness
- **Target-ligand relationships**: Exploring compound-target interactions
- **Drug discovery**: Identifying inhibitors, agonists, or bioactive molecules

## Quick Start

The skill provides:
1. **Project modules** in `src/tools/search/database/chembl/`: `chembl_client.py` (API client), `chembl_molecule.py`, `chembl_target.py`, `chembl_activity.py`, `chembl_similarity.py`, `chembl_substructure.py`, `chembl_drug.py` (atomic modules), `chembl_queries.py` (high-level helpers), `chembl_operations.py` (query/download operations); all re-exported via package. For programmatic use (including query-style APIs), import e.g. `from src.tools.search.database.chembl import download_chembl_molecule_by_id, ...`.
2. Reference: `references/api_reference.md`

### Agent Tools (Download Only)

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_chembl_molecule_by_id` | `mol_id`, `out_path` | Download molecule JSON by ChEMBL ID to file |
| `download_chembl_similarity_by_smiles` | `smiles`, `out_path`, `threshold` (optional, 0–100, default 70), `max_results` (optional) | Download Tanimoto similarity search results to JSON file |
| `download_chembl_substructure_by_smiles` | `smiles`, `out_path`, `max_results` (optional) | Download substructure search results to JSON file |
| `download_chembl_drug_by_id` | `chembl_id`, `out_path`, `max_results` (optional) | Download drug info (drug, mechanisms, indications) to JSON file |

No authentication required; ChEMBL API is publicly accessible.

### Project Modules (Programmatic Use)

| Capability | Function | Module | Purpose |
|------------|----------|--------|---------|
| Client | `get_client()` | chembl_client.py | Return ChEMBL `new_client` singleton |
| Molecule by ID | `get_molecule(chembl_id)` | chembl_molecule.py | Get molecule by ChEMBL ID |
| Molecule filter | `filter_molecules(**kwargs)` | chembl_molecule.py | Filter by name/properties (Django-style) |
| Target by ID | `get_target(chembl_id)` | chembl_target.py | Get target by ChEMBL ID |
| Target filter | `filter_targets(**kwargs)` | chembl_target.py | Filter targets by type/name |
| Activity filter | `filter_activities(**kwargs)` | chembl_activity.py | Filter bioactivities (target, type, value) |
| Similarity | `similarity_search(smiles, threshold=85)` | chembl_similarity.py | Find similar compounds by SMILES |
| Substructure | `substructure_search(smiles)` | chembl_substructure.py | Find compounds containing substructure |
| Drug / mechanism / indication | `get_drug(id)`, `get_mechanisms(mol_id)`, `get_indications(mol_id)` | chembl_drug.py | Drug record, MoA, indications |
| Query: molecule | `query_chembl_molecule_by_id(chembl_id)` | chembl_operations.py | Returns rich JSON in memory |
| Query: similarity | `query_chembl_similarity_by_smiles(smiles, threshold, max_results)` | chembl_operations.py | Returns rich JSON in memory |
| Query: substructure | `query_chembl_substructure_by_smiles(smiles, max_results)` | chembl_operations.py | Returns rich JSON in memory |
| Query: drug | `query_chembl_drug_by_id(chembl_id, max_results)` | chembl_operations.py | Returns rich JSON in memory |
| Download: molecule | `download_chembl_molecule_by_id(chembl_id, out_path)` | chembl_operations.py | Save molecule JSON to file, return rich JSON |
| Download: similarity | `download_chembl_similarity_by_smiles(smiles, out_path, threshold, max_results)` | chembl_operations.py | Save similarity results to file, return rich JSON |
| Download: substructure | `download_chembl_substructure_by_smiles(smiles, out_path, max_results)` | chembl_operations.py | Save substructure results to file, return rich JSON |
| Download: drug | `download_chembl_drug_by_id(chembl_id, out_path, max_results)` | chembl_operations.py | Save drug info to file, return rich JSON |
| High-level: molecule | `get_molecule_info(id)`, `search_molecules_by_name(name)`, `find_molecules_by_properties(...)` | chembl_queries.py | Convenience molecule helpers |
| High-level: target | `get_target_info(id)`, `search_targets_by_name(name)` | chembl_queries.py | Convenience target helpers |
| High-level: activity / drug | `get_bioactivity_data(...)`, `get_compound_bioactivities(mol_id)`, `get_drug_info(mol_id)`, `find_kinase_inhibitors(...)` | chembl_queries.py | Bioactivity and drug helpers |
| High-level: structure | `find_similar_compounds(smiles, threshold)` | chembl_queries.py | Similarity wrapper |
| Export | `export_to_dataframe(data)` | chembl_queries.py | Convert results to pandas DataFrame |

## Installation and Setup

### Python Client

```bash
uv pip install chembl_webresource_client
```

Optional for export: `uv pip install pandas`

## Core Capabilities

### 1. Molecule Download

**Download molecule by ChEMBL ID:**
```python
from src.tools.search.database.chembl import download_chembl_molecule_by_id

result = download_chembl_molecule_by_id("CHEMBL25", "output/chembl_aspirin.json")
# Returns rich JSON: {status, file_info, content_preview, biological_metadata, execution_context}
```

**Query molecule (in-memory, no file):**
```python
from src.tools.search.database.chembl.chembl_operations import query_chembl_molecule_by_id

result = query_chembl_molecule_by_id("CHEMBL25")
# Returns rich JSON: {status, content, content_preview, biological_metadata, execution_context}
```

### 2. Similarity Search

**Download similar compounds by SMILES:**
```python
from src.tools.search.database.chembl import download_chembl_similarity_by_smiles

result = download_chembl_similarity_by_smiles(
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin SMILES
    "output/chembl_similarity.json",
    threshold=70,              # Tanimoto threshold 0-100
    max_results=100,           # Optional limit
)
```

**Query similarity (in-memory):**
```python
from src.tools.search.database.chembl.chembl_operations import query_chembl_similarity_by_smiles

result = query_chembl_similarity_by_smiles("CC(=O)Oc1ccccc1C(=O)O", threshold=85)
```

### 3. Substructure Search

**Download substructure matches:**
```python
from src.tools.search.database.chembl import download_chembl_substructure_by_smiles

result = download_chembl_substructure_by_smiles(
    "c1ccccc1",               # Benzene substructure
    "output/chembl_substructure.json",
    max_results=50,
)
```

**Query substructure (in-memory):**
```python
from src.tools.search.database.chembl.chembl_operations import query_chembl_substructure_by_smiles

result = query_chembl_substructure_by_smiles("c1ccccc1", max_results=50)
```

### 4. Drug Information

**Download drug info (drug, mechanisms, indications):**
```python
from src.tools.search.database.chembl import download_chembl_drug_by_id

result = download_chembl_drug_by_id(
    "CHEMBL25",
    "output/chembl_drug.json",
    max_results=100,
)
```

**Query drug info (in-memory):**
```python
from src.tools.search.database.chembl.chembl_operations import query_chembl_drug_by_id

result = query_chembl_drug_by_id("CHEMBL25")
```

### 5. Low-Level Atomic Functions

For direct programmatic access (not exposed as agent tools):

```python
from src.tools.search.database.chembl import (
    get_client, get_molecule, filter_molecules,
    get_target, filter_targets,
    filter_activities,
    similarity_search, substructure_search,
    get_drug, get_mechanisms, get_indications,
)

# Client
client = get_client()

# Molecule
aspirin = get_molecule("CHEMBL25")
results = filter_molecules(pref_name__icontains="aspirin")

# Target
egfr = get_target("CHEMBL203")
kinases = filter_targets(target_type="SINGLE PROTEIN", pref_name__icontains="kinase")

# Activities
activities = filter_activities(
    target_chembl_id="CHEMBL203",
    standard_type="IC50",
    standard_value__lte=100,
)

# Similarity / substructure
similar = similarity_search("CC(=O)Oc1ccccc1C(=O)O", threshold=85)
sub_results = substructure_search("c1ccccc1")

# Drug
drug = get_drug("CHEMBL25")
mechanisms = get_mechanisms("CHEMBL25")
indications = get_indications("CHEMBL25")
```

## Common Workflows

### Workflow 1: Analyzing a Known Drug (download)

```python
from src.tools.search.database.chembl import (
    download_chembl_molecule_by_id,
    download_chembl_drug_by_id,
    download_chembl_similarity_by_smiles,
)

# Step 1: Download molecule data
mol_result = download_chembl_molecule_by_id("CHEMBL25", "output/aspirin_molecule.json")

# Step 2: Download drug info (mechanisms, indications)
drug_result = download_chembl_drug_by_id("CHEMBL25", "output/aspirin_drug.json")

# Step 3: Find similar compounds
sim_result = download_chembl_similarity_by_smiles(
    "CC(=O)Oc1ccccc1C(=O)O", "output/aspirin_similar.json", threshold=80
)
```

### Workflow 2: Structure-Activity Relationship (SAR) Study

```python
from src.tools.search.database.chembl import (
    download_chembl_similarity_by_smiles,
    download_chembl_substructure_by_smiles,
)

# Find similar compounds for SAR
download_chembl_similarity_by_smiles(
    "query_smiles_here", "output/sar_similar.json", threshold=70
)

# Find compounds sharing a core structure
download_chembl_substructure_by_smiles(
    "core_smiles_here", "output/sar_substructure.json", max_results=100
)
```

### Workflow 3: Finding Inhibitors (programmatic)

```python
from src.tools.search.database.chembl.chembl_molecule import get_molecule
from src.tools.search.database.chembl.chembl_target import filter_targets
from src.tools.search.database.chembl.chembl_activity import filter_activities

targets = filter_targets(pref_name__icontains="EGFR")
target_id = targets[0]["target_chembl_id"] if targets else None
if target_id:
    activities = filter_activities(
        target_chembl_id=target_id,
        standard_type="IC50",
        standard_value__lte=100,
    )
    compound_ids = [act["molecule_chembl_id"] for act in activities]
    compounds = [get_molecule(cid) for cid in compound_ids[:10]]
```

## Filter Operators

ChEMBL supports Django-style query filters for the low-level atomic functions:

- `__exact` - Exact match
- `__iexact` - Case-insensitive exact match
- `__contains` / `__icontains` - Substring matching
- `__startswith` / `__endswith` - Prefix/suffix matching
- `__gt`, `__gte`, `__lt`, `__lte` - Numeric comparisons
- `__range` - Value in range
- `__in` - Value in list
- `__isnull` - Null/not null check

## Response Format

### Download Response (success)
```json
{
  "status": "success",
  "file_info": {
    "file_path": "/absolute/path/to/file.json",
    "file_name": "file.json",
    "file_size": 12345,
    "format": "json"
  },
  "content_preview": "first 500 chars...",
  "biological_metadata": {"chembl_id": "CHEMBL25"},
  "execution_context": {"download_time_ms": 234, "source": "ChEMBL"}
}
```

### Query Response (success)
```json
{
  "status": "success",
  "content": "{...full JSON...}",
  "content_preview": "first 500 chars...",
  "biological_metadata": {"chembl_id": "CHEMBL25"},
  "execution_context": {"query_time_ms": 123, "source": "ChEMBL"}
}
```

### Error Response
```json
{
  "status": "error",
  "error": {"type": "NotFound", "message": "...", "suggestion": "..."},
  "file_info": null
}
```

## Helper Scripts

Scripts live in `src/tools/search/database/chembl/`. Import from package: `from src.tools.search.database.chembl import ...`

### chembl_operations.py

Central operations module providing both query and download functions:

- `query_chembl_molecule_by_id(chembl_id)` — returns rich JSON in memory
- `query_chembl_similarity_by_smiles(smiles, threshold, max_results)` — returns rich JSON in memory
- `query_chembl_substructure_by_smiles(smiles, max_results)` — returns rich JSON in memory
- `query_chembl_drug_by_id(chembl_id, max_results)` — returns rich JSON in memory
- `download_chembl_molecule_by_id(chembl_id, out_path)` — save to file, return rich JSON
- `download_chembl_similarity_by_smiles(smiles, out_path, threshold, max_results)` — save to file, return rich JSON
- `download_chembl_substructure_by_smiles(smiles, out_path, max_results)` — save to file, return rich JSON
- `download_chembl_drug_by_id(chembl_id, out_path, max_results)` — save to file, return rich JSON

**Test**: `bash script/tools/search/database/test_chembl.sh` — runs `chembl_operations.py --test`, outputs under `example/database/chembl/`.

### Atomic Modules

- `chembl_client.py` — `get_client()`: ChEMBL `new_client` singleton
- `chembl_molecule.py` — `get_molecule(chembl_id)`, `filter_molecules(**kwargs)`
- `chembl_target.py` — `get_target(chembl_id)`, `filter_targets(**kwargs)`
- `chembl_activity.py` — `filter_activities(**kwargs)`
- `chembl_similarity.py` — `similarity_search(smiles, threshold, max_results)`
- `chembl_substructure.py` — `substructure_search(smiles, max_results)`
- `chembl_drug.py` — `get_drug(chembl_id)`, `get_mechanisms(mol_id, max_results)`, `get_indications(mol_id, max_results)`

### chembl_queries.py (High-Level Helpers)

- `get_molecule_info(id)`, `search_molecules_by_name(name)`, `find_molecules_by_properties(...)`
- `get_target_info(id)`, `search_targets_by_name(name)`
- `get_bioactivity_data(...)`, `get_compound_bioactivities(mol_id)`, `get_drug_info(mol_id)`, `find_kinase_inhibitors(...)`
- `find_similar_compounds(smiles, threshold)`
- `export_to_dataframe(data)` — convert results to pandas DataFrame

## Performance Optimization

### Caching

The client automatically caches results for 24 hours. Configure:

```python
from chembl_webresource_client.settings import Settings

Settings.Instance().CACHING = False           # Disable caching
Settings.Instance().CACHE_EXPIRE = 86400      # Adjust expiration (seconds)
```

### Lazy Evaluation

Queries execute only when data is accessed. Convert to list to force execution:

```python
results = filter_molecules(pref_name__icontains='aspirin')
results_list = list(results)  # Force execution
```

## Important Notes

### Data Reliability
- ChEMBL data is manually curated but may contain inconsistencies
- Always check `data_validity_comment` field in activity records
- Be aware of `potential_duplicate` flags

### Units and Standards
- Bioactivity values use standard units (nM, uM, etc.)
- `pchembl_value` provides normalized activity (-log scale)
- Check `standard_type` to understand measurement type (IC50, Ki, EC50, etc.)

### Rate Limiting
- Respect ChEMBL's fair usage policies
- Use caching to minimize repeated requests
- Consider bulk downloads for large datasets

### Chemical Structure Formats
- SMILES strings are the primary structure format
- InChI keys available for compounds
- SVG images can be generated via the image endpoint

## Resources

### references/api_reference.md

Comprehensive API documentation including complete endpoint listing, filter operators, molecular properties, and advanced query examples.

### External

- ChEMBL website: https://www.ebi.ac.uk/chembl/
- API documentation: https://www.ebi.ac.uk/chembl/api/data/docs
- Python client GitHub: https://github.com/chembl/chembl_webresource_client
- Interface documentation: https://chembl.gitbook.io/chembl-interface-documentation/
- Example notebooks: https://github.com/chembl/notebooks
