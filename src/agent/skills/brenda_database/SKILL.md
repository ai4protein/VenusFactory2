---
name: brenda_database
description: Access BRENDA enzyme database via SOAP API. Retrieve kinetic parameters (Km, kcat), reaction equations, organism data, and substrate-specific enzyme information for biochemical research and metabolic pathway analysis.
license: Unknown
metadata:
    skill-author: VenusFactory.
---

# BRENDA Database

## Overview

BRENDA (BRaunschweig ENzyme DAtabase) is the world's most comprehensive enzyme information system, containing detailed enzyme data from scientific literature. **In this project the agent exposes only download tools**: save Km values, reactions, enzyme-by-substrate results, organism comparison, environmental parameters, kinetic data export, and pathway reports to files; each returns JSON `{success, file_path[, error]}`. For programmatic use, the package also provides query-style APIs (see Project Modules). Access over 45,000 enzymes with millions of kinetic data points for biochemical research, metabolic engineering, and enzyme discovery.

## When to Use This Skill

This skill should be used when:
- Searching for enzyme kinetic parameters (Km, kcat, Vmax)
- Retrieving reaction equations and stoichiometry
- Finding enzymes for specific substrates or reactions
- Comparing enzyme properties across different organisms
- Investigating optimal pH, temperature, and conditions
- Accessing enzyme inhibition and activation data
- Supporting metabolic pathway reconstruction and retrosynthesis
- Performing enzyme engineering and optimization studies
- Analyzing substrate specificity and cofactor requirements

## Quick Start

The skill provides:
1. **Agent tools (download only)** in `src/tools/search/tools_agent.py`: the agent exposes only **download** operations; each returns JSON `{success, file_path[, error]}`. See table below.
2. **Project modules** in `src/tools/database/brenda/`: `brenda_client.py` (SOAP), `brenda_queries.py`, `enzyme_pathway_builder.py`, `brenda_operations.py`; all re-exported via package. For programmatic use (including query-style APIs), import e.g. `from src.tools.database.brenda import download_brenda_km_values_by_ec_number, query_brenda_km_values_by_ec_number, ...`.
3. Reference: `references/api_reference.md`

### Agent Tools (Download Only)

| Tool name | Arguments | Purpose |
|-----------|-----------|---------|
| `download_brenda_km_values_by_ec_number` | `ec_number`, `out_path`, `organism` (optional), `substrate` (optional) | Download Km values by EC number to .json or .txt |
| `download_brenda_reactions_by_ec_number` | `ec_number`, `out_path`, `organism` (optional) | Download reaction equations by EC number to file |
| `download_brenda_enzymes_by_substrate` | `substrate`, `out_path`, `limit` (optional) | Download enzyme-by-substrate search results to JSON |
| `download_brenda_compare_organisms_by_ec_number` | `ec_number`, `organisms`, `out_path` | Download organism comparison by EC number to JSON |
| `download_brenda_environmental_parameters_by_ec_number` | `ec_number`, `out_path` | Download environmental parameters (pH, temperature) by EC number to JSON |
| `download_brenda_kinetic_data_by_ec_number` | `ec_number`, `out_path`, `format` (optional: json/csv) | Download kinetic data export by EC number to file |
| `download_brenda_pathway_report` | `pathway` (dict), `out_path` | Generate and save pathway report from pathway data to file |

All require `BRENDA_EMAIL` and `BRENDA_PASSWORD` in the environment.

### Project Modules (Programmatic Use)

| Capability | Function | Module | Purpose |
|------------|----------|--------|---------|
| SOAP Km | `get_km_values(ec_number, organism=*, substrate=*)` | brenda_client.py | Raw Km entries from BRENDA |
| SOAP reactions | `get_reactions(ec_number, organism=*, reaction=*)` | brenda_client.py | Raw reaction entries |
| Parse Km | `parse_km_entry(entry)` | brenda_queries.py | Parse Km string to dict |
| Parse reaction | `parse_reaction_entry(entry)`, `extract_substrate_products(entry)` | brenda_queries.py | Parse reaction / get substrates & products |
| Organism data | `extract_organism_data(entry)` | brenda_queries.py | Extract organism info from entry |
| Search by substrate | `search_enzymes_by_substrate(substrate, limit)` | brenda_queries.py | Find enzymes for substrate |
| Search by product | `search_enzymes_by_product(product, limit)` | brenda_queries.py | Find enzymes producing product |
| Search by pattern | `search_by_pattern(pattern, limit)` | brenda_queries.py | Find enzymes by reaction pattern |
| Compare organisms | `compare_across_organisms(ec_number, organisms)` | brenda_queries.py | Compare enzyme data across organisms |
| Organisms for enzyme | `get_organisms_for_enzyme(ec_number)` | brenda_queries.py | List organisms with enzyme |
| Environmental params | `get_environmental_parameters(ec_number)` | brenda_queries.py | pH, temperature, stability |
| Cofactors | `get_cofactor_requirements(ec_number)` | brenda_queries.py | Cofactor requirements |
| Substrate specificity | `get_substrate_specificity(ec_number)`, `compare_substrate_affinity(ec_number)` | brenda_queries.py | Substrate affinity data |
| Inhibitors / activators | `get_inhibitors(ec_number)`, `get_activators(ec_number)` | brenda_queries.py | Inhibition / activation |
| Thermophilic / pH variants | `find_thermophilic_homologs(ec_number, min_temp)`, `find_ph_stable_variants(...)` | brenda_queries.py | Engineering variants |
| Modeling params | `get_modeling_parameters(ec_number, substrate)` | brenda_queries.py | Km, kcat, Vmax for modeling |
| Export data | `export_kinetic_data(ec_number, format, filename)` | brenda_queries.py | Export to CSV/JSON/Excel |
| Pathway | `find_pathway_for_product(product, max_steps)`, `build_retrosynthetic_tree(target, depth)` | enzyme_pathway_builder.py | Pathway and retrosynthesis |
| Pathway utils | `suggest_enzyme_substitutions`, `calculate_pathway_feasibility`, `optimize_pathway_conditions`, `generate_pathway_report` | enzyme_pathway_builder.py | Pathway analysis and report |

## Core Capabilities

### 1. Kinetic Parameter Retrieval

Access comprehensive kinetic data for enzymes:

**Get Km Values by EC Number**:
```python
from src.tools.database.brenda import get_km_values

# Get Km values for all organisms
km_data = get_km_values("1.1.1.1")  # Alcohol dehydrogenase

# Get Km values for specific organism
km_data = get_km_values("1.1.1.1", organism="Saccharomyces cerevisiae")

# Get Km values for specific substrate
km_data = get_km_values("1.1.1.1", substrate="ethanol")
```

**Parse Km Results**:
```python
for entry in km_data:
    print(f"Km: {entry}")
    # Example output: "organism*Homo sapiens#substrate*ethanol#kmValue*1.2#commentary*"
```

**Extract Specific Information**:
```python
from src.tools.database.brenda import parse_km_entry, extract_organism_data

for entry in km_data:
    parsed = parse_km_entry(entry)
    organism = extract_organism_data(entry)
    print(f"Organism: {parsed['organism']}")
    print(f"Substrate: {parsed['substrate']}")
    print(f"Km value: {parsed['km_value']}")
    print(f"pH: {parsed.get('ph', 'N/A')}")
    print(f"Temperature: {parsed.get('temperature', 'N/A')}")
```

### 2. Reaction Information

Retrieve reaction equations and details:

**Get Reactions by EC Number**:
```python
from src.tools.database.brenda import get_reactions

# Get all reactions for EC number
reactions = get_reactions("1.1.1.1")

# Filter by organism
reactions = get_reactions("1.1.1.1", organism="Escherichia coli")

# Search specific reaction
reactions = get_reactions("1.1.1.1", reaction="ethanol + NAD+")
```

**Process Reaction Data**:
```python
from src.tools.database.brenda import parse_reaction_entry, extract_substrate_products

for reaction in reactions:
    parsed = parse_reaction_entry(reaction)
    substrates, products = extract_substrate_products(reaction)

    print(f"Reaction: {parsed['reaction']}")
    print(f"Organism: {parsed['organism']}")
    print(f"Substrates: {substrates}")
    print(f"Products: {products}")
```

### 3. Enzyme Discovery

Find enzymes for specific biochemical transformations:

**Find Enzymes by Substrate**:
```python
from src.tools.database.brenda import search_enzymes_by_substrate

# Find enzymes that act on glucose
enzymes = search_enzymes_by_substrate("glucose", limit=20)

for enzyme in enzymes:
    print(f"EC: {enzyme['ec_number']}")
    print(f"Name: {enzyme['enzyme_name']}")
    print(f"Reaction: {enzyme['reaction']}")
```

**Find Enzymes by Product**:
```python
from src.tools.database.brenda import search_enzymes_by_product

# Find enzymes that produce lactate
enzymes = search_enzymes_by_product("lactate", limit=10)
```

**Search by Reaction Pattern**:
```python
from src.tools.database.brenda import search_by_pattern

# Find oxidation reactions
enzymes = search_by_pattern("oxidation", limit=15)
```

### 4. Organism-Specific Enzyme Data

Compare enzyme properties across organisms:

**Get Enzyme Data for Multiple Organisms**:
```python
from src.tools.database.brenda import compare_across_organisms

organisms = ["Escherichia coli", "Saccharomyces cerevisiae", "Homo sapiens"]
comparison = compare_across_organisms("1.1.1.1", organisms)

for org_data in comparison:
    print(f"Organism: {org_data['organism']}")
    print(f"Avg Km: {org_data['average_km']}")
    print(f"Optimal pH: {org_data['optimal_ph']}")
    print(f"Temperature range: {org_data['temperature_range']}")
```

**Find Organisms with Specific Enzyme**:
```python
from src.tools.database.brenda import get_organisms_for_enzyme

organisms = get_organisms_for_enzyme("6.3.5.5")  # Glutamine synthetase
print(f"Found {len(organisms)} organisms with this enzyme")
```

### 5. Environmental Parameters

Access optimal conditions and environmental parameters:

**Get pH and Temperature Data**:
```python
from src.tools.database.brenda import get_environmental_parameters

params = get_environmental_parameters("1.1.1.1")

print(f"Optimal pH range: {params['ph_range']}")
print(f"Optimal temperature: {params['optimal_temperature']}")
print(f"Stability pH: {params['stability_ph']}")
print(f"Temperature stability: {params['temperature_stability']}")
```

**Cofactor Requirements**:
```python
from src.tools.database.brenda import get_cofactor_requirements

cofactors = get_cofactor_requirements("1.1.1.1")
for cofactor in cofactors:
    print(f"Cofactor: {cofactor['name']}")
    print(f"Type: {cofactor['type']}")
    print(f"Concentration: {cofactor['concentration']}")
```

### 6. Substrate Specificity

Analyze enzyme substrate preferences:

**Get Substrate Specificity Data**:
```python
from src.tools.database.brenda import get_substrate_specificity

specificity = get_substrate_specificity("1.1.1.1")

for substrate in specificity:
    print(f"Substrate: {substrate['name']}")
    print(f"Km: {substrate['km']}")
    print(f"Vmax: {substrate['vmax']}")
    print(f"kcat: {substrate['kcat']}")
    print(f"Specificity constant: {substrate['kcat_km_ratio']}")
```

**Compare Substrate Preferences**:
```python
from src.tools.database.brenda import compare_substrate_affinity

comparison = compare_substrate_affinity("1.1.1.1")
sorted_by_km = sorted(comparison, key=lambda x: x['km'])

for substrate in sorted_by_km[:5]:  # Top 5 lowest Km
    print(f"{substrate['name']}: Km = {substrate['km']}")
```

### 7. Inhibition and Activation

Access enzyme regulation data:

**Get Inhibitor Information**:
```python
from src.tools.database.brenda import get_inhibitors

inhibitors = get_inhibitors("1.1.1.1")

for inhibitor in inhibitors:
    print(f"Inhibitor: {inhibitor['name']}")
    print(f"Type: {inhibitor['type']}")
    print(f"Ki: {inhibitor['ki']}")
    print(f"IC50: {inhibitor['ic50']}")
```

**Get Activator Information**:
```python
from src.tools.database.brenda import get_activators

activators = get_activators("1.1.1.1")

for activator in activators:
    print(f"Activator: {activator['name']}")
    print(f"Effect: {activator['effect']}")
    print(f"Mechanism: {activator['mechanism']}")
```

### 8. Enzyme Engineering Support

Find engineering targets and alternatives:

**Find Thermophilic Homologs**:
```python
from src.tools.database.brenda import find_thermophilic_homologs

thermophilic = find_thermophilic_homologs("1.1.1.1", min_temp=50)

for enzyme in thermophilic:
    print(f"Organism: {enzyme['organism']}")
    print(f"Optimal temp: {enzyme['optimal_temperature']}")
    print(f"Km: {enzyme['km']}")
```

**Find Alkaline/ Acid Stable Variants**:
```python
from src.tools.database.brenda import find_ph_stable_variants

alkaline = find_ph_stable_variants("1.1.1.1", min_ph=8.0)
acidic = find_ph_stable_variants("1.1.1.1", max_ph=6.0)
```

### 9. Kinetic Modeling

Prepare data for kinetic modeling:

**Get Kinetic Parameters for Modeling**:
```python
from src.tools.database.brenda import get_modeling_parameters

model_data = get_modeling_parameters("1.1.1.1", substrate="ethanol")

print(f"Km: {model_data['km']}")
print(f"Vmax: {model_data['vmax']}")
print(f"kcat: {model_data['kcat']}")
print(f"Enzyme concentration: {model_data['enzyme_conc']}")
print(f"Temperature: {model_data['temperature']}")
print(f"pH: {model_data['ph']}")
```

## Installation Requirements

```bash
uv pip install zeep requests pandas
```

## Authentication Setup

BRENDA requires authentication credentials:

1. **Create .env file**:
```
BRENDA_EMAIL=your.email@example.com
BRENDA_PASSWORD=your_brenda_password
```

2. **Or set environment variables**:
```bash
export BRENDA_EMAIL="your.email@example.com"
export BRENDA_PASSWORD="your_brenda_password"
```

3. **Register for BRENDA access**:
   - Visit https://www.brenda-enzymes.org/
   - Create an account
   - Check your email for credentials
   - Note: There's also `BRENDA_EMIAL` (note the typo) for legacy support

## Helper Scripts

Scripts live in `src/tools/database/brenda/`. Import from package: `from src.tools.database.brenda import ...`

### brenda_queries.py

Provides high-level functions for enzyme data analysis:

**Key Functions**:
- `parse_km_entry(entry)`: Parse BRENDA Km data entries
- `parse_reaction_entry(entry)`: Parse reaction data entries
- `extract_substrate_products(entry)`: Return (substrates, products) from reaction entry
- `extract_organism_data(entry)`: Extract organism-specific information
- `search_enzymes_by_substrate(substrate, limit)`: Find enzymes for substrates
- `search_enzymes_by_product(product, limit)`: Find enzymes producing products
- `compare_across_organisms(ec_number, organisms)`: Compare enzyme properties
- `get_environmental_parameters(ec_number)`: Get pH and temperature data
- `get_cofactor_requirements(ec_number)`: Get cofactor information
- `get_substrate_specificity(ec_number)`: Analyze substrate preferences
- `get_inhibitors(ec_number)`: Get enzyme inhibition data
- `get_activators(ec_number)`: Get enzyme activation data
- `find_thermophilic_homologs(ec_number, min_temp)`: Find heat-stable variants
- `get_modeling_parameters(ec_number, substrate)`: Get parameters for kinetic modeling
- `export_kinetic_data(ec_number, format, filename)`: Export data to file

**Usage**:
```python
from src.tools.database.brenda import search_enzymes_by_substrate, compare_across_organisms

# Search for enzymes
enzymes = search_enzymes_by_substrate("glucose", limit=20)

# Compare across organisms
comparison = compare_across_organisms("1.1.1.1", ["E. coli", "S. cerevisiae"])
```

### enzyme_pathway_builder.py

Build enzymatic pathways and retrosynthetic routes:

**Key Functions**:
- `find_pathway_for_product(product, max_steps)`: Find enzymatic pathways
- `build_retrosynthetic_tree(target, depth)`: Build retrosynthetic tree
- `suggest_enzyme_substitutions(ec_number, criteria)`: Suggest enzyme alternatives
- `calculate_pathway_feasibility(pathway)`: Evaluate pathway viability
- `optimize_pathway_conditions(pathway)`: Suggest optimal conditions
- `generate_pathway_report(pathway, filename)`: Create detailed pathway report

**Usage**:
```python
from src.tools.database.brenda import find_pathway_for_product, build_retrosynthetic_tree

# Find pathway to product
pathway = find_pathway_for_product("lactate", max_steps=3)

# Build retrosynthetic tree
tree = build_retrosynthetic_tree("lactate", depth=2)
```

## API Rate Limits and Best Practices

**Rate Limits**:
- BRENDA API has moderate rate limiting
- Recommended: 1 request per second for sustained usage
- Maximum: 5 requests per 10 seconds

**Best Practices**:
1. **Cache results**: Store frequently accessed enzyme data locally
2. **Batch queries**: Combine related requests when possible
3. **Use specific searches**: Narrow down by organism, substrate when possible
4. **Handle missing data**: Not all enzymes have complete data
5. **Validate EC numbers**: Ensure EC numbers are in correct format
6. **Implement delays**: Add delays between consecutive requests
7. **Use wildcards wisely**: Use '*' for broader searches when appropriate
8. **Monitor quota**: Track your API usage

**Error Handling**:
```python
from src.tools.database.brenda import get_km_values, get_reactions
from zeep.exceptions import Fault, TransportError  # if using zeep directly

try:
    km_data = get_km_values("1.1.1.1")
except RuntimeError as e:
    print(f"Authentication error: {e}")
except Fault as e:
    print(f"BRENDA API error: {e}")
except TransportError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Common Workflows

### Workflow 1: Enzyme Discovery for New Substrate

Find suitable enzymes for a specific substrate:

```python
from src.tools.database.brenda import get_km_values
from src.tools.database.brenda import search_enzymes_by_substrate, compare_substrate_affinity

# Search for enzymes that act on substrate
substrate = "2-phenylethanol"
enzymes = search_enzymes_by_substrate(substrate, limit=15)

print(f"Found {len(enzymes)} enzymes for {substrate}")
for enzyme in enzymes:
    print(f"EC {enzyme['ec_number']}: {enzyme['enzyme_name']}")

# Get kinetic data for best candidates
if enzymes:
    best_ec = enzymes[0]['ec_number']
    km_data = get_km_values(best_ec, substrate=substrate)

    if km_data:
        print(f"Kinetic data for {best_ec}:")
        for entry in km_data[:3]:  # First 3 entries
            print(f"  {entry}")
```

### Workflow 2: Cross-Organism Enzyme Comparison

Compare enzyme properties across different organisms:

```python
from src.tools.database.brenda import compare_across_organisms, get_environmental_parameters

# Define organisms for comparison
organisms = [
    "Escherichia coli",
    "Saccharomyces cerevisiae",
    "Bacillus subtilis",
    "Thermus thermophilus"
]

# Compare alcohol dehydrogenase
comparison = compare_across_organisms("1.1.1.1", organisms)

print("Cross-organism comparison:")
for org_data in comparison:
    print(f"\n{org_data['organism']}:")
    print(f"  Average Km: {org_data['average_km']}")
    print(f"  Optimal pH: {org_data['optimal_ph']}")
    print(f"  Temperature: {org_data['optimal_temperature']}°C")

# Get detailed environmental parameters
env_params = get_environmental_parameters("1.1.1.1")
print(f"\nOverall optimal pH range: {env_params['ph_range']}")
```

### Workflow 3: Enzyme Engineering Target Identification

Find engineering opportunities for enzyme improvement:

```python
from src.tools.database.brenda import (
    find_thermophilic_homologs,
    find_ph_stable_variants,
    compare_substrate_affinity
)

# Find thermophilic variants for heat stability
thermophilic = find_thermophilic_homologs("1.1.1.1", min_temp=50)
print(f"Found {len(thermophilic)} thermophilic variants")

# Find alkaline-stable variants
alkaline = find_ph_stable_variants("1.1.1.1", min_ph=8.0)
print(f"Found {len(alkaline)} alkaline-stable variants")

# Compare substrate specificities for engineering targets
specificity = compare_substrate_affinity("1.1.1.1")
print("Substrate affinity ranking:")
for i, sub in enumerate(specificity[:5]):
    print(f"  {i+1}. {sub['name']}: Km = {sub['km']}")
```

### Workflow 4: Enzymatic Pathway Construction

Build enzymatic synthesis pathways:

```python
from src.tools.database.brenda import (
    find_pathway_for_product,
    build_retrosynthetic_tree,
    calculate_pathway_feasibility
)

# Find pathway to target product
target = "lactate"
pathway = find_pathway_for_product(target, max_steps=3)

if pathway:
    print(f"Found pathway to {target}:")
    for i, step in enumerate(pathway['steps']):
        print(f"  Step {i+1}: {step['reaction']}")
        print(f"    Enzyme: EC {step['ec_number']}")
        print(f"    Organism: {step['organism']}")

# Evaluate pathway feasibility
feasibility = calculate_pathway_feasibility(pathway)
print(f"\nPathway feasibility score: {feasibility['score']}/10")
print(f"Potential issues: {feasibility['warnings']}")
```

### Workflow 5: Kinetic Parameter Analysis

Comprehensive kinetic analysis for enzyme selection:

```python
from src.tools.database.brenda import get_km_values
from src.tools.database.brenda.brenda_queries import parse_km_entry, get_modeling_parameters

# Get comprehensive kinetic data
ec_number = "1.1.1.1"
km_data = get_km_values(ec_number)

# Analyze kinetic parameters
all_entries = []
for entry in km_data:
    parsed = parse_km_entry(entry)
    if parsed['km_value']:
        all_entries.append(parsed)

print(f"Analyzed {len(all_entries)} kinetic entries")

# Find best kinetic performer
best_km = min(all_entries, key=lambda x: x['km_value'])
print(f"\nBest kinetic performer:")
print(f"  Organism: {best_km['organism']}")
print(f"  Substrate: {best_km['substrate']}")
print(f"  Km: {best_km['km_value']}")

# Get modeling parameters
model_data = get_modeling_parameters(ec_number, substrate=best_km['substrate'])
print(f"\nModeling parameters:")
print(f"  Km: {model_data['km']}")
print(f"  kcat: {model_data['kcat']}")
print(f"  Vmax: {model_data['vmax']}")
```

### Workflow 6: Industrial Enzyme Selection

Select enzymes for industrial applications:

```python
from src.tools.database.brenda import (
    find_thermophilic_homologs,
    get_environmental_parameters,
    get_inhibitors
)

# Industrial criteria: high temperature tolerance, organic solvent resistance
target_enzyme = "1.1.1.1"

# Find thermophilic variants
thermophilic = find_thermophilic_homologs(target_enzyme, min_temp=60)
print(f"Thermophilic candidates: {len(thermophilic)}")

# Check solvent tolerance (inhibitor data)
inhibitors = get_inhibitors(target_enzyme)
solvent_tolerant = [
    inv for inv in inhibitors
    if 'ethanol' not in inv['name'].lower() and
       'methanol' not in inv['name'].lower()
]

print(f"Solvent tolerant candidates: {len(solvent_tolerant)}")

# Evaluate top candidates
for candidate in thermophilic[:3]:
    print(f"\nCandidate: {candidate['organism']}")
    print(f"  Optimal temp: {candidate['optimal_temperature']}°C")
    print(f"  Km: {candidate['km']}")
    print(f"  pH range: {candidate.get('ph_range', 'N/A')}")
```

## Data Formats and Parsing

### BRENDA Response Format

BRENDA returns data in specific formats that need parsing:

**Km Value Format**:
```
organism*Escherichia coli#substrate*ethanol#kmValue*1.2#kmValueMaximum*#commentary*pH 7.4, 25°C#ligandStructureId*#literature*
```

**Reaction Format**:
```
ecNumber*1.1.1.1#organism*Saccharomyces cerevisiae#reaction*ethanol + NAD+ <=> acetaldehyde + NADH + H+#commentary*#literature*
```

### Data Extraction Patterns

```python
import re

def parse_brenda_field(data, field_name):
    """Extract specific field from BRENDA data entry"""
    pattern = f"{field_name}\\*([^#]*)"
    match = re.search(pattern, data)
    return match.group(1) if match else None

def extract_multiple_values(data, field_name):
    """Extract multiple values for a field"""
    pattern = f"{field_name}\\*([^#]*)"
    matches = re.findall(pattern, data)
    return [match for match in matches if match.strip()]
```

## Reference Documentation

For detailed BRENDA documentation, see `references/api_reference.md`. This includes:
- Complete SOAP API method documentation
- Full parameter lists and formats
- EC number structure and validation
- Response format specifications
- Error codes and handling
- Data field definitions
- Literature citation formats

## Troubleshooting

**Authentication Errors**:
- Verify BRENDA_EMAIL and BRENDA_PASSWORD in .env file
- Check for correct spelling (note BRENDA_EMIAL legacy support)
- Ensure BRENDA account is active and has API access

**No Results Returned**:
- Try broader searches with wildcards (*)
- Check EC number format (e.g., "1.1.1.1" not "1.1.1")
- Verify substrate spelling and naming
- Some enzymes may have limited data in BRENDA

**Rate Limiting**:
- Add delays between requests (0.5-1 second)
- Cache results locally
- Use more specific queries to reduce data volume
- Consider batch operations for multiple queries

**Network Errors**:
- Check internet connection
- BRENDA server may be temporarily unavailable
- Try again after a few minutes
- Consider using VPN if geo-restricted

**Data Format Issues**:
- Use the provided parsing functions in scripts
- BRENDA data can be inconsistent in formatting
- Handle missing fields gracefully
- Validate parsed data before use

**Performance Issues**:
- Large queries can be slow; limit search scope
- Use specific organism or substrate filters
- Consider asynchronous processing for batch operations
- Monitor memory usage with large datasets

## Additional Resources

- BRENDA Home: https://www.brenda-enzymes.org/
- BRENDA SOAP API Documentation: https://www.brenda-enzymes.org/soap.php
- Enzyme Commission (EC) Numbers: https://www.qmul.ac.uk/sbcs/iubmb/enzyme/
- Zeep SOAP Client: https://python-zeep.readthedocs.io/
- Enzyme Nomenclature: https://www.iubmb.org/enzyme/