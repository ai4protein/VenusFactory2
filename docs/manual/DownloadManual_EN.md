# VenusFactory Download Tab User Guide

## InterPro Metadata
**Description**: Downloads protein domain information from InterPro database.

**Source**: [InterPro Database](https://www.ebi.ac.uk/interpro/)

**Download Options**:
- Single ID: Download data for a specific InterPro domain (e.g., IPR000001)
- From JSON: Batch download using a JSON file containing multiple InterPro entries

**Output Format**:
```
download/interpro_domain/
└── IPR000001/
    ├── detail.json    # Detailed protein information
    ├── meta.json      # Metadata including accession and protein count
    └── uids.txt       # List of UniProt IDs associated with this domain
```

## RCSB Metadata
**Description**: Downloads structural metadata from the RCSB Protein Data Bank.

**Source**: [RCSB PDB](https://www.rcsb.org/)

**Download Options**:
- Single ID: Download metadata for a specific PDB entry (e.g., 1a0j)
- From File: Batch download using a text file containing PDB IDs

**Output Format**:
```
download/rcsb_metadata/
└── 1a0j.json         # Contains structure metadata including:
                     # - Resolution
                     # - Experimental method
                     # - Publication info
                     # - Chain information
```

## UniProt Sequences
**Description**: Downloads protein sequences from UniProt database.

**Source**: [UniProt](https://www.uniprot.org/)

**Download Options**:
- Single ID: Download sequence for a specific UniProt entry (e.g., P00734)
- From File: Batch download using a text file containing UniProt IDs
- Merge Option: Combine all sequences into a single FASTA file

**Output Format**:
```
download/uniprot_sequences/
├── P00734.fasta      # Individual FASTA files (when not merged)
└── merged.fasta      # Combined sequences (when merge option is selected)
```

## RCSB Structures
**Description**: Downloads 3D structure files from RCSB Protein Data Bank.

**Source**: [RCSB PDB](https://www.rcsb.org/)

**Download Options**:
- Single ID: Download structure for a specific PDB entry
- From File: Batch download using a text file containing PDB IDs
- File Types:
    * cif: mmCIF format (recommended)
    * pdb: Legacy PDB format
    * xml: PDBML/XML format
    * sf: Structure factors
    * mr: NMR restraints
- Unzip Option: Automatically decompress downloaded files

**Output Format**:
```
download/rcsb_structures/
├── 1a0j.pdb          # Uncompressed structure file (with unzip)
└── 1a0j.pdb.gz       # Compressed structure file (without unzip)
```

## AlphaFold2 Structures
**Description**: Downloads predicted protein structures from AlphaFold Protein Structure Database.

**Source**: [AlphaFold DB](https://alphafold.ebi.ac.uk/)

**Download Options**:
- Single ID: Download structure for a specific UniProt entry
- From File: Batch download using a text file containing UniProt IDs
- Index Level: Organize files in subdirectories based on ID prefix

**Output Format**:
```
download/alphafold2_structures/
└── P/               # With index_level=1
    └── P0/          # With index_level=2
        └── P00734.pdb  # AlphaFold predicted structure
```

## Common Features
- **Error Handling**: All components support error file generation
- **Output Directory**: Customizable output paths
- **Batch Processing**: Support for multiple IDs via file input
- **Progress Tracking**: Real-time download progress and status updates

## Input File Formats
1. **PDB ID List** (for RCSB downloads):
```
1a0j
4hhb
1hho
```

2. **UniProt ID List** (for UniProt and AlphaFold):
```
P00734
P61823
Q8WZ42
```

3. **InterPro JSON** (for batch InterPro downloads):
```json
[
    {
        "metadata": {
            "accession": "IPR000001"
        }
    },
    {
        "metadata": {
            "accession": "IPR000002"
        }
    }
]
```

## Error Files
When enabled, failed downloads are logged to `failed.txt` in the output directory:
```
P00734 - Download failed: 404 Not Found
1a0j - Connection timeout
``` 