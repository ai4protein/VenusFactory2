# Download — Batch Retrieval from Biological Databases

The **Download** tab lets you pull sequences, structures, and metadata from major databases in bulk, instead of clicking through each web page. All six sources share the same workflow: choose **Single ID** or **From File**, click run, get an archive.

## Sources at a Glance

| Source | Route | ID example | What you get |
| :--- | :--- | :--- | :--- |
| UniProt | `/download/uniprot` | `P00734` | Protein sequences (FASTA) |
| NCBI | `/download/ncbi` | `NP_000517.1` | Protein sequences (FASTA) |
| RCSB Structure | `/download/rcsb-structure` | `1a0j` | 3D structures (`.pdb` / `.cif`) |
| AlphaFold | `/download/alphafold` | `P00734` | Predicted structures (`.pdb`) |
| RCSB Metadata | `/download/rcsb-metadata` | `1a0j` | Per-entry JSON metadata |
| InterPro | `/download/interpro` | `IPR000001` | Per-entry JSON metadata |

---

## Common Layout

Every download page uses the same form:

| Field | Notes |
| :--- | :--- |
| **Download Method** | `Single ID` or `From File`. |
| **ID input (Single)** | Type one ID — label / placeholder is source-specific. |
| **File upload (From File)** | Drop a `.txt` (one ID per line) or pick from **Workspace**. Hit **Use Example** to load a sample list. A preview shows the first 20 entries. |
| **Save Error File** | (Default on) Writes failures to `failed.txt` in the output directory. |
| **Source-specific options** | See per-source notes below. |

After the run, a status message and a result archive link appear. For structure sources you also get an inline 3D viewer.

---

## 1. UniProt Sequences

| Field | Notes |
| :--- | :--- |
| **ID format** | UniProt accession (e.g. `P00734`). |
| **Extra option** | **Merge FASTA** — combine all hits into a single multi-record FASTA. |

**Output layout** (Single + no merge):
```
download/uniprot_sequences/
├── P00734.fasta      # individual FASTA per ID
└── merged.fasta      # only when merge enabled
```

---

## 2. NCBI Sequences

| Field | Notes |
| :--- | :--- |
| **ID format** | RefSeq / GenBank protein accession (e.g. `NP_000517.1`, `XP_011541001.1`). |
| **Extra option** | **Merge FASTA** — combine all hits into one file. |

Same on-disk layout as UniProt.

---

## 3. RCSB Structures

| Field | Notes |
| :--- | :--- |
| **ID format** | 4-character PDB code (e.g. `1a0j`). |
| **File Type** | `pdb` or `cif` (mmCIF, recommended for large structures). |

**Output layout:**
```
download/rcsb_structures/
└── 1a0j.pdb          # or 1a0j.cif
```

The downloaded structure is rendered inline via Molstar so you can spot-check before downloading the archive.

---

## 4. AlphaFold Structures

| Field | Notes |
| :--- | :--- |
| **ID format** | UniProt accession (e.g. `P00734`). |
| **Extra display** | Per-structure pLDDT and B-factor stats are shown next to the Molstar viewer. |

**Output layout:**
```
download/alphafold_structures/
└── P00734.pdb        # AlphaFold predicted structure
```

---

## 5. RCSB Metadata

| Field | Notes |
| :--- | :--- |
| **ID format** | PDB code. |
| **Returns** | Per-entry JSON: resolution, experimental method, publication info, chain composition, etc. |

```
download/rcsb_metadata/
└── 1a0j.json
```

---

## 6. InterPro Metadata

| Field | Notes |
| :--- | :--- |
| **ID format** | InterPro accession (e.g. `IPR000001`). |
| **Returns** | Domain detail + list of UniProt IDs associated with this domain. |

```
download/interpro_domain/
└── IPR000001/
    ├── detail.json    # detailed protein information
    ├── meta.json      # accession + protein count
    └── uids.txt       # list of UniProt IDs in this domain
```

---

## Input File Formats

**ID lists (UniProt / NCBI / RCSB / AlphaFold / InterPro):** one ID per line.

```
P00734
P61823
Q8WZ42
```

**InterPro batch from JSON** (legacy): a JSON array of objects with `metadata.accession`.

```json
[
    {"metadata": {"accession": "IPR000001"}},
    {"metadata": {"accession": "IPR000002"}}
]
```

---

## Error Files

When **Save Error File** is on, failed IDs are logged to `failed.txt` in the output directory:

```
P00734 - Download failed: 404 Not Found
1a0j   - Connection timeout
```

---

## Tips

- **Batch in chunks of 50–200.** Public APIs throttle aggressive parallel requests; the backend already paces calls but very large lists are still slower.
- **Use `cif` for big structures.** PDB legacy format doesn't fit modern large assemblies; `cif` does.
- **AlphaFold ≠ experimental.** Always check the pLDDT panel — low-confidence regions (orange/red) should not be over-interpreted.
- **Merge FASTA** is only useful when you want a single file for downstream pipelines (e.g. MMseqs / BLAST). Keep merge **off** if you want per-ID outputs.
