# Quick Tools — Fast, No-Setup Protein Analysis

**Quick Tools** is the fastest way to run a single prediction. Pre-selected models, sensible defaults, minimal knobs. Pick the tool, drop your input, click **Start Prediction**.

For full control over which model / dataset to use, switch to **Advanced Tools** (same five tasks, plus more options).

## Tools at a Glance

| # | Tool | Route | Input | What it tells you |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Directed Evolution | `/quick-tools/directed-evolution` | FASTA or PDB | Best single-point mutations for a target function |
| 2 | Sequence Design | `/quick-tools/sequence-design` | PDB | Candidate sequences for a given backbone (ProteinMPNN) |
| 3 | Protein Discovery | `/quick-tools/protein-discovery` | PDB | Structural / sequence homologs via VenusMine |
| 4 | Protein Function | `/quick-tools/protein-function` | FASTA | Protein-level properties (solubility, localization, …) |
| 5 | Functional Residue | `/quick-tools/functional-residue` | FASTA | Per-residue activity / binding / conserved / motif sites |
| 6 | Physicochemical Property | `/quick-tools/physicochemical-property` | FASTA or PDB | MW, pI, SASA, secondary structure, etc. |

---

## Common Layout

Every Quick Tool page looks the same:

| Area | What's there |
| :--- | :--- |
| **Left — Input + Config** | Sequence / PDB input (paste / upload / Workspace / Use Example), task dropdown, optional **Enable AI analysis** toggle (+ LLM Provider when on), **Start Prediction** button. |
| **Right — Result Panel** | Status, tabs for the raw table, optional heatmap, optional AI Expert Analysis, plus **Download Results**. |

**Online mode:** sequence inputs are capped by the online FASTA limit (default 50 residues for paste). Protein Discovery is **view-only** in online mode.

---

## 1. Directed Evolution

Score every possible single-point mutation against a target function.

| Field | Notes |
| :--- | :--- |
| **Sequence input** | Paste FASTA, upload `.fasta` / `.fa` / `.pdb`, pick from Workspace, or use the example. |
| **Select Protein Function** | Target function for which to score mutations (see below). |
| **Enable AI analysis** | Optional; adds an AI Expert summary tab. |

**Function options:**

- **Activity** — impact of mutations on catalytic or biological activity.
- **Binding** — impact on the protein's ability to bind ligands or interaction partners.
- **Expression** — impact on expression level in the host cell.
- **Organismal Fitness** — impact on the survival / growth of the whole organism.
- **Stability** — impact on thermodynamic or conformational stability.

**Outputs:**
- **Raw table** — ranked mutants with prediction score
- **Prediction heatmap** — 2D matrix: Y = sorted positions, X = substituted amino acid; darker = stronger enhancement
- **AI Expert Analysis** (if enabled) — natural-language interpretation
- **Download Results**

---

## 2. Sequence Design

Generate candidate protein sequences for a given structure using ProteinMPNN with biology-friendly defaults.

| Field | Notes |
| :--- | :--- |
| **PDB input** | Upload `.pdb`, pick from Workspace, or use the example. |
| **Model Family** | Soluble (default — recommended for most cases), Vanilla (membrane proteins), CA (Cα-only coarse-grained). |
| **Designed Chains** | Optional, e.g. `A` or `A,B`. Empty = all chains. |
| **Fixed Residues** | Optional pin syntax, e.g. `A12,C13` or `A:12,13;B:5-8`. |
| **Number of sequences** | 4 / 8 / 16 / 32 (capped by online limit when enabled). |
| **Design Diversity** | Low / Medium / High (maps to ProteinMPNN sampling temperature). |
| **Enable AI analysis** | Optional. |

Defaults internally use `v_48_020` with `backbone_noise=0.20`, which works well for AlphaFold-style backbones and routine redesign.

**Outputs:**
- **Table** — generated sequences with header, length, score fields
- **Raw** — full JSON payload
- **AI Expert** (if enabled)
- **Download Result** — FASTA file

Need finer ProteinMPNN knobs? Use **Advanced Tools → Sequence Design**.

---

## 3. Protein Discovery

A one-click VenusMine pipeline for structural homolog search and clustering.

| Field | Notes |
| :--- | :--- |
| **PDB input** | Upload `.pdb`, pick from Workspace, or use the example. |
| **Advanced parameters** | Not exposed in Quick mode — backend defaults are used. |

Click the start button and wait. Outputs are compatible with the Advanced backend artifacts (tree / labels / archive download fields).

**Online mode:** the entire form is read-only.

For parameter tuning (protected region, MMseqs threads / iterations, cluster identity, e-value), switch to **Advanced Tools → Protein Discovery**.

---

## 4. Protein Function

Predict a protein-level property from a FASTA sequence.

| Field | Notes |
| :--- | :--- |
| **Sequence input** | Paste or upload FASTA, pick from Workspace, or use the example. |
| **Select Task** | Property to predict (see below). |
| **Enable AI analysis** | Optional. |

**Task options:**

- **Solubility** — whether the protein is likely to be soluble after expression (critical for purification).
- **Localization** — final location within the cell (nucleus / cytoplasm / mitochondria / …).
- **Metal ion binding** — whether the protein can bind specific metal ions.
- **Stability** — inherent stability against heat or chemical denaturation.
- **Sorting signal** — whether a signal peptide directs the protein to a specific organelle / secretion pathway.
- **Optimum temperature** — temperature range for maximum functional activity.

**Outputs:**
- **Raw table** — protein name, sequence, predicted class, confidence (0–1)
- **AI Expert Analysis** (if enabled)
- **Download Results**

Status message after run: *"All predictions completed. Results were aggregated using soft voting."*

---

## 5. Functional Residue

Predict residue-level functional sites along a sequence.

| Field | Notes |
| :--- | :--- |
| **Sequence input** | Paste or upload FASTA, pick from Workspace, or use the example. |
| **Select Task** | Type of residue-level site to predict (see below). |
| **Enable AI analysis** | Optional. |

**Task options:**

- **Activity Site** — residues responsible for catalytic / biological function.
- **Binding Site** — residues that bind ligands, ions, or other molecules.
- **Conserved Site** — residues highly retained during evolution; usually critical for structure or function.
- **Motif** — short amino acid patterns that form a specific structural / functional feature.

**Outputs:**
- **Raw table** — Position, Residue, Predicted Label (0/1), Probability (0–1)
- **Prediction heatmap** — 1-D probability strip along the residue axis
- **AI Expert Analysis** (if enabled)
- **Download Results**

---

## 6. Physicochemical Property

Compute biophysical properties — some sequence-only, some structure-only.

| Property | Input required |
| :--- | :--- |
| **Physical and chemical properties** | FASTA — MW, pI, aromaticity, instability index, GRAVY, predicted secondary-structure composition |
| **Relative solvent accessible surface area** | PDB — per-residue RSA |
| **SASA value** | PDB — total SASA (Å²) |
| **Secondary structure** | PDB — per-residue DSSP code (H, E, …) |

When you pick a PDB-only task with a multi-chain `.pdb`, an extra **PDB Chain** selector appears. Paste textarea is disabled for PDB-only tasks.

**Outputs:** task-specific table; **no** AI Expert tab for this module. Use **Download Results** to export.

---

## Tips

- **PDB-aware tasks need a PDB.** Don't paste a FASTA into a SASA / secondary-structure task — you'll get no result.
- **Use the example button** as a sanity check before uploading large files.
- **Watch the status pill** — most tasks finish in seconds, but Directed Evolution scales with `sequence_length × 20`.
- **AI summaries are optional.** Toggle them off if you only need the raw numbers and want faster turnarounds.
