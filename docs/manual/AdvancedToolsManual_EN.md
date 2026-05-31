# Advanced Tools — Full Control over Models & Datasets

**Advanced Tools** exposes the same task surface as Quick Tools, but lets you pick **which model** (and for some tasks, which datasets) actually runs. Use it when defaults aren't good enough, when you want to compare models, or when you need structure-based variants of mutation / discovery analyses.

## Tools at a Glance

| Tool | Route | Picks you make |
| :--- | :--- | :--- |
| Directed Evolution | `/advanced-tools/directed-evolution` | Sequence-based vs Structure-based · backbone model |
| Sequence Design | `/advanced-tools/sequence-design` | Full ProteinMPNN inference knobs |
| Protein Discovery | `/advanced-tools/protein-discovery` | VenusMine hyperparameters |
| Protein Function | `/advanced-tools/protein-function` | PLM + 1..N fine-tuned dataset checkpoints |
| Functional Residue | `/advanced-tools/functional-residue` | PLM for residue-level prediction |

> Physicochemical Property is **Quick Tools only** — it's a deterministic calculation, there's nothing to tune.

---

## Common Layout

| Area | What's there |
| :--- | :--- |
| **Left — Input + Config** | Sequence / PDB input, model selectors, task / dataset pickers, optional **Enable AI analysis** toggle, **Start Prediction**. |
| **Right — Result Panel** | Status, raw table, optional heatmap or prediction plots, optional AI Expert Analysis, **Download Results**. |

**Online mode:** Protein Discovery is read-only; sequence inputs are capped by the online FASTA limit.

---

## 1. Directed Evolution

Two prediction modes — toggle between them at the top:

### 1.1 Sequence Model

| Field | Notes |
| :--- | :--- |
| **Input** | FASTA via paste / upload / Workspace |
| **Select Model** | VenusPLM · ESM2-650M · ESM-1v |

### 1.2 Structure Model

| Field | Notes |
| :--- | :--- |
| **Input** | `.pdb` via upload / Workspace |
| **Select Model** | VenusREM · ProSST-2048 · ProtSSN · ESM-IF1 · SaProt · MIF-ST |

Both modes share: **Enable AI analysis** toggle, **LLM Provider** dropdown (DeepSeek / ChatGPT / Gemini) when AI is on.

**Outputs:** ranked mutant table + 2D prediction heatmap (Y = sorted positions, X = substituted AAs), optional AI Expert Analysis, Download.

---

## 2. Sequence Design

Full-control ProteinMPNN. Use this when you need reproducible benchmarks or specific design constraints.

| Group | Fields |
| :--- | :--- |
| **Structure** | Upload `.pdb`; Workspace picker |
| **Model & sampling** | Model Family (Soluble / Vanilla / CA), Model Name (`v_48_020` default, `v_48_002` for high-res native), Omit AAs (default `X`), Temperatures (e.g. `0.1` or `0.1,0.2`), Number of sequences (default 8, online capped), Design Diversity |
| **Chains** | Designed Chains, Fixed Chains, Fixed Residues (`A12,C13` or `A:12,13;B:5-8`), Homomer tying toggle |
| **Runtime** | Seed (default 0), Batch Size (default 1), Max Length (default 200000) |
| **Advanced rules (text)** | Tied Positions, Omit AA Rules, AA Bias, Bias-By-Residue, PSSM Rules — entered as text; backend auto-converts to JSONL |
| **PSSM numerics** | `pssm_multi`, `pssm_threshold`, `pssm_log_odds_flag`, `pssm_bias_flag` |

**Model recommendation:**
- Default: `v_48_020` (auto noise 0.20) — works for AlphaFold / AI-generated backbones
- High-resolution native structures: `v_48_002` (auto noise 0.02)

**Outputs:** summary status, FASTA table (header / sequence / length / score), raw JSON, Download FASTA.

For default-safe one-click usage, use **Quick Tools → Sequence Design**.

---

## 3. Protein Discovery (VenusMine)

Full parameter control over the VenusMine pipeline (structural alignment + sequence similarity + redundancy reduction + representation-based ranking + phylogenetic tree).

| Field | Default |
| :--- | :--- |
| **Protected Region Start / End** | 1 / 100 |
| **MMseqs Threads** | 96 |
| **MMseqs Iterations** | 3 |
| **MMseqs Max Sequences** | 100 |
| **Cluster Min Seq Identity** | 0.5 |
| **Cluster Threads** | 96 |
| **Tree Top-N Threshold** | 10 |
| **E-value Threshold** | 1e-5 |

**Input:** `.pdb` via upload / Workspace.

**Outputs:** clustering table + phylogenetic tree + downloadable archive.

**Online mode:** all controls are disabled (view-only).

---

## 4. Protein Function

Cross-validate a protein-level prediction across multiple fine-tuned datasets.

| Field | Notes |
| :--- | :--- |
| **Input** | FASTA via paste / upload / Workspace |
| **Model** | PLM backbone (ESM2-650M default) |
| **Task** | Protein-level property to predict — see option meanings below. |
| **Datasets** | Multi-select grid of fine-tuned datasets relevant to the selected task (e.g. DeepSol + ProtSolM + eSOL for solubility) |
| **Enable AI analysis** | Optional |

**Task options:**

- **Solubility** — whether the protein is soluble after expression.
- **Localization** — subcellular location.
- **Metal ion binding** — metal-binding capability.
- **Stability** — resistance to heat / chemical denaturation.
- **Sorting signal** — presence of a signal peptide / targeting motif.
- **Optimum temperature** — temperature for maximum activity.

**Outputs:** Raw table with an extra **Dataset** column (one row per dataset × sequence), prediction plots (e.g. bar charts for subcellular localization), optional AI Expert Analysis, Download.

---

## 5. Functional Residue

Residue-level prediction with a configurable PLM backbone.

| Field | Notes |
| :--- | :--- |
| **Input** | FASTA via paste / upload / Workspace |
| **Model** | PLM backbone (ESM2-650M default) |
| **Task** | Residue-level site type to predict — see option meanings below. |
| **Enable AI analysis** | Optional |

**Task options:**

- **Activity Site** — residues responsible for catalytic / biological function.
- **Binding Site** — residues that bind ligands, ions, or other molecules.
- **Conserved Site** — residues highly retained during evolution.
- **Motif** — short patterns with a specific structural / functional role.

**Outputs:** per-residue table (Position, Residue, Predicted Label, Probability), 1-D prediction heatmap, optional AI Expert Analysis, Download.

---

## When to Use Quick vs Advanced

| If you… | Use |
| :--- | :--- |
| Just want results, fastest path | **Quick Tools** |
| Want to compare 2-3 different PLMs on the same input | **Advanced Tools** |
| Need structure-based mutation scoring (PDB) | **Advanced Tools → Directed Evolution → Structure Model** |
| Need cross-dataset voting on the same task | **Advanced Tools → Protein Function** |
| Need to tune ProteinMPNN inference for a benchmark | **Advanced Tools → Sequence Design** |
| Need to tune VenusMine hyperparameters | **Advanced Tools → Protein Discovery** |
| Just want pI / SASA / secondary structure | **Quick Tools → Physicochemical Property** (no Advanced equivalent) |
