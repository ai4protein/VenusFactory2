# Report — One-Click Comprehensive Protein Analysis

The **Report** tab gives you a single document covering mutation, function, residue, and physicochemical analyses for one protein, with an AI-written narrative on top of the raw results.

## 1. When to Use It

- You're starting on a new protein and want a single overview without configuring multiple tools.
- You need a sharable analysis (HTML + PDF) for collaborators.
- You want mutation hotspots alongside functional context in one place.

For step-by-step, agent-driven analyses, use **Agent / Chat** instead.

---

## 2. Layout

The page has three columns:

| Column | What it does |
| :--- | :--- |
| **Left — Input & Controls** | Choose input mode (Paste / Upload), pick chain or sequence, tick the analyses to run, click **Generate Report**. Shows progress %. |
| **Center — AI Expert Analysis** | Renders the AI-generated narrative once the run finishes. Below it, links for **Download HTML** and **Download PDF**. |
| **Right — Streaming Logs** | Live log lines from the backend pipeline. |

A header status pill shows the current phase: *Ready / Processing Input / Generating Report*.

---

## 3. Inputs

| Input mode | How to use |
| :--- | :--- |
| **Paste** | Paste a sequence or FASTA into the textarea. |
| **Upload** | Drop a `.fasta` / `.fa` / `.pdb` file, pick one from **Workspace**, or hit **Use Default Example** to load a sample FASTA. |

After parsing:

- A **chain / sequence selector** appears if the input contains multiple chains / records.
- A short preview of the chosen sequence is shown.

> **Online mode:** upload is restricted; use Paste with a sequence within the FASTA character limit.

---

## 4. Selecting Analyses

Tick at least one of the four analyses:

| Icon | Analysis | Powered by |
| :---: | :--- | :--- |
| 🧬 | **Mutation** | Saturation mutagenesis scoring (ESM-2, ProSST, ProtSSN…) |
| 🔬 | **Function** | Fine-tuned predictors (solubility, localization, stability, optimum T°, sorting signal, metal-ion binding) |
| 🎯 | **Residue** | Activity / binding / conserved sites, motifs |
| ⚗️ | **Properties** | Sequence-based physicochemical calculation (MW, pI, instability index, GRAVY, secondary-structure composition) |

Ticking all four gives the most complete report.

---

## 5. Run & Watch

Click **Generate Report**. The page streams events:

- **Progress** — bar + message (e.g., "Predicting solubility…", "Scoring mutations…")
- **Logs** — appended to the right column
- **Done** — renders the AI narrative in the center, enables HTML / PDF links

If anything fails, an error block appears in the left column with a short reason; logs in the right column have the detail.

---

## 6. What's Inside the Report

| Section | Contents |
| :--- | :--- |
| **Comprehensive Summary** | Top-level brief: MW, theoretical pI, and a one-paragraph assessment. |
| **Mutation Prediction Analysis** | Top beneficial mutations table (Rank / Position / Mutation / Score / Notes), secondary list, and key-site optimization suggestions. |
| **Protein Function Analysis** | Per-task table: Property, Predicted Value, Confidence, Description — covering solubility, localization, metal ion binding, stability, sorting signal, optimum temperature. |
| **Functional Residue** | Binding-site / functional-residue / motif predictions with sequence positions and probabilities. |
| **Physical & Chemical Properties** | Biophysical characterization plus an instability-index-driven stability call. |
| **Experimental Recommendations** | Synthesized advice for function / stability, technical considerations, and protocol suggestions. |
| **Conclusion** | Final wrap-up of the protein's key characteristics and the most important optimization directions. |

---

## 7. Tips

- **Prefer PDB input when available.** Structure-aware analyses (e.g., structure-based mutation scoring) only kick in when a `.pdb` is provided.
- **Cross-read sections.** Beneficial mutations near a **predicted binding site** are usually a better experimental target than mutations far from any functional residue.
- **Save the downloads.** The HTML / PDF artifacts live in temporary backend storage; pull them locally before closing the browser.
- **Long sequences are slow.** Mutation scoring time scales with sequence length × number of amino acid substitutions. Trim signal peptides / disordered tails if you only care about a specific domain.
