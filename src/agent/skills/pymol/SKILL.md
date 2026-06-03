---
name: pymol
description: Headless PyMOL rendering of protein structures (PNG + PSE session) and structural superposition with RMSD. Use to produce static publication-style images, color a structure by pLDDT/B-factor/chain/secondary-structure, or compare two structures by cealign. Do NOT use for interactive 3D exploration (use MolstarViewer in the frontend), docking, MD simulation, or sequence-only analysis.
license: Apache-2.0 (adapted from google-deepmind/science-skills)
metadata:
    skill-author: VenusFactory2 (adapted from Google DeepMind)
---

# PyMOL Headless Rendering

## Overview

Two exposed agent tools drive an OSMesa-rendered PyMOL subprocess — no GPU, no X server. Each call writes a PNG image and a `.pse` session file to disk; the agent only sees `{file_path, metadata}` in the response, never raw image bytes.

This skill complements VenusFactory's interactive `MolstarViewer` in the frontend: use Molstar when the user is exploring a structure live, use PyMOL when you need a reproducible static image for a report.

## Prerequisites

PyMOL must be on the host's PATH. Verify with `pymol -V`. If missing, install via:

- conda (preferred): `conda install -n venus -c conda-forge pymol-open-source`
- pip: `pip install pymol-open-source-whl` (requires Python 3.10 ≤ x < 3.13)

When PyMOL is missing, the tools return `{status: "error", error: {type: "DependencyMissing", message: "<install hint>"}}` — surface the install command to the user instead of trying anything else.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **render_protein_structure** | `pdb_path` (required), `out_dir` (required), `color_by` (default `"plddt"`; one of `plddt` \| `bfactor` \| `chain` \| `ss`), `width` (default `1200`), `height` (default `900`), `dpi` (default `150`), `timeout_secs` (default `300`) | JSON: `{status, file_info {file_path -> <stem>_<color>.png}, content_preview (pymol stdout), biological_metadata {input_structure, color_by, png_path, pse_path, image_dimensions}}` | Cartoon-render a single structure with one of four coloring strategies. |
| **superpose_two_structures** | `pdb_a` (mobile, required), `pdb_b` (reference, required), `out_dir` (required), `width`/`height`/`dpi`/`timeout_secs` (same defaults) | JSON: `{status, file_info {file_path -> superpose_<a>_to_<b>.png}, biological_metadata {mobile, reference, rmsd_angstroms, aligned_residues, png_path, pse_path}}` | Align `pdb_a` onto `pdb_b` via `cealign` (auto-fallback to `align`), render the superposition, report RMSD. |

## Coloring Strategies

| `color_by` | Use for |
|---|---|
| `plddt` (default) | AlphaFold predictions — B-factor column = pLDDT, spectrum red↔blue at 50-100 |
| `bfactor` | Experimental structures — generic B-factor coloring |
| `chain` | Multi-chain complexes — one color per chain |
| `ss` | Highlight helix (green) / sheet (yellow) / loop (gray) |

## When to Use This Skill

- User asks "show me the structure", "render it", "make an image", "compare these two structures"
- Building a report or visualization that needs a static image
- After running an AlphaFold prediction or downloading a PDB, you want to visualize the result
- Comparing predicted vs. experimental structure → `superpose_two_structures`

## When NOT to Use

- User wants to spin / zoom / click on the structure → use the frontend MolstarViewer instead
- Docking / molecular dynamics → out of scope, recommend other tools
- Sequence-only operation → use sequence skills (uniprot, ncbi_sequence_fetch, etc.)
- Animated movies → not exposed; the underlying PyMOL supports it but no agent tool surfaces it

## Output Files

Each run produces two files in `out_dir`:

- **`*.png`** — the static image (returned in `file_info.file_path`).
- **`*.pse`** — a PyMOL session file. Tell the user they can open it locally in PyMOL to continue exploring.

## Rate / Cost

PyMOL runs locally; no network. Typical render is 5-30 seconds per structure. Default timeout is 5 min — increase `timeout_secs` for very large complexes or surfaces.

## Common Mistakes

- **Passing a sequence** instead of a structure file → `ValidationError: empty pdb_path` or `NotFound`. Download the structure first (e.g., `download_alphafold_structure_by_uniprot_id` or `download_rcsb_structure_by_pdb_id`).
- **Invalid `color_by`** → ValidationError listing the four allowed values.
- **PyMOL not installed** → `DependencyMissing` with the install command. Show the user — don't silently fall back.
- **Trying to render very large complexes (>1 M atoms) with surface** at high DPI → can time out. Drop dpi to 100 or increase `timeout_secs`.

## References

- `references/PYMOL_REFERENCE.md` — selection syntax, common commands, gotchas (copied verbatim from google-deepmind/science-skills).
- `references/RECIPES.md` — 15+ copy-paste PyMOL recipes for advanced visualizations beyond what the two exposed tools cover. Useful when the user wants something custom — produce a one-off script via the lower-level `run_pymol_script` (in `src/tools/visualize/pymol/pymol_runner.py`) and run it from the agent's code-execution tool.
- [PyMOL open-source homepage](https://github.com/schrodinger/pymol-open-source) / [PyMOL wiki](https://pymolwiki.org/)
