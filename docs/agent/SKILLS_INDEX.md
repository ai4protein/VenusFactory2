# VenusFactory2 Skills Index

Author guide: [`src/agent/skills/AGENTS.md`](../../src/agent/skills/AGENTS.md) · Validate: `python3 scripts/validate_skills.py`

## Dual roots

| Root | Path | Role |
|------|------|------|
| **VenusFactory2** | `src/agent/skills/` | Tool-bound protein-engineering skills (prefer for execution) |
| **scientific-agent-skills** | `src/agent/skills/vendor/scientific-agent-skills/skills/` (git submodule) | Full reference library via `read_skill`; upstream `scripts/` / `uv` **not** auto-run |

Init: `git clone --recurse-submodules …` or `git submodule update --init --recursive`. Missing submodule → VF2-only, no crash.

**Collisions:** VF2 keeps bare `skill_id`; upstream same directory name → `sas_<id>` (e.g. `sas_biopython`). Catalogs show a short scientific section so prompts stay bounded; call `read_skill` / `list_skills` for full text.

## Shared discovery across chat modes

| Mode | Purpose | Skill access |
|------|---------|--------------|
| **Agent** | Answer questions and use existing capabilities | Discover and read platform or scientific skills through MCP; choose when a skill is useful |
| **Scientist** | Develop, evaluate, iterate and review new capabilities | Research roles use the same skill registry and enabled tools; a frozen experiment may intentionally pin its own skill snapshot |
| **Researcher** | Investigate a protein and produce an evidence-based report | Research tools share the same registry; skills guide methods, while tool results supply evidence |

Harness choice does not create a separate skill source. MCP `list_skills` and
`read_skill` are the portable interfaces. Local Kimi can additionally discover
generated `.kimi-code/skills/` links using the same IDs. Online uses MCP reads
instead of unrestricted built-in filesystem access. Reading a skill never
automatically executes its scripts or enables a disabled computation tool.

### Code ownership

| Layer | Paths | Responsibility |
|-------|-------|----------------|
| **Core** | `src/agent/skills.py`, `src/agent/skills/*`, optional SAS submodule | Dual-root discover / read; `sas_` collisions; shared JSON envelopes; graded catalogs |
| **Role-tool adapter** | `mcp_tools/tools/skill/tools_agent.py` | LangChain `read_skill` → core envelope → `json.dumps` |
| **Agent adapter** | `mcp_tools/tools/skill/tools_mcp.py`, `src/agent/kimi_skills.py` | MCP `read_skill` / `list_skills`; `.kimi-code/skills` symlinks for VF2 + SAS |
| **Chat integration** | `backend/web_v2/chat_api/`, `src/agent/harness/` | Harness routing, session boundary and mode policy; no second copy of skill content |

New **VF2** skills go under `src/agent/skills/` — both adapters pick them up.
Keep upstream source in `src/agent/skills/vendor/scientific-agent-skills/`;
the registry provides the merged view.
Use `sas_biopython` to read the upstream package or `biopython` for the native
tool-oriented guide. `scanpy` and `scikit-learn` are examples of upstream-only IDs.

## Platform workflows (protein engineering)

| skill_id | Use for |
|----------|---------|
| `protein_engineering_hypothesis` | Hypothesis / next-round planning |
| `zero_shot_mutation_workflow` | Beneficial mutations (seq/struct PLM) |
| `proteinmpnn_design_workflow` | Inverse folding / fixed-backbone design |
| `protein_structure_pipeline` | AlphaFold/ESMFold → confidence → render |
| `protein_property_prediction` | Physchem / RSA / function heads |
| `venus_finetune_workflow` | Custom train → predict |
| `structure_file_prep` | FASTA/PDB/MAXIT prep |
| `foldseek_structural_similarity` | Structural homologs |
| `interpro_domain_annotation` | Domains / families |
| `hpa_expression_context` | Human tissue / localization |

## Databases & search

| skill_id | Use for |
|----------|---------|
| `uniprot_database` | Sequence / metadata / mapping / SPARQL |
| `alphafold_database` | AF DB download + pLDDT/PAE |
| `rcsb_database` | Experimental structures |
| `string_database` | PPI + enrichment |
| `kegg_database` | Pathways / KEGG REST |
| `brenda_database` | Enzyme kinetics (SOAP) |
| `chembl_database` | Bioactive molecules |
| `ncbi_sequence` / `ncbi_gene` / `ncbi_clinvar` | NCBI resources |
| `protein_sequence_similarity_search` | MMseqs2 / BLAST |
| `clustalo_msa` | MSA |
| `fda` | openFDA (`query_fda` + deep scripts) |
| `arxiv` / `biorxiv` / `pubmed` / `openalex` | Literature |

## Libraries & communication

| skill_id | Use for |
|----------|---------|
| `biopython` / `rdkit` / `pymol` | Library guidance (+ hub where wired) |
| `matplotlib` / `seaborn` | Exploratory plots |
| `nature_figure` / `nature_writing` / `nature_polishing` | Publication writing & figures |
| `workflow_skill_creator` | Distill a session into a new skill |

## Progressive disclosure

| Skill type | How to load extras |
|------------|--------------------|
| Database / library (slim) | `references/*.md` and archived `references/legacy_guide.md` (**SKILL.md is authoritative** for hub tool names / envelopes) |
| `nature_figure` | `manifest.yaml` → `static/core/*.md` → backend fragments |
| `nature_writing` / `nature_polishing` | `manifest.yaml` → `_shared_nature/core/*.md` + `static/core/*.md` |
| Orchestration workflows | Usually self-contained in `SKILL.md` (no legacy archive) |

```text
read_skill(skill_id="<id>")
read_skill(skill_id="<id>", relative_path="references/legacy_guide.md")
read_skill(skill_id="nature_writing", relative_path="_shared_nature/core/ethics.md")
```

`_shared_nature/` is **not** a skill (loader skips `_` dirs); only reachable via the whitelist above.
