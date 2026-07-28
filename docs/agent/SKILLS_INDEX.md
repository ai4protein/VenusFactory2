# VenusFactory2 Skills Index

Author guide: [`src/agent/skills/AGENTS.md`](../../src/agent/skills/AGENTS.md) · Validate: `python3 scripts/validate_skills.py`

## Dual chat modes

| Mode | Engine | How skills are loaded |
|------|--------|------------------------|
| **Science Expert** | LangGraph PI→CB→MLS | CB sees metadata; plan may **require** `read_skill` before code steps; MLS executes |
| **Science Agent** | kimi-code | System prompt embeds the skill **catalog**; agent **decides when** to call MCP `read_skill` / `list_skills` (no mandatory skill-first). Online denies kimi built-in `Skill`; use MCP. Local may also use built-in `Skill` via `.kimi-code/skills/`. |

### Code ownership

| Layer | Paths | Responsibility |
|-------|-------|----------------|
| **Core** | `src/agent/skills.py`, `src/agent/skills/*` | Discover / read packages; shared JSON envelopes (`build_read_skill_response`, `build_list_skills_response`); catalogs |
| **Expert adapter** | `src/tools/skill/tools_agent.py` | LangChain `read_skill` → core envelope → `json.dumps` |
| **Agent adapter** | `src/tools/skill/tools_mcp.py`, `src/agent/kimi_skills.py` | MCP `read_skill` / `list_skills`; optional `.kimi-code/skills` symlinks |
| **chat_api entry** | `messages.py` → `_stream.py` (Expert) / `_stream_kimi.py` (Agent); mode helpers in `_shared.py` | Route by `chat_mode` / engine; session gates & snapshots — **no** skill business logic in `_shared` |

Policy reminder: **Agent** self-directs when to open a skill; **Expert** may force skill-first via plan helpers. New skills only need a package under `src/agent/skills/` — both adapters pick them up automatically.

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
