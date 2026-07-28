# VenusFactory2 Agent Skills Guide

Skills are discovered from **two roots** (see `src/agent/skills.py`):

1. **VenusFactory2 (tool-bound):** `src/agent/skills/<skill_id>/`
2. **scientific-agent-skills (reference library, optional git submodule):**  
   `third_party/scientific-agent-skills/skills/<id>/`

The agent loads **metadata** into the Computational Biologist (CB) prompt and full skill files into the Machine Learning Specialist (MLS) via `read_skill`. Protein-engineering **execution** should prefer VF2 tool-bound skills; the scientific library is knowledge for `read_skill` / `list_skills`. Upstream `scripts/` and `uv` runners are **not** auto-executed.

Catalog: [`docs/agent/SKILLS_INDEX.md`](../../../docs/agent/SKILLS_INDEX.md) · Validate: `python scripts/validate_skills.py`

## Submodule init

```bash
git clone --recurse-submodules https://github.com/AI4Protein/VenusFactory2.git
# or, after a plain clone:
git submodule update --init --recursive
```

If the submodule is missing, the loader uses VF2 skills only (no crash).

## Name collisions (`sas_`)

VF2 keeps the bare `skill_id`. If scientific-agent-skills has the same directory name, it registers as `sas_<dirname>` (e.g. VF2 `biopython` wins; upstream is `sas_biopython`). Non-colliding upstream packages keep their original directory name as `skill_id`.

## Layout

```text
src/agent/skills/                          # VF2 root
├── AGENTS.md                              # this file
├── <skill_id>/                            # skill_id == directory == read_skill argument
│   ├── SKILL.md                           # required
│   ├── references/                        # optional progressive disclosure
│   ├── assets/                            # optional
│   └── static/ + manifest.yaml            # only for nature_* routers (legacy pattern)
└── _shared_<domain>/                      # NOT a skill — no SKILL.md; skipped by loader

third_party/scientific-agent-skills/skills/   # optional submodule root
└── <id>/SKILL.md                          # may appear as sas_<id> on collision
```
## Frontmatter (VF2 profile)

```yaml
---
name: <skill_id>   # MUST equal directory name (snake_case allowed)
description: >-
  What it does. Use when …. Do NOT use when … (point to other skill_ids).
license: Apache-2.0  # or Unknown
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---
```

Allowed top-level keys only: `name`, `description`, `license`, `compatibility`, `allowed-tools`, `metadata`.

Notes:

- Prefer **snake_case** skill_ids (`zero_shot_mutation_workflow`). Do not rename existing kebab-case names without a migration plan.
- `allowed-tools` is ignored by VF2 (permissions come from LangChain tool hub).
- Put OpenClaw/Hermes host blocks under `metadata` only if you publish outside VF2.

## Content model

| Type | When | Must include |
|------|------|----------------|
| **A — Tool-bound** | Wraps `src/tools/**` APIs | `## Project Tools (VenusFactory2)` with exact `@tool` names |
| **B — Orchestration** | Chains existing tools | `## Workflow` steps with tool names + JSON envelope notes |
| **C — Nature router** | writing / polishing / figures | Short router + `read_skill(..., relative_path=…)` fragments |

Rules:

1. **Never reimplement** download/predict logic inside a skill; call hub tools.
2. Keep `SKILL.md` under ~500 lines; move long API docs to `references/`.
3. MLS loads package files with `read_skill(skill_id, relative_path="references/…")`.
4. Large payloads stay on disk (`file_info.file_path`); do not dump PDB/FASTA into chat.

## Runtime

- Discovery: `src/agent/skills.py` → `get_skills_metadata()` / `get_skills_metadata_string()`
- **Science Expert** (LangGraph): LangChain `read_skill` in `src/tools/skill/tools_agent.py`; CB prompt injects metadata; plan helper may enforce skill-first before code steps; MLS executes
- **Science Agent** (kimi-code): system prompt embeds compact catalog (`get_skills_catalog_for_agent`); agent **self-decides** when to open MCP `read_skill` / `list_skills` (no mandatory gate); local optional kimi `Skill` via `.kimi-code/skills/` (`agent.kimi_skills.ensure_kimi_project_skills`); online denies built-in `Skill`
- Plan helper auto-inserts `read_skill` before code steps using domain keyword rules in `plan_helpers.py`
- Validate: `python3 scripts/validate_skills.py`
- Cache: call `invalidate_skills_cache()` after runtime skill adds

## Creating skills

Use `workflow_skill_creator` for distillation from a completed session, or follow `docs/agent/CONTRIBUTING_SKILLS.md`.

## Do not

- Vendor/copy the scientific-agent-skills tree into `src/agent/skills/` (use the submodule + dual-root loader).
- Auto-run upstream `scripts/` / `uv` / `# /// script` isolation (use `src/tools/` + conda; `read_skill` may still show those files).
- Register `_shared_*` directories as skills.
- Put skill unit tests inside the skill directory (use `tests/`).
