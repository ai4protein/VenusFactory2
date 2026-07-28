# VenusFactory2 Agent Skills Guide

Skills live in `src/agent/skills/<skill_id>/`. The agent loads **metadata** into the Computational Biologist (CB) prompt and full skill files into the Machine Learning Specialist (MLS) via `read_skill`.

This profile is inspired by the open [Agent Skills](https://agentskills.io/) standard and [scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills), adapted for VenusFactory2’s **tool-bound** protein-engineering agent (not a generic science mega-catalog).

Catalog: [`docs/agent/SKILLS_INDEX.md`](../../../docs/agent/SKILLS_INDEX.md) · Validate: `python scripts/validate_skills.py`

## Layout

```text
src/agent/skills/
├── AGENTS.md                 # this file
├── <skill_id>/               # skill_id == directory == read_skill argument
│   ├── SKILL.md              # required
│   ├── references/           # optional progressive disclosure
│   ├── assets/               # optional
│   └── static/ + manifest.yaml   # only for nature_* routers (legacy pattern)
└── _shared_<domain>/         # NOT a skill — no SKILL.md; skipped by loader
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

- Copy all 154 skills from scientific-agent-skills into this tree.
- Add `uv` / `# /// script` per-skill runners (use `src/tools/` + conda).
- Register `_shared_*` directories as skills.
- Put skill unit tests inside the skill directory (use `tests/`).
