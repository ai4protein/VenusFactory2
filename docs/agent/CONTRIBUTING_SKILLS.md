# Contributing VenusFactory2 Skills

See also: [`src/agent/skills/AGENTS.md`](../../src/agent/skills/AGENTS.md).

## Checklist (PR)

1. [ ] Directory name = frontmatter `name` = `read_skill` `skill_id` (snake_case).
2. [ ] `description` includes Use when / Do NOT use (with alternate skill_ids).
3. [ ] `metadata.version` quoted string; bump on every content change.
4. [ ] Tool-bound or orchestration skills list **exact** hub tool names under `## Project Tools` or `## Workflow`.
5. [ ] No secrets, hardcoded API keys, or private URLs.
6. [ ] `SKILL.md` ≤ ~500 lines; long material in `references/`.
7. [ ] `python scripts/validate_skills.py` passes (use `--strict` when fixing legacy nature_*).
8. [ ] Prefer instruction-only orchestration over new Python unless an API is unwrapped.

## Naming

| Kind | Pattern | Example |
|------|---------|---------|
| Database wrapper | `{source}_database` | `uniprot_database` |
| Package guidance | `{package}` | `rdkit`, `pymol` |
| Platform workflow | `{domain}_workflow` / `_pipeline` | `zero_shot_mutation_workflow` |
| Meta | fixed | `workflow_skill_creator` |

## Template

```markdown
---
name: example_skill
description: >-
  One-line capability. Use when …. Do NOT use when … (use other_skill).
license: Unknown
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# Title

## Overview

## VenusFactory execution
- Call hub tools by exact name; parse `status` / `file_info` / `biological_metadata`.
- Load this package: `read_skill` then optional `relative_path` for `references/`.

## Project Tools (VenusFactory2)
| Tool | Args | Returns | When |

## Workflow
### 1. …

## When to use / When NOT to use

## Common mistakes

## References
- `references/….md`
```

## scientific-agent-skills (submodule)

The full upstream library is already wired via git submodule
`third_party/scientific-agent-skills` and the dual-root loader. Contributors
normally **do not** copy packages into `src/agent/skills/`.

| Topic | Rule |
|-------|------|
| Init | `git submodule update --init --recursive` (or clone with `--recurse-submodules`) |
| Addressing | VF2 same-name wins; upstream collision → `sas_<id>` |
| Execution | Prefer VF2 tool-bound skills; SAS is reference knowledge via `read_skill` |
| Upstream scripts | Not auto-run; no per-skill `uv` runners in VF2 |
| Validate | `validate_skills.py` strictly checks VF2; SAS only needs `SKILL.md` present |

**Still do not** treat `allowed-tools` as VF2’s permission model, and do not
rewrite upstream SKILL bodies just to bind hub tools unless you are creating a
first-class VF2 skill under `src/agent/skills/`.
