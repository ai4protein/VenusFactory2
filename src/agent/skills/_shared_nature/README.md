# `_shared/` — Common content for nature-* skills

This directory is **not a skill**. It has no `SKILL.md` and is not registered with the plugin loader. It exists so multiple skills can reference the same content without duplication.

Files here are referenced by sibling skills via relative paths in their `manifest.yaml`, for example:

```yaml
always_load:
  - ../_shared/core/reader-workflow.md
```

## Current contents

| File | Used by |
|---|---|
| `core/reader-workflow.md` | nature-polishing, nature-writing |
| `core/paper-type-taxonomy.md` | nature-polishing, nature-writing |
| `core/ethics.md` | nature-polishing, nature-writing |
| `core/terminology-ledger.md` | nature-polishing, nature-writing |
| `journal-formats/nat-comms.md` | nature-polishing, nature-writing |

## When to add a file here

Only when ≥ 2 skills need the same content. If only one skill needs it, keep it inside that skill's `static/`.

## When to keep content skill-local instead

The shared layer holds **definitions and reference material** (taxonomy, reader workflow, ethics rules). The **action layer** — how a specific skill diagnoses, drafts, or revises — stays inside each skill's `static/fragments/`. Two skills can use the same paper-type taxonomy but apply different actions on top of it.
