---
name: workflow_skill_creator
description: Distills a completed user workflow or interaction into a reusable VenusFactory agent skill. Use when the user says "make this a skill", "create a skill from what we just did", "package this workflow" or similar. Adapts the workflow into the VenusFactory tools wiring + SKILL.md pattern. Do not use for creating skills from scratch without an existing workflow.
license: Apache-2.0 (adapted from google-deepmind/science-skills)
metadata:
  version: "1.1"
  skill-author: VenusFactory2 (adapted from Google DeepMind)
---

# Workflow-to-Skill Distiller

Turns a completed VenusFactory workflow into a reusable agent skill. Extracts patterns from an interaction that **already happened** and packages them following the project's conventions.

> [!CAUTION] **You MUST complete Phase 1 (Brainstorming) before writing any code or SKILL.md content.** Skipping brainstorming produces skills that are either too rigid or too vague.

## Phase 1: Brainstorming (MANDATORY)

Have an **iterative back-and-forth conversation** with the user. Do NOT ask all questions at once. Pick 2-3 relevant questions per round, refine your understanding, and ask follow-ups.

### Round 1: Understand the Workflow

Start by summarizing what you observed, then ask:

1. "Here's my understanding of the workflow: [summary]. Is this accurate?"
2. "What are the expected inputs and outputs?"
3. "How often will this run? One-off, recurring, or part of a larger pipeline?"

### Round 2: Flexibility and Error Handling

For each step:

1. "If [step X] fails (API down, no results), should the agent (a) ask for guidance, (b) try alternatives automatically, or (c) fail loudly?"
2. "Are there steps where the exact method matters (must use a specific database / model), vs. steps where any reasonable approach is fine?"

### Round 3: Reuse Existing VenusFactory Tools

Before asking these questions, check `src/tools/database/`, `src/tools/visualize/`, `src/tools/predict/`, `src/tools/mutation/`, `src/tools/search/`, `src/tools/train/`, `src/tools/file/`, `src/tools/bioinfo/` for overlap. **If an existing tool covers a step, the new skill MUST reference it — do not reimplement.**

1. "I noticed the workflow uses [tool X, tool Y] that already exist. The new skill will reference these. Anything else to incorporate?"
2. "Are there rate limits for new external APIs that aren't covered?"
3. "Any reference docs (API specs, papers, datasets) I should include under `references/`?"

### Round 4: Scope, Code, and Naming

1. "Should the skill cover [X, Y, Z] from the workflow, or include/exclude anything?"
2. Determine whether the skill needs new code:
    - **Needs new code** if any step calls an external API not yet wrapped, processes files, or computes results not already in `src/tools/`. → Follow the **6-file wiring recipe** below.
    - **Instruction-only** if every step orchestrates existing VenusFactory tools. → Write SKILL.md only, no Python.
3. Propose a name: `{verb}_{noun}_database` for new DB wrappers, `{noun}` for analysis/visualization, matching existing folders (`alphafold_database`, `pymol`, `clustalo_msa`, etc.).

### Round 5: Validation (Optional)

1. "Sample query + expected answer I can use to verify? Optional but helpful."

### Brainstorming Completion Criteria

Move to Phase 2 only when you can answer:

- [ ] Purpose and scope
- [ ] Inputs and outputs
- [ ] Strict vs flexible steps
- [ ] Which existing VenusFactory tools are reused
- [ ] What new scripts (if any) are needed
- [ ] Rate limits
- [ ] Error handling strategy
- [ ] Code needed (→ 6-file wiring) or instruction-only (→ SKILL.md only)
- [ ] Sample query/answer

## Phase 2: Skill Design

Produce a **design document** (markdown plan) and present for approval:

1. **Skill name and frontmatter** (see Rule 6).
2. **Directory structure** showing all planned files.
3. **Existing VenusFactory tools referenced** with rationale.
4. **New code files** with proposed function signatures.
5. **Rate limiting strategy** for any new external API.
6. **Error handling strategy** per step.

**Wait for explicit user approval before Phase 3.**

## Phase 3: Implementation

### Guiding Principles

- Match the project's existing Python style — use the conda env at `~/miniconda3/envs/venus/bin/python` (see `environment.yaml`); do NOT introduce `uv` or per-script `# /// script` headers.
- Prefer `requests` + `urllib3.util.retry.Retry` for HTTP (this is the established pattern, e.g. `src/tools/database/alphafold/alphafold_structure.py`).
- All public download/query functions return a JSON **string** with the rich envelope:
  - Success: `{"status": "success", "file_info": {...} | "content": "...", "content_preview": str, "biological_metadata": {...}, "execution_context": {...}}`
  - Error: `{"status": "error", "error": {"type", "message", "suggestion"}, "file_info": null}`
- Use `to_client_file_path()` from `src/tools/path_sanitizer.py` for any `file_path` in responses.
- Document rate limits in comments and respect them with `time.sleep` between polls.

### Rule 1: Reuse Existing Tools

When the workflow uses functionality already in `src/tools/`, the new SKILL.md **MUST** reference the existing tool by name in its `## Project Tools` table — do not duplicate the implementation.

### Rule 2: Rate Limiting for New APIs

For any new external API, the implementation **MUST** handle rate limits:

- Look up the API's documented rate limit; if undocumented, default to **1 req/s**.
- For submit-poll-download APIs (EBI, ColabFold, Foldseek webserver), poll interval ≥ 5 s with a sensible upper-bound timeout (5–10 min).
- Use `time.monotonic()` for timing.
- Retry transient 5xx + 429 with exponential backoff (existing pattern: `Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])`).
- On non-retriable 4xx, include the response body in the error message — bodies contain actionable details (e.g. "invalid accession") that let the agent self-correct.

### Rule 3: The 6-File Wiring Recipe (When Code Is Needed)

To add a new database/visualization/etc tool to VenusFactory, touch exactly these files in this order:

1. **`src/tools/<category>/<name>/__init__.py`** — export the public function(s).
2. **`src/tools/<category>/<name>/<name>_operations.py`** — copy the `_error_response()` / `_download_success_response()` envelope builders from `src/tools/database/alphafold/alphafold_operations.py`, then implement the entry function returning the JSON string.
3. **`src/tools/database/tools_agent.py`** (or category equivalent) — add `from .<name> import ...`, define `class <Tool>Input(BaseModel)`, write the `@tool("…", args_schema=<Tool>Input)` wrapper with a try/except, and append to `DATABASE_TOOLS`.
4. **`src/tools/database/tools_api.py`** — add a `@router.get` / `@router.post` route calling the same core function.
5. **`src/tools/database/tools_mcp.py`** — add a `@mcp.tool(name="…")` wrapper (no Pydantic schema; positional + typed args).
6. **`src/tools/tools_agent_hub.py`** — add the new `_tool` name to (a) the top `from tools.database.tools_agent import (...)` block, (b) the `DATABASE_TOOLS` list, and (c) the `__all__` re-export list.

After wiring, verify:
```bash
~/miniconda3/envs/venus/bin/python -c "from src.tools.tools_agent_hub import get_tools; print(len(get_tools()))"
```
The count must have increased by the number of new tools.

### Rule 4: SKILL.md Goes in `src/agent/skills/<name>/`

Every skill needs a `SKILL.md` with the VenusFactory frontmatter:

```markdown
---
name: <snake_case_name>   # MUST equal directory name / read_skill skill_id
description: <capability + Use when... + Don't use for... (other skill_ids)>
license: <e.g. Apache-2.0 / Unknown>
metadata:
  version: "1.0"
  skill-author: VenusFactory2
---

# <Skill Title>

## Overview
<1 paragraph>

## VenusFactory execution
- Hub tools by exact name; progressive files via read_skill(..., relative_path=...)

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **<tool_name_in_agent>** | <required + defaults> | <JSON envelope shape> | <one-line purpose> |

## When to Use This Skill
<bulleted scenarios>

## Common Mistakes
<2-3 pitfalls>
```

Optional: a `references/` subdirectory for API specs, recipe books, or sample payloads.
See `src/agent/skills/AGENTS.md` and `docs/agent/CONTRIBUTING_SKILLS.md`.

### Rule 5: Instruction-Only Pattern (No New Code)

If the workflow purely orchestrates existing VenusFactory tools, skip steps 1–6 of Rule 3 and write only `src/agent/skills/<name>/SKILL.md`. Use a clear `## Workflow` section:

```markdown
## Workflow

### 1. Step Name
- Description
- Which existing tool to call (e.g. `download_alphafold_structure_by_uniprot_id`)
- How to chain its output to the next step

### 2. Next Step
...
```

### Rule 6: File-First Output

All download tools **MUST** write large payloads to disk and return only `{file_path}`. Never return raw PDB / mmCIF / large JSON in the response — it explodes the agent's context window. The corresponding `analyze_*` / `read_*` tools then read from those files.

## Phase 4: Validation

After implementation:

1. **Import smoke test**: `python -c "from src.tools.<category>.<name> import <fn>"` (if new code).
2. **Hub count check**: confirm `get_tools()` count increased (if new tools).
3. **Skill validate**: `python scripts/validate_skills.py`
4. **Manual invocation** through `chat_agent.py` with a prompt that should trigger the new skill.
5. **Sample query** from Round 5 (if provided) end-to-end.

## References

- `references/cli_script_template.py` — original google-deepmind template, kept for reference (NOT used directly in VenusFactory; we don't use `uv` / inline scripts).
- Canonical VenusFactory wiring example: `src/tools/database/alphafold/` end-to-end (see also `alphafold_analyze.py` for a pure-local-analysis pattern, no external API).
- Canonical submit-poll-download example: `src/tools/database/foldseek/`.
