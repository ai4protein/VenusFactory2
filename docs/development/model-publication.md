# Model discovery and publication

Model Plaza and MCP discovery use `mcp_tools/models/platform_catalog.py` for model identities, runtime readiness and live administrator switches. Agent starts with search / describe / call, not all model schemas. Model names and schema metadata are searchable on the host; only bounded results enter the LLM context. Skills use list_skills(query, limit, offset), then read_skill.

Scientist writes reviewed deliverables to the goal's private_capabilities directory in both modes. Independent scientific review does not grant public visibility. Browser users cannot publish by changing visibility or invoking legacy goal publication. Publication applications are emails to tanyang.august@sjtu.edu.cn, not stored in an application inbox.

## Operator publication

Inspect the candidate's source, test report, data/license provenance and admission receipt. On the local deployment, use the platform Python environment:

```bash
WEBUI_V2_MODE=local python scripts/publish_research_model.py path/to/reviewed-candidate \
  --goal-id RESEARCH_ID --author-email author@example.org --curator operator@example.org
```

The command requires valid independent admission and unchanged source files. It stores a separate publication decision bound to the candidate and review hashes, preserving the original research manifest. The published author email is visible in discovery and Model Plaza. No publication command is exposed as an MCP tool. Use Admin's model switch to disable a released model.

Existing uncurated autonomous entries are retained on disk but excluded from the public capability catalog until selected and published with this command. Earlier paper-reproduction campaign libraries remain local-only; they are not imported into the online curated catalog.
