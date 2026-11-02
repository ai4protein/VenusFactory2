# Upstream source dependencies

`scientific-agent-skills/` is a Git submodule with its own license, history and
pinned revision. Author platform-specific skills in `src/agent/skills/`; update
upstream reference skills through the submodule.

The loader in `src/agent/skills.py` exposes both roots through one registry and
MCP `list_skills` / `read_skill`. No manual copying or extra registration is
needed. Native skill names take precedence and upstream collisions are named
`sas_<id>`. The optional Kimi links are generated from this same registry.

```bash
# For a checkout that has not initialized the optional dependency:
git submodule update --init --recursive
python scripts/validate_skills.py --check-refs
```

Do not place model weights or installed model environments here. Those belong
in `runtime/<mode>/model-runtimes/`; adapters belong in `mcp_tools/`.
