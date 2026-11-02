# Developer commands

Run maintained developer commands from this directory. There is no separate
`script/` or `Scripts/` tree.

| Area | Entry points | Purpose |
| --- | --- | --- |
| Services | `serve.py` | API, MCP and fixed worker pools; see [service guide](../docs/architecture/services.md) |
| Installation | `setup_quickstart.py`, `bootstrap.sh`, `install/` | Environment installation |
| Configuration | `check_env.py`, `admin_access.py`, `setup_public_query_tools.py` | Environment checks and administration |
| Model assets | `download_ckpts.py`, `upload_ckpts.py` | Published platform checkpoint management |
| Research | `rsi/`, `harness/` | Controlled experiments and harness checks |
| Documentation | `sync_platform_docs.py`, `validate_skills.py` | Shared manual/wiki and skill validation |
| Architecture | `check_layout.py` | Canonical source paths and example entry points |
| CLI recipes | `examples/tools/` | Editable database, design, prediction and training examples |

CLI recipes are local developer commands. Their dataset paths, model weights and
compute settings must match the installation. Some perform real downloads or
training; `test_*.sh` here are live examples, not offline regression tests. Web
users submit work through the API/MCP queue, which enforces online restrictions.

```bash
python scripts/check_layout.py
python scripts/validate_skills.py --check-refs
bash scripts/examples/tools/search/source/literature.sh
```

The shell recipes locate the repository from their own path. Python module
entry points use `mcp_tools.*` (or the existing `src.train/evaluate/predict` CLI).
Keep reusable implementation in its owning package, not in a script.

Removed CLI interfaces have historical examples in
[`archive/cli-examples/`](../archive/cli-examples/README.md).
