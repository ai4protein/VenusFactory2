# Retired implementations

This directory preserves superseded interfaces and installation utilities. It is
not on the current service import path, is not packaged in the application wheel,
and is not part of the default test suite. Use `scripts/serve.py` for current services.

| Folder | Retired content |
| --- | --- |
| `gradio/` | Original `src/webui.py`, `src/web/` tabs, assets and helpers |
| `frp/` | Project FRPC binary, download helper, CLI, tests and old installer configuration |
| `notebooks/` | `Scripts_notebook.ipynb`, preserved with its original outputs |
| `cli-examples/` | Old examples whose CLI entry points were removed; separate provenance manifest |

`manifest.json` records original locations, archive locations, sizes and SHA-256
checksums. Original files are retained byte-for-byte. Historical scripts retain
their original imports and are reference material, not a second supported service.
The notebook and FRPC executable remain local archives, excluded from Git.

Active shared computation was extracted to `mcp_tools/common/` and the mutation,
prediction and discovery workflow modules under `mcp_tools/tools/`. These files
contain no Gradio dependency. Frozen RSI audit capsules in `src/agent/harness/_legacy*`
retain their original locations and hashes; they are not obsolete UI assets.

Dated research logs and machine-specific notes are preserved in ignored
`research-notes/`; the release migration receipt is under `research_runs/`.
They are local evidence, not installation instructions.
