# Dependency profiles

| Entry point | Use |
| --- | --- |
| `../pyproject.toml`, `../requirements.txt` | Existing platform environment; GPU wheels are selected by `install.py` |
| `macos.txt` | macOS dependency profile; Linux sandbox and GPU support differ |
| `service-security.txt` | Validated service versions, applied by the initializer together with the model compatibility constraints |
| `optional/cube.txt` | Optional CubeSandbox SDK; requires a separately configured local service |
| `optional/e2b.txt` | Optional E2B SDK; requires the configured local E2B infrastructure |

The default session sandbox is bubblewrap, a Linux system dependency. It does not
require either optional SDK. Moving these profiles does not select a new provider
or enable host/cloud fallback. See [sandbox configuration](../docs/session-sandbox.md).

BindCraft, BoltzGen, DiffDock and other conflicting upstream environments stay in
their own runtime venvs. Do not install all upstream requirements into the API
environment. Runtime profiles point to their Python executable and verified assets.

The existing core environment still includes scientific Python dependencies;
this layout change does not claim that the API can run from a minimal web-only
requirements file. Dependency removal requires import and execution verification.

`bootstrap-constraints.txt` is the compatibility envelope used by `initialize.py`
for the platform environment (Torch/PyG, MCP/FastMCP, Transformers). It does not
apply to independently versioned third-party environments. The initializer also applies
`service-security.txt` and rechecks dependencies when either constraints file changes. See
[installation](../docs/installation.md).
