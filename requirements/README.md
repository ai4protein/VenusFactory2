# Dependency profiles

| Entry point | Use |
| --- | --- |
| `../pyproject.toml`, `../requirements.txt` | Existing platform environment; GPU wheels are selected by `install.py` |
| `macos.txt` | macOS dependency profile; Linux sandbox and GPU support differ |
| `service-security.txt` | Validated service versions, applied by the initializer together with the model compatibility constraints |
| `optional/cube.txt` | Optional CubeSandbox SDK; requires a separately configured local service |
| `optional/esm3.txt` | Isolated ESM3 preprocessing environment; audited separately from the platform |
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
and the compatibility `install.py` entry points
for the platform environment (Torch/PyG, MCP/FastMCP, Transformers). It does not
apply to independently versioned third-party environments. All platform dependency installers also apply
`service-security.txt`; the initializer rechecks dependencies when either constraints file changes. See
[installation](../docs/installation.md).

CI resolves both platform profiles and audits them against OSV. See the
[security review](../docs/security/dependency-review-2026-11-08.md#continuous-audit)
for the exact open model-runtime advisories and the strict audit command.

ESM3 structure-token generation runs in its own interpreter. The core platform
uses Transformers 5.10.4; ESM3's upstream Transformers 4 requirement is confined
to the optional runtime. Do not combine the two dependency profiles.
See [runtime setup](../docs/model-runtimes.md#esm3-structure-token-preprocessing)
and the [updated security review](../docs/security/dependency-review-2026-11-09.md).
