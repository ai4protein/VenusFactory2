# Repository layout

One source checkout runs local and online services with separate state roots.

```text
frontend/                       React UI, styles, assets and frontend tests
backend/                        HTTP API, auth, quotas, sessions, persistent jobs
  web_v2/                       Shared queue, worker pool, events and API modules
mcp_tools/                      Shared scientific capability service
  mcp_server.py                 MCP HTTP entry point
  scheduler/                    Discovery, admission, async jobs and workers
  models/
    pretrained/                 Upstream prediction/design/docking methods
    factory/                    Platform training and checkpoint prediction
    evolved/                    Reviewed RSI model/capability access
    profiles.py                 Runtime configuration (metadata, no weights)
    execution.py                Isolated environment command dispatch
  tools/                        Search, databases, files, sandbox, calculations
  common/                       Shared input/output and scientific helpers
src/agent/                      Agent, Scientist, Researcher orchestration
  harness/                      Harness adapters and session sandbox adapters
  rsi/                          Research lifecycle, evaluation and admission
  skills/                       Native SKILL.md packages
    vendor/scientific-agent-skills/  Intact optional upstream Git submodule
configs/                        Runtime, queue, install and research configuration
requirements/optional/          Opt-in Cube/E2B dependencies
requirements/macos.txt         macOS dependency profile
scripts/                        Service launch, installation, migration and examples
tests/                          Backend, MCP, runtime and agent tests
docs/manual/                    Canonical bilingual platform manual
docs/wiki/                      Generated manual pages and installation guides
docs/images/                    Current UI screenshots with capture provenance
archive/                        Historical Gradio, FRP and notebooks
runtime/local/                  Local sessions, queue, artifacts, installed models
runtime/online/                 Online sessions, queue, artifacts and policy state
research_runs/                  Research evidence and immutable historical runs
ckpt/                          Existing factory model weights (compatibility store)
data/, example/                 Shared datasets and examples
```

## Ownership rules

The frontend renders state and calls the API. The API authenticates owners, validates
inputs and enqueues work. Independent workers claim leased jobs and persist events.
Scientific capabilities are served through MCP; model packages own implementation
and runtime adapters. Agent code plans work and calls tools; it does not contain
another copy of model inference code.

Model origin is separate from scientific function. For example, ProteinMPNN and
LigandMPNN are both pretrained methods; DeepSol · ESM2-650M is a factory-trained
task model; a reviewed Scientist solubility predictor is an RSI artifact. All use
the same discovery, queue and access-control path. An unreviewed research result
must not become public merely because its code executed successfully.

Skills are instructions, not executable plugins. Keep upstream licenses and the
submodule revision. Native IDs take precedence; collisions use `sas_<name>`.
The generated `.kimi-code/skills/` directory is only a projection, not a source tree.

## Configuration and data

`.env` stores deployment credentials and environment overrides; the gitignored
`model-config.yaml` remains the compatible LLM endpoint configuration. Runtime
installations use `configs/model_runtimes.json`; copy the disabled example only on
a new installation. Administrator availability and quotas are persisted per mode.
Dated work logs and machine-specific profiles are archived locally under ignored
`archive/research-notes/`. Experiment evidence is preserved under `research_runs/`.
Only reusable protocols and credential-free templates belong in source control.

Do not merge local and online state. Do not move a Python venv after installation:
its entry-point shebangs can contain absolute paths. Installations can be shared
read-only; task directories, ownership databases and secret keys cannot.

Optional Cube and E2B adapters are not the active bubblewrap provider. Install their
dependency profiles only when an administrator has configured the corresponding
local service. Gradio and FRP remain archived, outside active imports.

Run `python scripts/check_layout.py` and `python scripts/validate_skills.py` after
changing package ownership. Preserve public tool names and model IDs. Update CLI
recipes, runtime script paths, tests and the manual alongside any moved package.

## Initialization

Use `python3 initialize.py plan`, `python3 initialize.py install` or `python3 initialize.py doctor` from the repository root.
The declarative catalogue is `configs/bootstrap.json`; implementation lives in
`scripts/bootstrap/`. See [installation and health checks](../installation.md).
