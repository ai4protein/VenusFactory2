# MCP tools

The MCP service is the shared capability boundary for Chat and Model Plaza.
Public tool names and model IDs are independent of Python package paths.

| Directory | Responsibility |
| --- | --- |
| `models/pretrained/` | Upstream methods: ProSST, ProtSSN, ESM-1v/IF1, ProteinMPNN, LigandMPNN, AntiBMPNN, BindCraft, BoltzGen, DiffDock, Vina, ADMET and S4PRED |
| `models/factory/` | Platform training, evaluation and prediction; dataset-specific checkpoints such as DeepSol · ESM2-650M |
| `models/evolved/` | Internal Scientist/RSI evidence and offline replay; public case preview/download lives in the RSI Plaza API, not MCP prediction |
| `models/profiles.py` | Administrator-owned runtime configuration and repository-relative paths |
| `models/execution.py` | Shared bounded subprocess dispatch, without importing model weights into the API |
| `scheduler/` | Tool discovery, resource queues, async jobs, MCP client and worker entry points |
| `tools/` | Search, databases, remote protein services, files, skills, sandbox and non-model calculations |
| `common/` | Shared sequence, structure, input/output and prediction utilities |

`runtime/<mode>/` contains mutable state. External installations and weights belong in
`runtime/local/model-runtimes/` (or a configured external directory), never in an API
package. Online may use an administrator-enabled installation without sharing local
conversations, jobs or uploads. `ckpt/` remains the compatible store for existing
factory checkpoints; it is not a second source tree.

MCP exposes both functional category and `model_family` (`pretrained`, `factory`,
`rsi`, `foundation`, `utility`). A runtime is an execution environment, not a model
identity or a publication decision. Factory checkpoints record dataset, backbone,
training method, version and artifact digest; RSI artifacts additionally retain
their independent review and visibility records.

See [model integration](../docs/architecture/model-integration.md),
[service startup](../docs/architecture/services.md), and
[repository layout](../docs/architecture/repository-layout.md).

## Initialization

Use `python3 initialize.py plan`, `python3 initialize.py install` or `python3 initialize.py doctor` from the repository root.
The declarative catalogue is `configs/bootstrap.json`; implementation lives in
`scripts/bootstrap/`. See [installation and health checks](../docs/installation.md).

## Installation families

`install_full.py` installs the core plus optional `external`, `factory`, `rsi`, and `harnesses` families.
Database queries, web search, public protein services and basic analysis belong to core.
Installation recipes are in `configs/bootstrap.json` and `scripts/bootstrap/`; runtime enable switches remain administrator-owned.
RSI library selection is stored in ignored `runtime/local/components/rsi.json`; existing explicit configuration takes precedence.
