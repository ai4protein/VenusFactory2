# Configuration ownership

Use one `configs/` directory; do not recreate `config/`, `script/` or `Scripts/`.

| File / directory | Owner and purpose |
| --- | --- |
| `model_runtimes.example.json` | Versioned, disabled templates for installed model environments and weights |
| `model_runtimes.json` | Gitignored administrator configuration; enabled installations and bounded execution parameters |
| `install.json` | Installer dependency/GPU profiles |
| `job_queue.example.env` | Reference queue and worker limits, not an automatically loaded second environment file |
| `harness/` | Harness/free-route templates and research-specific profiles; select explicitly |
| `rsi/` | Research protocols and experimental configurations; not implicit production defaults |
| `../.env` | Deployment credentials and environment overrides, excluded from Git |
| `../model-config.yaml` | Existing gitignored LLM endpoint configuration; retained for launcher/admin compatibility |
| `../model-config.yaml.example` | Credential-free LLM endpoint template |

Runtime paths resolve against the repository root, independent of the current
working directory. Existing external absolute installation paths remain supported.
Use `VENUS_MODEL_RUNTIMES_CONFIG` to select a different administrator-owned profile.
Do not copy the disabled example over an already configured deployment.

Local and online use separate `runtime/local/` and `runtime/online/` state. The
administrator's per-mode tool switches are persisted in that mode's state store;
they are not model installation files. A verified installation can be disabled
without deleting its weights, and enabling a card does not install missing assets.

## Optional and historical settings

Cube and E2B remain opt-in sandbox adapters. Their dependency files are
`requirements/optional/cube.txt` and `requirements/optional/e2b.txt`; neither is
part of the active bubblewrap setup. Do not install them for ordinary session code.
Do not silently switch sandbox provider when its service is unavailable.

Historical experiment configs are kept for reproducibility. A provider/model
listed in a dated research profile is not necessarily enabled in online routing.
Online admission continues to enforce its configured free-model catalog, shared
daily quota, upload cap and compute policy independently of those research files.

Cleanup removes a setting only after its loader, examples and runtime callers have
been checked. Generic upstream environment forwarding and historical evidence can
make a simple text search insufficient. Never include actual provider keys in
documentation, test logs, screenshots or installation receipts.

Validate: `python scripts/check_layout.py`; inspect effective deployment paths:
`python scripts/serve.py --mode local --check` and `--mode online --check`.

## Initialization

Use `python3 initialize.py plan`, `python3 initialize.py install` or `python3 initialize.py doctor` from the repository root.
The declarative catalogue is `configs/bootstrap.json`; implementation lives in
`scripts/bootstrap/`. See [installation and health checks](../docs/installation.md).

## Installation families

`install_full.py` installs the core plus optional `external`, `factory`, `rsi`, and `harnesses` families.
Database queries, web search, public protein services and basic analysis belong to core.
Installation recipes are in `configs/bootstrap.json` and `scripts/bootstrap/`; runtime enable switches remain administrator-owned.
RSI library selection is stored in ignored `runtime/local/components/rsi.json`; existing explicit configuration takes precedence.

## Research data boundary

`rsi/paper_steps.example.json` is a versioned empty template; `rsi/paper_steps.json`
is the ignored host registry. Store new definition payloads in
`research_runs/case_definitions/`, with their checksums in the registry. The loader
still accepts the old `configs/rsi/paper_step_definitions/` location. Do not commit
measurement tables, frozen test labels or per-machine experiment settings.
