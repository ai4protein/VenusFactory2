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
    engine/                     Reusable execution and evaluation primitives
    artifacts/                  Model/method contract and selected releases
    datasets/, cases/            Data adapters and paper-specific reproductions
    campaigns/                  Multi-topic study orchestration
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
archive/                        Optional local history (ignored; absent from releases)
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

Chat streaming separates transport from display state. `frontend/src/lib/sse.ts`
owns event IDs and reconnects; `chatStreamState.ts` owns snapshot reconciliation
and message transitions. `useChatStream.ts` manages that state and the disposable
frame queue; `ChatPage.tsx` owns navigation and user actions. A new session, run or
user turn cannot inherit the previous streamed answer. Waiting and terminal
snapshots are authoritative, and leaving the view discards queued animation work
without cancelling the backend job.

Researcher keeps job ownership and recovery in `backend/web_v2/research_jobs.py`.
The long-form coordinator schedules retrieval and specialist writers; document
acceptance and rendering belong to `research_documents.py`. API views and ZIP
exports use that document layer, which does not call models or launch jobs.
Completed-chapter counts and delivery status are derived from current review
receipts for both views. Rechecking a report preserves its saved drafts, evidence,
request budget and queue lifecycle; it never rewrites the historical checkpoint.
Source-review receipts and revision feedback belong to `research_source_review.py`;
author and editor modules consume those receipts without owning the reviewer.
Chapter evidence selection and writer context belong to `research_chapter_context.py`.
Specialists receive topic-selected source excerpts under one serialized context
budget, with citation identities, query scope and material extraction limits.
Original records and full prediction arrays remain in the dossier; the independent
reviewer resolves its anchors against those originals.


Tool discovery separates session navigation from model metadata. Harness adapters
use the shared discovery broker to track which tool was inspected during a turn.
The MCP discovery package owns catalog lookup and the mapping of model, dataset
and task choices to a selected tool's declared input fields. Search returns names
and summaries; only description loads that interface. Availability is refreshed
at description and enforced again by the invocation path.

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

RSI experiment history is distinct from a deployable capability. A topic keeps
trial code and metrics in its workspace; the host packages the selected,
independently approved result. Later versions of the same topic and scope replace
the default discovery entry while retaining immutable earlier versions. See
[Scientist delivery](../scientist-model-development.md) and
[RSI source ownership](../../src/agent/rsi/README.md).

Do not merge local and online state. Do not move a Python venv after installation:
its entry-point shebangs can contain absolute paths. Installations can be shared
read-only; task directories, ownership databases and secret keys cannot.

Optional Cube and E2B adapters are not the active bubblewrap provider. Install their
dependency profiles only when an administrator has configured the corresponding
local service. Gradio and FRP remain in the local ignored archive and Git history, outside release sources and active imports.

Run `python scripts/check_layout.py` and `python scripts/validate_skills.py` after
changing package ownership. Preserve public tool names and model IDs. Update CLI
recipes, runtime script paths, tests and the manual alongside any moved package.

## Initialization

Use `python3 initialize.py plan`, `python3 initialize.py install` or `python3 initialize.py doctor` from the repository root.
The declarative catalogue is `configs/bootstrap.json`; implementation lives in
`scripts/bootstrap/`. See [installation and health checks](../installation.md).
