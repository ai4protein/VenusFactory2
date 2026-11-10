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

Harness runtime discovery belongs to `agent.harness.installation`. Agent and
Scientist services pass their checkout root to that module; discovery must not
construct a research service or load an experimental CLI script. The module keeps
the existing environment > managed receipt > legacy installation/PATH precedence.
The research CLI retains a thin compatibility entry point. Frozen experiments
keep their original source snapshots and bindings; changing current discovery
does not migrate an earlier experiment.

Kimi protocol selection belongs to `agent.harness.kimi_adapter`, with the existing
`adapters.KimiAdapter` import retained for compatibility. The adapter probes the
configured executable before selecting the supported 0.19.2 server or 2.5.2 Web
protocol. The Web transport, transcript projection and ephemeral sandbox each
have one module; they share the existing MCP worker and audited filesystem mounts.
The Web server requires its private bearer token, and task instructions are bound
to its private `SYSTEM.md`. Reconnection replays transcript cursors without
resubmitting a prompt or duplicating tool events. Cancellation waits for the server
and MCP worker to stop, then removes temporary credentials.

Provider response receipts remain the source for reported token accounting. The
native session rollup is recorded separately: it can still be zero immediately
after turn completion and must not erase a nonzero streamed observation. Neither
runtime usage nor a connection check independently verifies provider billing.
The legacy RSI runtime and its published source bindings are not rewritten by a
protocol upgrade.

Prediction result plots belong to `mcp_tools/common/visualization.py`; API
workflows import those functions instead of maintaining their own renderers.
Legacy Quick and Advanced routes use canonical package imports. Compatibility
belongs at the route boundary, not in repeated import fallbacks.

Chat streaming separates transport from display state. `frontend/src/lib/sse.ts`
owns event IDs and reconnects; `chatStreamState.ts` owns snapshot reconciliation
and message transitions. `useChatStream.ts` manages that state and the disposable
frame queue; `ChatPage.tsx` owns navigation and user actions. A new session, run or
user turn cannot inherit the previous streamed answer. Waiting and terminal
snapshots are authoritative, and leaving the view discards queued animation work
without cancelling the backend job.
Sidebar history uses `useChatSessionHistory.ts` for cache and request lifecycle and
`chatSessions.ts` for metadata reconciliation. The API and page share that metadata
contract. Server ownership remains authoritative; transient submission rows stay
out of persisted history, and stale refreshes cannot overwrite later user actions.

Training configuration is deterministic: the calling agent supplies explicit
settings to the shared Factory core. Code execution uses the session-bound sandbox
tools. The retired `agent_generated_code` name returns a migration response and
never starts another model request or interpreter. Training modules do not load
LLM credentials or carry a second code-generation implementation; startup owns
environment configuration.

Custom Model headers and previews use the same authorized table reader in
`backend/web_v2/custom_model_api.py`. Folder discovery applies its depth limit
before entering subdirectories and prunes inaccessible or linked directories.
Listing model folders must not require walking another visitor's result tree.

Model training streams reconstruct curves and test metrics with
`backend/web_v2/training_progress.py`, a log parser without filesystem or queue
dependencies. The model API owns job access, output files and download URLs.
Reconnecting replays the saved worker log; parsing a metric never starts a task
or grants artifact access. Model routes share JSON event encoding in
`backend/web_v2/sse_events.py`; non-finite metrics become null values for browser parsing.

Researcher keeps job ownership and recovery in `backend/web_v2/research_jobs.py`.
The long-form coordinator schedules retrieval and specialist writers; document
acceptance and rendering belong to `research_documents.py`. API views and ZIP
exports use that document layer, which does not call models or launch jobs.
Completed-chapter counts and delivery status are derived from current review
receipts for both views. Specialist reports require independent source review even
when an older checkpoint has no review flag; a missing receipt or chapter identity
leaves the draft unaccepted. Writers, repairs and recovery share the document
layer's structural checks before source review. Rechecking a report preserves its
saved drafts, evidence, request budget and queue lifecycle; it never rewrites the
historical checkpoint.
Source-review receipts, revision feedback and bounded recovery belong to
`research_source_review.py`; author and editor modules consume those receipts
without owning the reviewer. Response decoding, complete and partial assessment
validation belong to `research_source_validation.py`. That pure module checks
coverage, original-source anchors and measured values without calling models or
changing report state; normal responses and retry results use the same validator.
Explicit measurement checks belong to `research_measurements.py`; values retain
their dimensions and unit conversions. Changes to acceptance policy invalidate
older receipts while preserving their original review evidence.
Chapter evidence selection and writer context belong to `research_chapter_context.py`.
Specialists receive topic-selected source excerpts under one serialized context
budget, with citation identities, query scope and material extraction limits.
Passage selection for both writing and review belongs to
`research_evidence_passages.py`. Its shared tokenizer retains scientific
abbreviations and splits structured field names for retrieval. Original records
and full prediction arrays remain in the dossier; the independent reviewer
resolves its anchors against those originals.


Tool discovery separates session navigation from model metadata. Harness adapters
use the shared discovery broker to track which tool was inspected during a turn.
The MCP discovery package owns catalog lookup and the mapping of model, dataset
and task choices to a selected tool's declared input fields. Search returns names
and summaries; only description loads that interface. Availability is refreshed
at description and enforced again by the invocation path.
Session-bound argument preparation and exposed input schemas belong to
`src/agent/harness/session_boundary.py`. Workers call that boundary instead of
importing one another. `events.py` owns the JSON event transport shared by the
worker entry point and its adapters; tool results and diagnostics retain separate
streams.

The selected tool's own input model is validated through
`mcp_tools/common/tool_arguments.py` after session paths are bound and again at
MCP admission. Native harnesses share this boundary. The common contract module
also owns closed-surface field checks and bounded feedback, including declared
fields of a malformed nested object. The session boundary owns path binding and
server-assigned workspace policy. Feedback never copies submitted values,
undeclared field names or validator traces; no compute job is created for invalid
arguments.

Model origin is separate from scientific function. For example, ProteinMPNN and
LigandMPNN are both pretrained methods; DeepSol · ESM2-650M is a factory-trained
task model; a reviewed Scientist solubility predictor is an RSI artifact. All use
the same discovery, queue and access-control path. An unreviewed research result
must not become public merely because its code executed successfully.
Evolved-model calls can carry a compact `scope_ref` from their selected
model description. `mcp_tools/models/evolved/scope.py` binds it to the immutable
entry and complete scope before numerical execution; the library still owns
evidence verification and prediction receipts.

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
