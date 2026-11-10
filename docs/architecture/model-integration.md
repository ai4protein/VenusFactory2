# Integrating models and tools

## Choose the owner

| Model origin | Code owner | Artifact identity |
| --- | --- | --- |
| Upstream method | `mcp_tools/models/pretrained/` | Method, variant, upstream revision and weights |
| Factory-trained model | `mcp_tools/models/factory/` | Dataset, backbone, training strategy, version and weight digest |
| Scientist/RSI result | `mcp_tools/models/evolved/` | Author/owner, research run, scope, version, evidence, independent review and visibility |
| Non-model service | `mcp_tools/tools/` | Service/tool interface and configuration |

ProSST, ProtSSN, ESM-1v, ESM-IF1, ProteinMPNN and LigandMPNN belong to the same
pretrained family even when their implementations use different environments.
Vina is also an upstream computational method, although its `model_type` is `tool`.
`model_family` records origin; `model_type` records model kind; `category` records
scientific function; `runtime_id` selects execution configuration. None grants
permission to publish or execute.

## Execution contract

1. Declare a typed input schema, output format, resource class, timeout and online
   policy. Preserve stable public names; expose adjustable scientific parameters.
   Required parameter combinations belong in the shared input contract as well as
   execution validation. Agent, MCP and queue admission must expose and enforce
   the same alternatives; prompts do not replace missing schema constraints.
2. Add the implementation under its owning model family. Expensive dependencies
   load inside a claimed worker/subprocess, not when API modules are imported.
3. Put installations and weights outside source; add a disabled profile to
   `configs/model_runtimes.example.json`. Resolve paths through
   `mcp_tools.models.profiles.repository_path`; do not depend on the shell cwd.
4. Expose the callable through the MCP hub and server. Catalog discovery and
   execution must both enforce the same administrator and online policy.
5. Add Model Plaza/Workflow metadata where applicable. A UI card alone is not an
   executable model, and a registered private checkpoint is not a public model.
6. Validate real bounded inference, parameter forwarding, missing dependencies,
   disabled-tool denial and owner isolation. Record input provenance and output
   artifacts; do not equate a smoke test with scientific validation.

## Factory and RSI artifacts

Factory training writes a checkpoint plus configuration and provenance to the
owner's runtime storage. Pretrained backbones and task checkpoints are distinct:
DeepSol and another solubility dataset may share ESM2-650M while producing different
model IDs, weights and scopes. Existing `ckpt/` distributions remain supported.

Training merges dataset defaults before normalizing metrics and structure channels.
The CLI boundary accepts comma-separated strings or JSON lists and passes lists
to model code. Explicit CLI values override dataset defaults, while a preset owns
its sequence/label column mappings and selected structure channels. SES adapters
require at least one nonempty structure channel; malformed values fail before a
model or dataset is loaded.
Dataset presets in `src/constant.json` reference versioned JSON files under
`data/`. The layout check validates those paths and documents; a planned dataset
without a configuration is not an executable preset.

Scientist creates a repository-like deliverable with code, report, tests and
artifacts. Research completion and public admission are separate. Reproducibility,
interface validation, evaluation evidence, leakage checks and independent review
must pass before public tools can discover the result. The registry owns author,
version, protein/property/scenario and visibility; model source directories do not.

Agent discovers capabilities by search, reads selected interfaces and scope, then
calls through MCP. Do not inline thousands of model descriptions into a prompt.
Dedicated RSI search and unified search share protein/property relevance ranking.
Enablement, runtime compatibility and stage filters apply before stable pagination;
metadata order cannot push a named protein behind incidental description matches.
The RSI category returns concise registered-model cards. Both model-list and
capability search match the card's prediction description as well as its recorded
protein and property; adding a prediction request must not hide an available
predictor. Runtime compatibility and administrator switches still apply before
ranking. Describing a selected
card binds its immutable model and exact applicability scope for the current
discovery session; the exposed call then accepts scientific inputs only. Resolving
that call uses the original prediction tool, permissions and queue. A new session
must inspect the model again. Existing explicit entry/scope tool calls remain valid.

`mcp_tools/models/evolved/catalog.py` owns recorded discovery metadata. Its private
`.discovery/` cache survives MCP worker restarts and contains no model weights or
generated source. An authenticated record binds the library and registry file
identities; creation, removal, replacement or changed file metadata invalidates it.
Building or refreshing this metadata reads complete immutable model packets;
unchanged catalogs do not. Concurrent workers share that build, while runtime compatibility and administrator switches are
checked for each catalog response. Read-only or unsupported hosts retain the
original discovery path. Keep `.discovery/` outside session mounts and downloads.
This cache provides recorded descriptions only: selected-model inspection and
prediction keep their original content and scientific-evidence checks.

`agent.harness.capability_catalog` owns discovery for reviewed Scientist programs.
Dedicated search and unified tool/model search use its current-version and public
publication filters. Unified search ranks ID references with the existing bilingual
matcher, merges them with the other model families, and loads compact cards only
for the selected page. Inventory counts include enabled published programs. Search
returns bounded name, description and scope summaries; full validation, interfaces
and publication evidence stay behind selected-program inspection. The index is not
an admission receipt: description and execution still validate the original bundle.

Selected-model calls validate input with the original model interface before
joining the compute queue. Explicit prediction calls use the same input check
before numerical replay. Scope, data and evidence validation still run at execution.
Prediction results carry their verified scope, including target, units and applicable
conditions, bound to the inference receipt. Successful execution alone does not
establish a complete answer: report the values with their units and relevant limits.

The evolved-model delivery module owns prediction exports. Native executors call it
after the original tool returns, under the existing workspace lock. It checks the
session-bound receipt against normalized inputs, scope and output before writing
CSV, JSON and a report through the shared descriptor-anchored file API. The chat
stores a separate result card; model-authored answers and scientific history remain
unchanged. Reload/replay deduplicates cards by receipt identity, and downloads keep
the normal session authorization. Export failure retains the computed result without
requesting another inference.

## Preparing registered models for inference

Full Scientist publication verification replays training, scientific feedback and
historical model dependencies. Keep that audit in registration and explicit
verification; ordinary Agent prediction can use a host-prepared audit instead:

```bash
PYTHONPATH=src python scripts/rsi/scientist_model_library.py \
  --library research_runs/rsi/scientist_model_library_v1 \
  prepare-inference --entry-ids runtime/selected-model-ids.json
```

The input is a JSON array of existing entry IDs (1–256). Preparation groups IDs
from the same research goal, runs the original full verification, and checks that
all evidence is unchanged before and after it. It makes no LLM calls and does not
create a model or grant public visibility. Run it as the service owner after
publication or a source/runtime upgrade, outside request workers.

The host stores immutable signed records and a private signing key in the library's
`.inference/` directory. This directory must stay outside session/sandbox mounts,
downloads and source control. Version 2 audit records reference the existing
immutable model packet by its content identity, preserving the other verified
metadata without duplicating learned parameters. Prediction reloads that packet
and checks its identity and current registry binding. Earlier inline records
remain readable and immutable; source upgrades still require full preparation.
The 32 MiB audit-record limit remains in force.

Preparation also returns a `verification_runtime_id`, the content fingerprint of
verification code and its numerical environment. Audit pointers are stored per
fingerprint, so preparation from a new frozen source checkout can use the same
library without replacing the legacy reader's pointer. Each worker selects its own
runtime audit and still checks every evidence binding. An invalid selected audit
fails closed; another runtime's record cannot substitute for it. Old single-index
libraries remain readable with their original verification rules.

For an upgrade, prepare and test the target IDs from the new immutable checkout
before directing workers to it. Keep the old checkout available for workers still
using it; do not edit a running worker's source in place. Retaining an older audit
does not preserve access to changed or retired model evidence.

Evidence snapshots include complete original and
ancestral goal trees, registry bindings, declared source datasets, current
verification/model code and numeric package versions. Per-request validation
hashes actual file contents with four bounded file readers; it does not trust
timestamps or repeat baseline training. Preparing a later revision preserves the
earlier audit IDs referenced by existing usage receipts. Changed evidence, retirement, a changed index or an invalid signature
blocks prepared inference until a full audit passes again. Locks and generated
Python bytecode are excluded from content identity; symlinks and special files
are rejected even in those locations.

Unprepared models retain the original verification path. `resolve()` and the
`verify` CLI command always perform a scientific audit regardless of preparation.
Successful prepared predictions include an `inference_audit_id` in their ordinary
usage receipt. They preserve scope/input validation, ownership, administrator
policy, queue limits and prediction export behavior. A prepared private model is
still private and still requires the separate curated-publication process.
