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

Scientist creates a repository-like deliverable with code, report, tests and
artifacts. Research completion and public admission are separate. Reproducibility,
interface validation, evaluation evidence, leakage checks and independent review
must pass before public tools can discover the result. The registry owns author,
version, protein/property/scenario and visibility; model source directories do not.

Agent discovers capabilities by search, reads selected interfaces and scope, then
calls through MCP. Do not inline thousands of model descriptions into a prompt.
