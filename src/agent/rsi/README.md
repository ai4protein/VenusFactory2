# RSI implementation

Scientist runs the RSI process; a research topic produces a coherent project and
selected, reviewed capabilities. Experiments are evidence, not separate releases.

| Package | Responsibility |
| --- | --- |
| `engine/` | Reusable execution, evaluation, sandbox runtime and library primitives |
| `artifacts/` | Shared model/method contract, release selection and runnable packaging |
| `datasets/` | Public data adapters and task preparation |
| `cases/` | Paper-specific reproduction implementations |
| `campaigns/` | Batch orchestration and cross-topic study protocols |

Web Scientist orchestration and independent admission are in `../harness/`.
MCP exposes admitted capabilities through `../../../mcp_tools/models/evolved/`.
CLI entry points are classified under `../../../scripts/rsi/`; deployment state,
generated models and research outputs never belong in this source package.

New imports use the owning package, such as
`agent.rsi.engine.model_gateway`. `_compat.py` lazily maps previous flat Python
import names to the same modules for older callers. It does not duplicate code or
eagerly load model libraries. Modules within the same engine use relative imports;
do not route new dependencies through the legacy alias layer. Published scientific
modules whose source bytes are part of a saved contract retain their recorded
imports. Historical source snapshots are not rewritten.
Missing flat-layout source references use the same alias table within their
original checkout; complete source hashes and symlink rejection still apply. An
existing file takes precedence, including when its contents fail verification.
Prepared inference records retain the declared path and bind its resolved path,
so a later change requires another full audit. New task preparation stores encoder
bytes with its immutable dataset artifacts instead of referencing an installation.

Published contextual feedback policies bind the complete bytes of their numerical
modules. Keep `context_controls.py`, `context_v2_controls.py`,
`context_v3_controls.py` and their feedback modules unchanged within those policy
versions, including import layout. Their separate implementations preserve saved
Scientist evidence. Consolidate numerical code in a new policy version with an
explicit migration; matching scores alone do not preserve the source contract.
`context_baseline_refit.py` remains independent for numerical verification.

`context_feedback_dispatch.verification_session` owns numerical replay reuse during
one model or shared-goal audit. Its bounded cache stores input digests and verified
source bindings; each reuse checks current source bytes. Nested donor verification
shares this scope, and the scope is discarded when the audit ends. New audits
replay their evidence independently. Publication, ownership, retirement and model
artifact checks remain in the registry and lifecycle modules.

Engine modules that build isolated execution bundles remain together: their
relative source dependencies are part of the runtime contract. Moving an engine
file requires updating and testing the generated execution bundle as well.

See [Scientist development and delivery](../../../docs/scientist-model-development.md)
and [repository ownership](../../../docs/architecture/repository-layout.md).
