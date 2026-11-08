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
do not route their internal dependencies through the legacy alias layer. Frozen
historical source snapshots are not rewritten.

Engine modules that build isolated execution bundles remain together: their
relative source dependencies are part of the runtime contract. Moving an engine
file requires updating and testing the generated execution bundle as well.

See [Scientist development and delivery](../../../docs/scientist-model-development.md)
and [repository ownership](../../../docs/architecture/repository-layout.md).
