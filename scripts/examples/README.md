# Local tool recipes

These examples were consolidated from `script/tools/`. They are grouped by task:
database access, sequence design, evaluation, mutation scoring, file conversion,
property prediction, literature search and model training.

Edit data, model and output settings before running a recipe. Inputs under
`example/` are fixtures; place new outputs in `runtime/local/files/cli/<run>/`
and trained model deliveries in `runtime/local/models/`. A recipe can contain
several training runs and is not a cost-free smoke test.

Use Model Plaza, Chat or queued MCP tools for user-facing jobs. Direct developer
recipes do not provide per-user scheduling or online quotas. Do not use them as
HTTP handlers or make them available as host shell commands to an Agent.

`python scripts/check_layout.py` checks entry-point paths. Offline validation
does not establish that each optional model, dataset or remote API is installed.
