# Tool and model administration

Open `/admin`, `/en/admin` or `/zh/admin` and sign in with an existing administrator account. Local Settings includes an Administration link; online Settings includes neither Administration nor Environment variables. The direct online admin URL still requires authentication.

## Availability controls

Tools & models lists native/MCP tools, Scientist workspace tools, built-in models, external runtimes, admitted Scientist capabilities and community models. Search and filter by type, function or switch state. The switch applies to the current deployment only. Missing weights, runtime configuration, publication review, online restrictions and budgets remain independent requirements; enabling a switch never overrides them.

- Local policy: `runtime/local/state/capabilities.sqlite3`.
- Online policy: `runtime/online/state/capabilities.sqlite3`.
- A custom `VENUS_RUNTIME_ROOT` changes the base, retaining the two mode subfolders.
- Tool aliases share one switch. Model variants have stable, separate switches.
- A disabled model stays visible in Model Plaza. Its prediction action is unavailable. Hovering or clicking its card explains availability.
- Tool discovery omits disabled tools. A single-model tool disappears when its model is disabled; tools serving multiple variants validate the selected variant on each call.
- Agent, Scientist, Researcher, MCP, legacy HTTP routes, direct prediction adapters and worker execution read the policy again. Cached conversations and queued jobs cannot rely on their earlier availability decision.
- A running computation is not forcibly killed by changing an availability switch. Use its job cancellation action to stop it. Subsequent calls and jobs not yet executing are rejected.

Updates use a transaction and revision check; simultaneous administrator edits cannot silently overwrite one another. Audit records contain the administrator and affected key. Policies persist across API and worker restarts. No model weights are loaded merely by enabling a switch.

## Environment configuration

Environment configuration is available inside the authenticated administrator page in both modes. The online public `/api/settings/env` endpoint remains forbidden and its old UI route redirects to Chat. Existing credential values are masked; replacement is explicit, unchanged masks preserve the original value. Values never enter the audit log. The editor uses the deployment `.env` file, which is shared if both previews run from one checkout. Environment changes take effect according to service startup/reload behavior; availability switches take effect immediately without restart.

Admin cookies are mode-specific, so local and online previews on the same host do not overwrite each other's sign-in.

## Validation

`tests/test_capability_policy.py` covers mode isolation, revision conflicts, stale references, per-model matching, visibility, direct calls, real FastMCP listing/call refusal, queue replay, legacy HTTP APIs, administrator authentication, cross-origin rejection and masked environment updates. The related queue, access, training-policy and model-platform regressions remain part of verification.
