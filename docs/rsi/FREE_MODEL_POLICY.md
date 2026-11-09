# Research model access

Live research uses the enabled free routes in the server execution catalog.
`configs/rsi/model_access.json` points to
`configs/harness/free_execution_catalog.json`; `VENUS_EXECUTION_CATALOG` can select
a server-owned catalog. Credentials come from the configured environment or
credential file. See the [backend configuration guide](../harness/backend_configuration.md)
for route, protocol and output-budget settings.

The common gateway checks model access before opening a client and before each
request. It records the policy digest, transmitted request hash, actual usage
and provider failures. A native harness receives a temporary relay token rather
than the upstream API key. Role requests and harness requests share the run's
physical request budget.

## Free routing

DMX entries must be explicitly enabled with zero configured input and output
prices. OpenRouter additionally requires an explicit free route and enforces
zero maximum provider prices, no provider fallback and support for required
parameters. Fixed-model routes validate the returned model identity; a successful
OpenRouter response must attest zero billed cost.

The shared OpenRouter policy applies to Chat Completions, native Responses and
the Codex/Claude protocol bridges. It omits the optional
`parallel_tool_calls=true` SDK default, which can otherwise exclude a provider
that has not advertised that parameter. An explicit `false` value is retained:
a caller's serial-execution constraint must not be weakened for compatibility.
Request evidence reflects the normalized payload.

OpenRouter's [provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters)
defines `require_parameters`. Tool schemas, model identity, price limits and
explicit execution constraints remain required.

## Protocols and verification

Kimi Code and DeepSeek Harness use Chat Completions. Codex can use native
Responses or the explicit Responses-to-Chat bridge; Claude Code uses the
Messages-to-Chat bridge for these free routes. The bridges support text, tool
declarations and paired tool history. They validate tool responses before
execution and preserve provider-reported usage.

Static configuration readiness does not establish live compatibility. Verify
each selected harness/model combination with a bounded tool call before a
research campaign. Store the source revision, protocol, limits, raw failure
evidence and numerical acceptance checks with that campaign. Exhausting a run
budget is distinct from provider rate limiting; neither implies completion.

Historical snapshots retain their original bindings and evidence. New snapshots
include the current policy. Offline tests with `httpx.MockTransport` do not
contact providers or establish live availability.
