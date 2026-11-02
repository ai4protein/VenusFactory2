# Codex execution and Responses configuration

The research host supports `codex` in Agent mode and `expert-codex` in Scientist
mode. Scientist retains the PI, CB, MLS and SC sequence: PI and CB plan through
the common gateway, MLS runs a native Codex session, and SC reviews the actual
tool results and artifact. All four roles use the same configured model and
total request budget. Scientific validation, model publication and future-task
evaluation remain the responsibilities of the existing host and lifecycle
controllers.

This implementation was checked against native Codex CLI **0.163.0**. The pinned
Linux runtime lives under `research_runs/rsi/envs/codex-0.163.0`; its package
layout must be retained. `VENUS_CODEX_BIN` can explicitly select another binary,
whose path and digest are recorded at preflight. Other versions need their own
conformance tests. No global Codex configuration or login is required by the
research worker.

## Explicit model binding

Codex uses the Responses protocol. An existing Chat Completions model binding
is not automatically converted. A host-approved provider must support
`POST /v1/responses`, the configured model alias, and the function-tool format
emitted by the pinned native runtime. Namespace tools are preserved on the wire
and normalized to the common MCP tool identity for accounting.

The profile must explicitly declare:

```json
{
  "model": {"protocol": "responses"},
  "budget": {
    "thinking": null,
    "temperature": null,
    "reasoning_effort": "low"
  },
  "request_recovery": "none"
}
```

This is a fragment of the existing profile schema; retain the full model,
credential-reference and resource-budget fields when constructing a profile.
`thinking` is a Chat Completions control. Setting it to null explicitly removes
it for Responses. `temperature: null` omits that sampling parameter. A declared
reasoning effort must be supported by the chosen provider and model. The host
does not choose another model, provider or credential when a request fails.

An optional model catalog entry can declare these settings as
`inference_controls`. Only `thinking`, `temperature` and `reasoning_effort` are
accepted there; model-call, tool-call, output-token and wall-time budgets still
come from the selected execution profile. Existing catalog entries retain their
previous behavior. The public compatibility matrix admits the two supported
protocols and reports incompatible combinations before execution. Static
admission does not establish live provider availability.

## Execution contract

The gateway validates a complete JSON or SSE response before releasing it to
the worker. It checks the exact model alias, completion status, event order,
function identities, paired output items, final arguments, text and usage.
Server-side conversation history, background execution and undeclared provider
tools are excluded from this bounded interface. A physical failed request still
consumes its allocation. Automatic provider retries are disabled.

The worker receives a short-lived local gateway token. Its fresh Codex home
enables only the task's admitted MCP functions, with explicit per-tool execution
permission. Shell execution, web search, subagents, hooks and native goals are
disabled for this worker; the Scientist controller owns the long-running goal.
The native runtime and capability bundle are mounted read-only, while the task
workspace is writable. The runtime shares the host network, as the existing
native research adapters do; this is not a claim of network isolation.

Native tool events are paired by identity and arguments. Generated-model tasks
use the existing generated-code result validator; supported paper-reproduction
tools use their own domain validator. The independent Scientist audit checks
role order, budget intervals, request settings, critic evidence and artifact
integrity. A critic decision alone cannot publish a model.

## Validation on 2026-10-23

Evidence: `research_runs/rsi/20261023_codex_transport_changes_v1/`.

- `host_fixture_v1/report.json`: the production host ran native Codex, completed
  one real MCP call and consumed its result in a second model response.
- `expert_host_fixture_v1/report.json`: the production Scientist host completed
  PI, CB, native Codex MLS and SC, using five model responses and one MCP call.
  The independent role, request, event and artifact audits passed.
- Both runs used deterministic **local provider fixtures**, with zero external
  model calls, publications or new autonomous models. They establish software
  integration, not scientific model quality or self-improvement.
- Earlier failed fixture attempts are retained. They exposed the native
  namespace format and MCP tool permission requirements before the successful
  runs. The first broad regression also retained an outdated test expectation
  that Codex was unimplemented; that expectation was corrected.

The current change does not establish live GLM Responses compatibility or a
complete Codex-generated model publication/reuse campaign. Kimi and DeepSeek
continue to use their existing Chat Completions route. Provider conformance and
scientific experiments require separate allocations using explicit bindings.

## Official protocol references

- [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Codex gateway compatibility](https://learn.chatgpt.com/docs/enterprise/gateway-compatibility)
- [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp)
- [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)

Historical model compatibility is checked separately from new execution. A
pinned pre-Responses verifier preserves original source bindings, while unknown
revisions remain rejected. All three previously imported library models passed
read-only verification after this compatibility change. No old model was
retrained for publication, newly registered or counted as autonomous development.
The failed first compatibility check and missing-dependency diagnostic are
retained alongside the successful `library_verify_compatibility_v2.json` result.
