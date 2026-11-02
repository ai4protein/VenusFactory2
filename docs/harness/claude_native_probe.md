# Native Claude Code integration evidence

The isolated research runtime pins the official Claude Code native client to 2.1.312 and verifies the package SHA512 and binary SHA256. A common scientific MCP connection exports the same capability for Claude Code, Codex and DeepSeek Harness; each client retains its own native execution machinery.

## Messages bridge and native events

`src/agent/harness/messages_bridge.py` implements an experimental text-and-MCP Messages interface over an explicitly admitted free Chat Completions endpoint. The host retains credentials, the exact model binding, physical request counts, output limits and free-provider response checks. It buffers and checks the complete upstream response before releasing any tool arguments. Unsupported content, unknown tools, incomplete responses, mismatched identities and inconsistent token evidence fail before execution. Native metadata and cache hints are not presented as equivalent upstream controls.

`src/agent/harness/claude_events.py` maps complete native assistant tool calls and user tool results into the existing event contract. Partial stream events do not duplicate actions. Session identity, call pairing, tool limits and terminal completion are checked. Client cost estimates and advertised model caps are not substituted for gateway usage or request limits.

The actual native client passed four network-isolated scenarios through this bridge: a successful two-request exchange with one real MCP execution; an undeclared provider tool; a truncated provider response; and an upstream503 response. Each failure stopped before MCP execution, with one physical mock attempt and no gateway budget-rejected retry. The bridge and related gateway suite passed80tests. These are synthetic provider responses, not live model capability measurements.

The experimental `ClaudeAdapter` now connects the prepared scientific worker, strict native tool list, complete-event normalization and cleanup to the common execution host. `MessagesBridge` serves role Chat and native Messages requests using the same physical request counter and free-provider checks. The expert-team dispatcher can explicitly exercise the adapter in conformance fixtures. Native host Agent, four-role Scientist and cancellation probes have passed. Normal registry selection remains unavailable: the bridge remains experimental and the separate Web Agent lifecycle regression still exceeds its unchanged cold-start budget. Official Claude gateway documentation does not support non-Claude model routing; this bridge remains experimental.

## Kimi startup limits

A four-task POISK pilot exposed native executor timeouts before model submission. The wrapper had a25second MCP wait and a35second server wait. A deliberate30second MCP startup test showed that increasing the outer wait alone was insufficient: the installed Kimi client also stopped connection after30000ms.

The current adapter now sets server readiness, the native per-server `startupTimeoutMs`, and wrapper MCP readiness within the remaining phase budget, with a180second startup cap. Failures identify their stage; a reported native MCP failure stops before prompt submission. Legacy direct callers retain their default limits unless they opt in. The final relevant regression suite passed84tests.

The second delayed-start run confirmed the180000ms native configuration but still failed at the inner SDK60000ms initialization-request timeout. Inspection showed the SDK connection lacked a request-timeout override. The new prepared-worker path resolves the tested failure: a real native probe with75seconds of injected delay loaded the same FastMCP process in128.14seconds, then completed native connection and tool discovery in0.97seconds. The82-test regression passed, including actual tool results and cancellation cleanup. Readiness explicitly distinguishes capability loading from protocol initialization. Both previous failures are retained; all three probes submitted zero model prompts. The terminal POISK pilot remains unchanged; a separately frozen four-target follow-up uses this repaired path.

Evidence and retained failures are under `research_runs/rsi/20261029_claude_native_v1/`. The POISK pilot's frozen code and outcomes are unchanged. No external inference calls were made by these harness tests.

Primary interface references: [Claude headless execution](https://code.claude.com/docs/en/headless), [Claude gateway support](https://code.claude.com/docs/en/llm-gateway), and [Kimi MCP configuration](https://github.com/MoonshotAI/kimi-code/blob/main/docs/en/customization/mcp.md). The installed Kimi binary was also checked: the per-server field exists even though the newer global environment-variable names are absent from that pinned binary.

Prepared-worker evidence: `research_runs/rsi/20261029_mcp_readiness_v1/`. Current experimental host work: `research_runs/rsi/20261029_claude_host_v1/`; consult its check and native-probe reports before interpreting availability.


## Host and free-provider closeout

The actual native host completed Agent and four-role Scientist conformance with two and five scripted physical requests respectively, each with one genuine MCP execution. A cancellation scenario retained its interrupted checkpoint and stopped both isolated processes. These are implementation fixtures, not autonomous training results. Expert fixture v1 had insufficient reserved request budget and is preserved; v2 passed after the fixture budget was corrected.

The first two live free DMX probes each made one request and received HTTP 200, but conversion rejected the returned model identity `u2-flash`. The exact alias is now accepted only for requested `unisound-u2-flash` at `https://www.dmxapi.cn/v1`. No provider request changes model, and response receipts retain both names. New tests reject other origins, requested models and returned models. The original failures and their response-shape diagnostic remain immutable.

The post-fix live probe status is **passed**. Total physical free requests across all three versions: **10**, within the original limit of 12; no paid fallback. Separate Agent and Scientist outcomes are in `live_free_native_v3/report.json`. This is a synthetic tool-use compatibility test and registers zero scientific models.

Whole-harness regression before the localized response-alias repair: `1 failed, 538 passed in 697.32s (0:11:37)`. Affected post-repair regression: `142 passed in 189.27s (0:03:09)`. The unresolved Web lifecycle startup failure is reported separately; the whole suite is not described as passing.

The separately frozen POISK follow-up completed 30 free requests and verified two exploratory acquired scopes (HER2/P04626 and MET/P08581), using one generated program fitted independently. Both passed fresh reuse; EGFR and VEGFA failures remain. They do not meet large-catalog counting floors and do not establish causal RSI or a predictive advantage. No manuscript was changed in this increment.
