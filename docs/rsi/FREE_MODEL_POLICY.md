# Research model access

New live research requests are restricted by `configs/rsi/model_access.json` to
DMXAPI `glm-4-flash` and `unisound-u2-flash`. Their free status comes from the
user-provided list, not a live tariff lookup. Speech models are not admitted.
There is no fallback to another provider, model or protocol.

The common research gateway checks the policy before creating a live client and
again before each request. Paid aliases, different endpoints, unlisted protocols,
excess output budgets, and missing or malformed policies are rejected before
upstream transmission. A request records the policy digest. Explicit
`httpx.MockTransport` tests remain offline and are identified as such; there is no
environment variable or CLI switch that disables the live check.

New profiles in `configs/harness/free_text/` use a matched 4,096-token output
ceiling and no provider-specific thinking parameter. They cover Scientist and
Agent with Kimi Code and DeepSeek harness adapters. These are prepared profiles,
not evidence of function-call compatibility. Codex still requires the separately
implemented Responses protocol; these Chat Completions profiles do not establish
Codex compatibility.

The policy is included in new execution source snapshots. Historical frozen
snapshots and profiles are unchanged and remain available for offline audits.
Do not run historical paid-model snapshots. The policy controls the current
research gateways; it is not an account-wide billing control and does not modify
the separate deployment checkout or its environment.

Both free aliases returned HTTP 403 with `insufficient_user_quota` in the
2026-10-23 preflight. Resolve account access before further requests. The two
preflights produced no Scientist goals or autonomous models. After restoration,
validate one bounded tool-call request, then start a new frozen lifecycle pilot.
