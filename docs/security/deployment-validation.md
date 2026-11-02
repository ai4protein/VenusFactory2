# Deployment validation — 2026-11-03

This record covers the RSI repository reorganization and the online/local PBS
service update. It is a validation record, not a claim that all dependencies or
every model runtime are vulnerability-free.

## Verified scope

- Source and all reachable Git history passed the configured Gitleaks scan.
  The only dataset exception matches exact public SHA-256 checksum declarations,
  including their old and reorganized paths; it does not exempt whole folders.
- Repository layout, skill discovery and manual/wiki synchronization checks passed.
  Runtime environments, weights, sessions, private configuration, research data
  and generated builds are excluded from version control.
- Frontend: 97 tests passed, production build passed, npm audit reported no known
  findings after the dependency updates.
- Backend regression: 980 tests passed with six skips on the first run. One real
  worker test exceeded its 60-second cold-start deadline while other checks and
  environment preparation were running. Its complete module passed independently
  (23 tests) without changing its timeout or assertions.
- Harness and targeted RSI execution tests: 707 passed, one skipped.
- Deployment, runtime separation and per-mode Kimi settings: eight tests passed.
- Dual-service startup verified independent API/MCP ports and state roots.
  Online configuration access was rejected; admin endpoints required authentication.
  Kimi settings, MCP declarations and daemon state are separate between modes.

The earlier full RSI reorganization regression is complemented here by checks
against the patched Web/MCP dependency set. No paid model or scientific experiment
was invoked for this deployment validation.

## Dependency boundary and remaining work

`requirements/service-security.txt` records the patched service dependency versions
used for validation. Auditing that installed dependency overlay produced no known
findings. This narrower result must not be presented as a clean audit of the whole
scientific environment.

The legacy scientific environment contains Transformers 4.48.1 and Accelerate
1.13.0 advisories, including model/checkpoint loading and conversion paths. Its
Gradio installation is unrelated to the current React/FastAPI service and is not
started. Existing optional PyMOL/Numpy and legacy Gradio/Pillow dependency conflicts
also require separate runtime modernization. Major Transformers upgrades require
actual ESM/ProSST and checkpoint compatibility tests; they are not silently applied
during a Web release.

Public operation therefore keeps the existing backend capability policy, private
artifact ownership, ESM2-8M-only online training and isolated generated-code
execution. Arbitrary model repositories or unreviewed checkpoints must not be
treated as trusted runtime packages. Dependency auditing should continue separately
for each installable external/factory/evolved model runtime.

## Deployment behavior

The PBS supervisor binds APIs and MCP services to loopback, starts the public
forwarder only after readiness, and stops its own recorded descendants on exit.
Online and local modes share the allocation's CPU/GPU hardware but not session,
quota, queue or artifact state. Separate allocations are needed for independent
hardware budgets. See [the PBS guide](../deployment-pbs.md).
