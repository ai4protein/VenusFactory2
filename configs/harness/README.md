# Harness configuration

Harness adapters live in `src/agent/harness/`. Install optional clients with
`python3 initialize.py install --harnesses <id>` and inspect availability in the
administrator console. Supported client IDs are declared in `configs/bootstrap.json`.

Use the root `.env` for provider credentials and model IDs. See
[API configuration](../../docs/model-providers.md) for current provider variables,
protocol compatibility and online free-model policy. Runtime installs, credentials
and generated client configuration belong under ignored local state, not here.

Files in this directory are explicit configuration examples, not auto-loaded
deployment defaults. A historical experiment profile does not enable a provider
or grant permission to run paid inference. Machine-specific research notes are
preserved locally under `archive/research-notes/`.
