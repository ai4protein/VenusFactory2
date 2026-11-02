# Research configuration

The web services use the shared harness catalog, `.env`, and mode-specific runtime
state. Research campaign profiles here are only loaded by explicit research CLIs.

`engineering.example.json` describes the older Kimi engineering probe interface.
Copy it to a local profile and replace the three installation paths before using
`scripts/rsi/run_experiment.py --config <profile>`. Choose a provider configuration
that satisfies the requested model and budget policy. The template does not
contain credentials or start a model call. Historical `engineering_*` and
`generated_models_glm*` site profiles are ignored, preserved locally, and are not
portable installation defaults. Paper/campaign reproduction also requires its
separately acquired data and frozen evidence; a source clone does not include it.

`paper_steps.example.json` is an empty host registry template. Local
`paper_steps.json` admits checksum-bound definitions stored in ignored
`research_runs/case_definitions/`. Existing legacy definition paths remain readable.
Do not commit training measurements, hidden evaluation labels, or run receipts.
