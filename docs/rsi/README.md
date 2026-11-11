# Research protocols

The active acquisition plan is the [property-first RSI programme](RESEARCH_PROJECT_MODELS_10000.md).
New breadth-oriented cohorts use its property policy and report unfilled coverage explicitly.

This directory contains reusable experiment protocols and scientific contracts.
For the running software, use the [repository layout](../architecture/repository-layout.md),
[model integration guide](../architecture/model-integration.md),
[Scientist tutorial](../manual/en/scientist.md) and
[service guide](../architecture/services.md).

Dated work logs, revision backups and machine-specific experiment profiles are
preserved locally under ignored `archive/research-notes/`. Measurement tables,
frozen cases and run evidence belong in ignored `research_runs/`; they are not
part of a source installation or a claim about current scientific performance.

The host-admitted computation registry `configs/rsi/paper_steps.json` is local
configuration. Start from `paper_steps.example.json` when needed. New definitions
are stored in `research_runs/case_definitions/` and verified by checksum; legacy
`configs/rsi/paper_step_definitions/` entries remain readable for existing sites.
