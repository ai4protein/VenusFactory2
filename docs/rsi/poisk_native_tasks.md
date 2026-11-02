# Native paired-antibody model tasks

`generated_antibody.v1` carries paired variable heavy/light chains through the existing Scientist development, independent model assessment, scoped registry and Agent inference interfaces. It is a scientific plugin shared by harness adapters. Claude Code, Codex and DeepSeek MCP connection exports do not select an inference model or imply that a native client has passed conformance testing.

## Current source and applicability

The first native tasks retain the partitions from the audited [POISK v1.1 source](https://zenodo.org/records/21775623) in `research_runs/rsi/20261028_poisk_adapter_v1`. Four conventional-antibody SPR cohorts cover EGFR, HER2, MET and VEGFA, with 29, 33, 48 and 30 observations. The endpoint is negative-log source-reported molar concentration. The extraction does not establish that these unprefixed measurements are KD. Explicit KD, IC50 and EC50 are distinct scope endpoints and cannot be pooled under a new name.

The interface also validates VHH inputs, but the current four real cohorts are conventional VH/VL pairs. VHH lifecycle checks use synthetic fixtures. Target sequence constructs are bound by source hashes; the numerical representation does not contain antigen sequence embeddings. Experimental conditions absent from the source remain `unreported_in_source`.

## Numerical and identity contracts

Inputs retain observation ID, target accession, antibody-pair digest, heavy chain, light chain, target-construct hash, sorted patent IDs, source independence group and numerical values. The frozen 118-dimensional representation contains separate heavy/light amino-acid composition and hashed dipeptides, lengths, light-chain presence and the original 11-target panel encoding. Every row is re-encoded and compared with its numerical values.

Partitions cannot share an observation, antibody pair, exact chain, patent or declared group. Paired chains within five percent global edit distance are additionally rejected across partitions. Shared execution and catalog admission check these dependencies across tasks. Patent-family resolution beyond the recorded US patents remains incomplete; these checks do not establish broad homology independence.

The development split uses training-group identities and counts, never label magnitudes. The Scientist receives the training workspace and its internal feedback. Independent acceptance and final labels remain host-side. Only an assessed model can be registered; a later inference session must reproduce its predictions before it counts as reused.

## Fixed controls

The native suite implements training mean, standardized ridge (alpha 1), random forest (100 trees, depth 10, leaf minimum 3), standardized SVR (C 1, epsilon 0.1) and paired-chain edit-distance nearest neighbor. Nearest-neighbor ties average the corresponding training labels. Parameters are fixed before independent evaluation. This recipe differs from the earlier feasibility controls and its results must not be substituted into that historical run. Published affinity-specific predictors remain a separate pending comparison.

## Preparation and execution

Use `scripts/rsi/prepare_poisk_tasks.py` with `--source`, `--feasibility` and a fresh `--output` directory. Preparation binds source observations, original task declarations, grouping evidence and the frozen encoder. It preserves original partitions, creates no model calls and registers no models. `scripts/rsi/run_shared_science_goal.py` consumes the resulting `tasks.json` through the existing lifecycle; shared admission must succeed first.

The four small real cohorts are valid individual exploratory tasks but do not meet the stricter large-catalog row and group floors. Do not relax those floors to increase the model count. Scale requires additional source-supported cohorts and adequate independent groups.

Synthetic integration tests exercise numerical development, independent registration, fresh MCP reuse, Agent model cards, ordinary and evolving goals, shared task admission and catalog construction. Real-data internal controls and MCP inspection are engineering evidence, not autonomous model acquisition or evidence of recursive improvement. Live experiments must use an explicitly admitted free model, retain failed attempts and count distinct qualified scopes only once.
