# Scientist model library

`ScientistModelLibrary` provides an owner-scoped model catalog above completed
Scientist goals. It accepts the generated regression, feature regression and
classification interfaces admitted by `model_tasks`.

Publication independently checks the frozen goal and task, the native four-role
execution, candidate artifact, training replay, validation threshold and actual
inference in a distinct Scientist session. It never scores future labels. The
native publication auditor covers `expert-kimi-code`, `expert-deepseek-harness`
and `expert-codex`. Codex has passed real-CLI four-role tests with local model
fixtures; an external-model Codex publication/reuse campaign is still unverified.
See [CODEX_RESPONSES.md](CODEX_RESPONSES.md). LLM/provider bindings remain independent
of the harness. Fixture providers cannot publish scientific entries.

## Identity and counts

The catalog groups executable program syntax, scientific scope, frozen training
data and model interface. Repeated registration, runtime-only upgrades, different random seeds, display
names, comments and docstrings do not increase the count. Additional verified
origins and fitted artifacts remain recorded separately. Variable renaming can
still change a program hash: catalog diversity is not algorithmic novelty or
evidence for RSI efficacy. Stage coverage, task coverage and program diversity
must be reported alongside the count.

`list` reports indexed metadata with `verified_entries: null`. `list --verify`
replays the current canonical origin of every entry before reporting a verified
count. Retired, changed or missing origins fail closed. Supporting origins are
historical records; they do not silently replace an invalid canonical origin.
Legacy goals retain their original scope and source provenance. They are listed
with unspecified stage and no inferred paper attribution.

## Runtime compatibility

Historical native Scientist records can select the fixed pre-Responses audit
capsule only when their recorded source identities match its explicit pins.
The capsule executes the original verifier and preserves its checks; it never
loads code from a run directory or updates original evidence. Unknown revisions,
changed capsule files and altered artifacts are rejected. The live numerical
model implementation remains independently checked as described below.


The canonical model must remain compatible with the live numerical sandbox and
assessment implementation. A changed implementation causes publication or
resolution to fail; original policies and results are not rewritten. The explicit
`migrate` command supports the archived pre-numerical-evidence regression
interface. It verifies the original native Scientist execution and frozen
interface, replays training and validation in the current sandbox, and requires
the fitted model, validation metrics and fresh-session predictions to match.
It creates a separate current numerical assessment and migration certificate;
the original goal, policy, failures and results remain unchanged. Unsupported
interfaces, altered outputs or retired models are rejected. Migration does not
generate a new autonomous model or score future labels. Repeating a migration
does not increase the model count. A canonical runtime update retains the old
immutable entry and records its successor; it does not choose a different goal
or model based on evaluation scores.

`publish-goal` submits every accepted generation of a completed goal and records
each publication, duplicate or rejection in a separate immutable attempt report.
It is idempotent for model counts, retains unsuccessful attempts, and never
counts ingestion as a new autonomous execution. The command exits nonzero when
any model is rejected, so a scheduler cannot mistake partial ingestion for
success. Both `run_science_goal.py` and `run_science_evolution.py` accept
`--model-library PATH`: after a successful run or resume, they invoke this
publication hook. Status, create and pause actions never publish. A publication
failure preserves the completed goal, retains the separate rejection report,
and returns an error to the caller.

## Use

Run from the research repository with `PYTHONPATH=src` and the Venus Python:

```sh
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 publish --goal PATH --model-id ID
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 publish-goal --goal PATH
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 list --verify
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 inspect --entry-id ID
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 migrate --goal PATH --model-id ORIGINAL_ID
python scripts/rsi/scientist_model_library.py --library research_runs/rsi/scientist_model_library_v1 predict --entry-id ID --inputs inputs.json --scope scope.json --session-id agent-session --output predictions.json
```

The library and its origin goals must be beneath the same owner directory.
Prediction requires the exact registered scope, including feature encoder
identity where applicable. It strips target labels, revalidates origin evidence,
uses the registered inference sandbox, and records a separate usage receipt.
An ordinary Agent inference is not a new autonomous discovery or a fresh
scientific generalization experiment. Existing goal histories remain intact.

This module is the shared model service boundary. Automatic campaign scheduling,
web discovery and model-library tools in Agent conversations are separate
integration steps. A thousand allocated tasks or test fixtures do not satisfy a
thousand verified-model objective.


## Agent access

Local Agent chat (LangChain) and the existing MCP server expose
`list_evolved_models`, `describe_evolved_model` and `predict_with_evolved_model`.
They use `research_runs/rsi/scientist_model_library_v1` by default. An operator
may set `VENUS_SCIENTIST_MODEL_LIBRARY`; tool arguments cannot select a path.
These tools are disabled in online mode because this catalog belongs to the
local research owner.

Search returns names and short descriptions. `describe_evolved_model` reads the
selected publication, its input contract, full scope, recorded metrics and
`scope_ref`. Pass that reference with `entry_id` and up to 256 unlabelled records
to `predict_with_evolved_model`. The reference binds the selected entry and its
complete scope, so callers do not need to retype the scope object. Existing calls
with the full inline `scope` remain supported.

Prediction checks current publication bindings, evidence, runtime policy and
scientific inputs. A scope reference does not bypass those checks. Target labels
are removed before inference. Responses contain no source code, dataset records,
internal paths or LLM settings. These tools do not train or publish models.

The MCP wrapper binds usage to the actual transport session. The LangChain
wrapper uses the host's configurable session_id or thread_id; when neither is
present it creates a tool-request identifier. Such ordinary Agent usage is not
counted as a fresh-session Scientist experiment or a new autonomous model.


## Scientist experience transfer

Scientist task admission can reuse verified library implementations as bounded development context. Every recipient must fit, validate, register and demonstrate fresh-session reuse for its own model. The ordinary Agent inference tools remain unchanged. See [Cross-task model experience](LIBRARY_EXPERIENCE.md) for admission and provenance checks.
