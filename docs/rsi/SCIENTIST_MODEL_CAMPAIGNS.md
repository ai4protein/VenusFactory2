# Persistent Scientist model campaigns

Native Codex four-role execution is now available through `expert-codex` with an explicit Responses profile; see [CODEX_RESPONSES.md](CODEX_RESPONSES.md). Its real-CLI local integration tests do not yet establish an external-model publication campaign.

The campaign runner schedules admitted protein modelling tasks through the
existing Scientist goal and evolution drivers. Each allocation freezes its task,
profile, lineage length and budget. Harness and LLM choices remain separate in
the existing profile contract. Model publication uses the shared independent
library validator; a completed goal alone never increments the verified count.

## Plan format

A JSON plan contains `allocations` and additional `resources`. Each allocation
has a unique `id`, a full `model_tasks` declaration in `task`, a public `profile`,
`generations` (1..3), and a `goal_budget` containing `max_cycles`, `model_calls`,
`tool_calls` and `wall_seconds`. Resources contain aggregate `model_calls`,
`tool_calls` and `wall_seconds`. No credential belongs in a plan.

Plans are appended as immutable versions. `extend` can add new tasks and explicit
resource budgets, preserving all earlier task records and failures. It cannot
replace a prior allocation. The campaign's research objective, model target,
required stages and runtime source remain fixed.

## Execution and completion

```sh
PYTHONPATH=src python scripts/rsi/run_scientist_model_campaign.py create --output RUN --plan plan.json --objective 'Develop models across the admitted protein tasks' --target-models 1000 --require-stage activity --require-stage stability --model-library LIBRARY
PYTHONPATH=src python scripts/rsi/run_scientist_model_campaign.py run --output RUN --credential-toml CONFIG
PYTHONPATH=src python scripts/rsi/run_scientist_model_campaign.py status --output RUN
PYTHONPATH=src python scripts/rsi/run_scientist_model_campaign.py extend --output RUN --plan next_plan.json
```

Use a frozen repository for a sustained campaign. The library and campaign must
share an owner directory. The runner supports at least 1000 allocations, but
allocations, synthetic tests, repeated seeds and repeated registrations do not
count as verified models. Completion replays the canonical publication evidence,
counts distinct entries originating in this campaign, and checks the declared
stage coverage. Models originally published by another campaign are retained as
duplicates and excluded from this campaign's new-model count. Exhausting the
plan below the target produces `needs_tasks`, not `completed`.

Each child goal retains its own long-horizon discussion, implementation,
validation, strict improvement and fresh-session reuse loop. The scheduler does
not score future labels. It stops model access after a recorded 401/403 rather
than attempting every remaining allocation. `resume` is an explicit operation
after access recovers; it preserves the child goal and consumed resources.

## Pause and interruption

`pause` requests a campaign pause and pauses the current child goal. `resume`
continues that existing goal without resetting its call budget. A single-runner
lease prevents concurrent scheduling. Process identity includes PID, boot ID and
start ticks; a lock file or stale JSON status alone is not evidence of liveness.

An interrupted allocation keeps its entire unresolved reservation. `recover`
requires that the recorded process is no longer alive, marks that allocation as
interrupted, and allows other queued allocations to proceed. It never silently
restarts the interrupted task. Unknown charges are conservative; a separately
verified reconciliation is needed to reclaim them.

Task budgets limit model and tool calls through the existing goal gateway.
Wall-time accounting includes observed execution and publication time; overrun
is retained and stops further allocations. Publication failures and interrupted
work remain separate from scientific model acceptance. The current publication
auditors cover native Kimi Code, DeepSeek and Codex expert execution. Codex
requires a Responses binding; its external-model publication campaign remains
unverified despite successful local four-role integration tests. Local Agent chat and MCP can discover
and reuse published models through the tools described in
[SCIENTIST_MODEL_LIBRARY.md](SCIENTIST_MODEL_LIBRARY.md). Configure
`VENUS_SCIENTIST_MODEL_LIBRARY` to the same library supplied with
`--model-library`; a campaign-specific library is not automatically copied to
other catalogs.


## Experience across tasks

Allocations can now select explicit verified library donors or inherit eligible publications from earlier campaign tasks. The chosen experience is frozen before the receiving goal starts; resume does not reselect donors. See [Cross-task model experience](LIBRARY_EXPERIENCE.md) for the none, summary and source conditions, ancestry overlap checks and matched-study design.


## Independent campaign evaluation

Prospectively admit a study, wait for every allocation to terminate, and independently score and replay its verified publications. Failed allocations remain in the feasibility denominator, and observed costs remain separate from reserved budgets. See [Scientist campaign evaluation](SCIENTIST_CAMPAIGN_EVALUATION.md).

See [Generation publication](GENERATION_PUBLICATION.md) for sharing verified generations while their parent Scientist goal continues. Existing frozen campaigns keep their admitted source; the new path applies to newly created runs.
