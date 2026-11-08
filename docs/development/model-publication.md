# Model discovery and publication

Model Plaza and MCP discovery use `mcp_tools/models/platform_catalog.py` for model identities, runtime readiness and live administrator switches. Agent starts with search / describe / call, not all model schemas. Model names and schema metadata are searchable on the host; only bounded results enter the LLM context. Skills use list_skills(query, limit, offset), then read_skill.

Scientist writes reviewed deliverables to the goal's private_capabilities directory in both modes. Independent scientific review does not grant public visibility. Browser users cannot publish by changing visibility or invoking legacy goal publication. Publication applications are emails to tanyang.august@sjtu.edu.cn, not stored in an application inbox.

## Operator publication

Inspect the candidate's source, test report, data/license provenance and admission receipt. On the local deployment, use the platform Python environment:

```bash
WEBUI_V2_MODE=local python scripts/publish_research_model.py path/to/reviewed-candidate \
  --goal-id RESEARCH_ID --author-email author@example.org --curator operator@example.org
```

The command requires valid independent admission and unchanged source files. It stores a separate publication decision bound to the candidate and review hashes, preserving the original research manifest. The published author email is visible in discovery and Model Plaza. No publication command is exposed as an MCP tool. Use Admin's model switch to disable a released model.

Existing uncurated autonomous entries are retained on disk but excluded from the public capability catalog until selected and published with this command. Earlier paper-reproduction campaign libraries remain local-only; they are not imported into the online curated catalog.

## Revalidate a local research library

Operator audits can verify up to 256 registered entry IDs together with the current
runtime. Put the IDs in a JSON array, then run:

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src python scripts/rsi/scientist_model_library.py \
  --library path/to/model_library verify \
  --entry-ids entry_ids.json --output verification.json
```

This command replays training and checks the registered model, native research
history, validation evidence and reuse receipts. Models from one shared goal reuse
that goal's audit within the call; single-model and batch inference use the same
registry binding checks. No validation cache survives the request.

A failed verification creates no success report, and an existing report is never
overwritten. Reports describe verified historical models, not newly developed
models or scored future labels. The frozen-campaign audit in
`scripts/rsi/evaluation/audit_shared_publications.py` remains available for
reproducing the original environment's checks. Neither audit grants public access;
operator selection and independent admission remain separate requirements.

Agent discovery returns short model names and descriptions. The
`describe_evolved_model` tool loads the exact applicability scope, input columns
and recorded validation metrics only after a model is selected. Description checks
the publication hashes, index, goal anchors and runtime interface without running
the research again. It reports `verified_now: false`; the metrics are the recorded
assessment, not a new validation. Actual prediction still resolves and verifies
native execution, numerical evidence and reuse receipts before calling the model.
No metadata response or cache grants execution approval.

Category search identifies a reader before querying model records. A navigation
result supplies the next description call, then the selected reader receives the
original topic as suggested arguments. Model listing uses the shared bilingual
matcher, so a phrase such as "Scientist model <entry ID>" still finds the record.
The first discovery response contains no model input schemas or full catalog.


Agent discovery and Model Plaza share a short-lived metadata snapshot. Its
15-second lifetime starts after reading completes, and concurrent requests share
one read. New index entries invalidate it. Discovery reads each model's canonical
publication without traversing older retraining variants; the operator catalog
still reports the complete origin history.

Runtime compatibility and administrator switches are checked for each catalog
response. Invocation checks current permissions and evidence independently of
the metadata snapshot. Online Model Plaza excludes the local research library;
curator-selected publications retain their author information and normal access.
