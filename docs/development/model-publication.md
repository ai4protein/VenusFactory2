# Model discovery and RSI case publication

Agent and Researcher discover enabled platform tools and Factory-trained models
through search, description and invocation. RSI-developed models are excluded from
those execution catalogs in both local and online mode, including older registered
RSI entries. Public RSI results are repository cases, not callable tools.

Scientist retains private project deliverables and scientific review evidence.
Private cases are visible only through their owner's authorized research sessions.
The RSI Plaza within Model Plaza provides README/file previews and ZIP downloads.

## Operator case publication

Inspect the candidate's source, test report, data/license provenance and admission
receipt. The existing local selection command records publication of the approved
case package and its author:

```bash
WEBUI_V2_MODE=local python scripts/publish_research_model.py path/to/reviewed-candidate \
  --goal-id RESEARCH_ID --author-email author@example.org --curator operator@example.org
```

Independent admission and unchanged source bindings remain required. Publishing a
case does not register an MCP prediction tool. The case gallery reads the separate
operator decision and exposes only the approved project package. Unselected entries
and unrelated workspace files remain private. No publication command is available
as an MCP tool.

Users request case publication by email to tanyang.august@sjtu.edu.cn, including
the project, evaluation report, author email, scientific scope and sharing permission.
There is no public application inbox or automatic publication on research completion.

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
