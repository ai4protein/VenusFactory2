# Public biology MCP tools

Agent, Scientist and Researcher share the same public-service adapters. These
tools are registered in the native tool hub and the unified VenusFactory MCP
server; the Kimi Code, Codex, Claude Code and DeepSeek harness integrations use
that shared catalog. `search_research_tools` discovers the current tool schemas.

| Tool | Use |
| --- | --- |
| `query_ncbi` | Search NCBI protein, nuccore, gene, taxonomy or PubMed; retrieve summaries, FASTA or abstracts |
| `query_ncbi_blast` | Submit a protein sequence to NCBI BLAST against SwissProt, then check the returned job ID |
| `query_foldseek_remote` | Submit a mapped UniProt structure or PDB chain to the public Foldseek server, then check the job ID |
| `query_gget` | Structured access to gget info, seq, pdb, archs4, bgee, opentargets and enrichr |
| `query_public_protein_service` | InterPro/Pfam, QuickGO, STRING, Reactome, AlphaFold DB, PDBe, and monitored BLAST/Foldseek analysis |

No separate API key is required for these public database services. Chat model
selection continues to follow the configured free-model policy. Public remote
Foldseek is available online; local Foldseek/VenusMine remains restricted.

## gget runtime

From the repository, using the application Python:

```bash
python scripts/setup_public_query_tools.py
```

This installs gget 0.30.8 into `.venvs/public-query`, leaving the web environment
unchanged. Restart the research web/MCP processes after updating the tool catalog.

Example arguments:

```json
{"operation":"info","identifiers":["ENSG00000141510"]}
{"operation":"seq","identifiers":["ENSG00000141510"],"translate":true}
{"operation":"pdb","identifiers":["1A3N"],"resource":"entry"}
{"operation":"archs4","identifiers":["TP53"],"resource":"tissue"}
{"operation":"opentargets","identifiers":["ENSG00000141510"],"resource":"diseases","limit":10}
```

`info`/`seq` accept at most five Ensembl IDs (also supported WormBase/FlyBase IDs).
Other lookups accept one identifier; enrichment accepts 2–200 gene symbols and
an explicit library name. Results include source URLs, completion status and
row truncation information. gget runs in an isolated child process without API
credentials, with a 120-second deadline and bounded network responses/output.
No shell command, output path, paid gget GPT module or local training is exposed.

## Remote jobs

Use the explicit BLAST/Foldseek tools for interactive chat. A submission returns
`status: pending`, `job_id`, `url` and `next_poll` (Unix time). To check the same
job, call the same tool with **only** `job_id`. A pending job is not a failed or
completed analysis. Do not resubmit the input while waiting. NCBI polling is
limited to once per minute per request, across server workers.

The report pipeline may use the generic adapter to monitor a job within its
longer execution budget and preserve its checkpoint. Empty results, unavailable
providers and pending jobs remain distinguishable from completed evidence.
