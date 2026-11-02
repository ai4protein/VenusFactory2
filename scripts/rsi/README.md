# RSI command-line entry points

Run commands from the repository root. These scripts reuse `src/agent/rsi/`;
they do not contain a second Scientist implementation.

| Directory | Commands |
| --- | --- |
| `scripts/rsi/` | General topic, worker and campaign entry points |
| `cases/` | Individual paper reproductions such as CLEAN and RFantibody |
| `data_preparation/` | Fetching public data and preparing task manifests |
| `evaluation/` | Audits, evaluation, replay, screening and evidence freezing |
| `figures/` | Figure and summary generation from existing evidence |

Use each script's `--help` before execution. For example:

```bash
python scripts/rsi/cases/run_clean_annotation.py --help
python scripts/rsi/data_preparation/prepare_poisk.py --help
```

New recipes use these paths. Saved experiments retain their original manifests
and source snapshots. Do not rewrite historical hashes or import generated model
code into this scripts directory. Store new experiment output under the configured
runtime or `research_runs/` root.
