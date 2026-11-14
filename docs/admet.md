# Small-molecule ADMET — package selection and integration

Reviewed 2026-11-01. There is no supported claim that a single package is best on
every endpoint, assay protocol or chemical domain.

| Candidate | Fit for VenusFactory2 | Current decision |
| --- | --- | --- |
| [ADMET-AI](https://github.com/swansonk14/admet_ai) | Local pretrained batch prediction, Python/CLI, MIT package; Chemprop v2 | Default inference engine, pinned to PyPI 2.0.1 |
| [OpenADMET Models](https://github.com/OpenADMET/openadmet-models) | Traditional ML, deep learning and active-learning workflows; program-specific training | Candidate for future Scientist training workflows; not installed in this change |
| [ADMETlab 3.0](https://academic.oup.com/nar/article/52/W1/W422/7640525) | Broad hosted prediction and API, including 119 reported endpoints/properties | Optional future remote source; not used for inference here |

ADMET-AI's 2024 paper and public web server use v1. The current package uses
retrained Chemprop v2 models without the old RDKit fingerprint inputs. Do not
transfer the v1 benchmark ranking to v2 or call it a current universal SOTA result.
The [package documentation](https://pypi.org/project/admet-ai/) explicitly states
that v1 and v2 predictions differ. Physicochemical descriptors are still computed
by RDKit; that is separate from the v1 fingerprint model input.

## Product contract

- Model Plaza / ADMET: paste one SMILES per line, upload CSV/TSV/JSON with a SMILES
  column, or upload SDF/MOL. Chat uses the same registered model and schema.
- Tool name `predict_admet`: shared native Agent and MCP implementation, searchable
  with `search_research_tools`. Standard tool scheduler places it on the CPU queue.
- Uses an isolated CPU-only environment. No paid LLM or hosted ADMET API is called.
- Local/online share immutable installed dependencies and weights. Inputs, results,
  queues and ownership remain in their respective mode runtime directories.
- Online: maximum 32 molecules per call; local: 1000. Both: 256 heavy atoms per
  molecule, 2 compute threads, 300-second runtime budget by default (600 hard cap).
- Invalid molecules retain their input row with an explicit status; valid duplicate
  molecules share computation but preserve all original rows. All-invalid input
  fails clearly. Input/prediction column collisions are rejected rather than
  overwriting supplied measurements.
- The result view groups endpoints by ADMET category. CSV/JSON export retains all
  rows, and endpoint metadata includes original units, species and source links.
  DrugBank percentiles are not emitted or presented as calibrated uncertainty.
- Classification outputs are positive-class probabilities for the named assay,
  not a universal safety score. This is a small-molecule model, not a predictor of
  antibody/protein pharmacokinetics. No across-endpoint traffic-light verdict is
  fabricated. Benchmark metrics were not recomputed in this integration.

## Runtime configuration

`configs/model_runtimes.json` entry `admet_ai` needs `enabled`, repository-relative (or absolute) `python`
and a verified `installation` receipt path; optional `timeout_seconds` defaults
to 300. The installation receipt includes package/dependency versions and weight
hashes. Missing or unverified installation leaves the model unavailable.

Environment: `runtime/local/model-runtimes/admet-ai-env`.
ADMET package license: MIT. Individual underlying TDC dataset licenses remain
separate; this change does not redistribute those training datasets.
