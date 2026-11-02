# Protein discovery and mutation scoring

## Local and online policy

| Capability | Local | Online | Execution |
| --- | --- | --- | --- |
| BindCraft, BoltzGen, DiffDock | Requires an installed runtime and weights | Visible, disabled | Isolated model environment; queued GPU job |
| NCBI BLAST | Enabled | Enabled | NCBI public API, blastp against SwissProt |
| Foldseek | Enabled | Enabled | Public Foldseek API; PDB and AlphaFold SwissProt databases |
| VenusREM | Enabled | Enabled | ProSST-2048 inference plus remote Foldseek retrieval |
| ProSST-20 / 128 / 512 / 1024 / 2048 / 4096 | Enabled | Enabled for inference | Matching model checkpoint and structure vocabulary |
| VenusMine / local Foldseek database search | Enabled if installed | Disabled | Local databases and compute |
| PLM fine-tuning | Configured backbones | ESM2 8M only | Training queue |

Administrator switches remain authoritative in both modes. Allowing ProSST prediction does not enable training larger backbones online. `Not configured` is an installation state, not permission to bypass a runtime check.

## Model Plaza, workflows, and Chat

NCBI BLAST and Foldseek appear in **Protein Discovery**. Both use the existing public-service MCP implementation; Model Plaza queues a workflow over that same implementation. BLAST takes a FASTA/sequence table (one to five proteins per job); Foldseek takes a PDB structure and optional chain. Results contain actual hits, remote job identifiers, and source links. Waiting, unavailable, or timed-out requests are not reported as completed zero-hit predictions.

Remote jobs are submitted once and their tickets checkpointed before polling. NCBI requests retain global spacing and per-RID polling limits; public services retain their daily submission cap. Chat tools `query_ncbi_blast` and `query_foldseek_remote` support submit/query by job ID. The UI job worker performs bounded polling for uploaded data and reports intermediate status.

VenusREM uses the same Foldseek gateway, not the local database workflow. It retains up to 20 alignments per selected database. The output records the retrieval URL and alignment count. Service failure fails the prediction; a genuine zero-hit search is explicitly labelled as a ProSST-only fallback in the exported scores.

## Runtime profiles

Configure installations in `configs/model_runtimes.json`; see `configs/model_runtimes.example.json`. Relative paths resolve against the repository, independently of the worker directory. A missing BindCraft/BoltzGen/DiffDock environment or checkpoint leaves its local card disabled. Installing an adapter alone is not installation of its upstream model.

ProSST shares one environment and structure encoder. Each variant selects a distinct checkpoint and clustering vocabulary. A `prosst.variants` object may override `weights`, `model_id`, `structure_weights`, `enabled`, or runtime settings per vocabulary size. Non-2048 variants never inherit a 2048-only checkpoint path. Checkpoints may be pre-cached by the administrator; the default loader downloads the selected public Hugging Face model on first use.

## Wet-lab capability priorities

These are proposed additions, not currently advertised as installed tools.

1. **SignalP 6.0 fast**: signal peptide type and cleavage site; useful for secretion constructs and mature-protein boundaries. Use the distilled fast mode for routine work. Obtain the standalone package and confirm the license covers a public platform before deployment. The free website is not a documented unrestricted programmatic API.
2. **DeepTMHMM**: membrane topology and transmembrane helices, complementary to SignalP for construct design. The standalone distribution requires a license request; do not bundle it without permission.
3. **InterProScan / HMMER through EMBL-EBI**: novel-sequence domain/family annotation. Prefer the documented job API to hosting large signature databases. Existing InterPro accession lookup is not equivalent to a fresh sequence scan.
4. **S4PRED**: CPU single-sequence secondary structure; no MSA or local search database, approximately 430 MB of weights. Useful for a low-cost structure summary alongside sequence properties.
5. **DeepCoil2**: coiled-coil regions and heptad positions. CPU support, but its older Python/embedding stack warrants a separate runtime and measured memory limits.

Use structured outputs for residue coordinates, scores, model version, and evidence/source URLs, so Agent, Scientist, Researcher, Model Plaza, and workflows share results. Put remote annotation in network queues and local sequence inference in bounded CPU/GPU queues.

Official sources checked 2026-11-02:

- https://blast.ncbi.nlm.nih.gov/doc/blast-help/urlapi.html
- https://blast.ncbi.nlm.nih.gov/doc/blast-help/developerinfo.html
- https://search.foldseek.com/
- https://services.healthtech.dtu.dk/services/SignalP-6.0/
- https://github.com/fteufel/signalp-6.0/blob/main/installation_instructions.md
- https://interproscan6.readthedocs.io/stable/licensed_applications/
- https://www.ebi.ac.uk/jdispatcher/docs/
- https://github.com/psipred/s4pred
- https://github.com/labstructbioinf/DeepCoil
