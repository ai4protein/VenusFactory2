# Model runtimes

Model Plaza and Workflows use the same typed parameters and queued execution.
The catalog includes LigandMPNN, AntiBMPNN, BindCraft, BoltzGen, AutoDock Vina
and DiffDock. LigandMPNN, AntiBMPNN, AutoDock Vina and DiffDock are enabled in
both online and local mode when installed and enabled by the administrator. They
use the same resource queues and input limits through Model Plaza, Workflow and
Chat. BindCraft and BoltzGen remain local-only, with visible but disabled online
cards. Missing installations are marked Not configured.

Set `VENUS_MODEL_RUNTIMES_CONFIG` to an administrator-owned JSON file, or use
`configs/model_runtimes.json`. This is server configuration, never browser input.
Copy `configs/model_runtimes.example.json` to `configs/model_runtimes.json`.
Paths in the profile are relative to the **repository root**, regardless of the
worker's current directory. Absolute paths remain supported for external managed
environments. Install dependencies and weights before setting `enabled: true`.

```json
{
  "vina": {
    "enabled": false,
    "executable": "runtime/local/model-runtimes/vina/bin/vina",
    "cpus": 2,
    "timeout_seconds": 1800
  }
}
```

The example profile includes seven external runtimes, four mutation runtimes, ProteinMPNN and two task-model runtimes. Installed profiles,
weights and environments are excluded from Git; code, schemas and templates are
versioned. Built-in mutation models can select per-model Python environments; without a profile,
ProSST/ESM-1v/ESM-IF1/ProtSSN and ProteinMPNN retain the existing application Python in a child process.
Other built-in models still use application Python in queued workers. External
models use separate CLI environments; Scientist-generated
models use their reviewed artifact and isolated execution. These are runtime
strategies, not three copies of the model catalog. A CPU-only utility does not
need its own permanent server process.

Runtime installations and weights are separate from the application. Review
upstream licenses (including BindCraft dependencies) during installation.
BoltzGen may download large weights on first execution; provision its cache
before enabling it. Its current web form covers protein/peptide binder design,
not every upstream antibody, redesign or small-molecule protocol.

## Inputs and outputs

- LigandMPNN: protein–ligand complex PDB, chain/residue constraints and sampling settings.
- AntiBMPNN: PDB, antibody chain IDs and explicitly selected design positions;
  comma-separated position groups correspond to the space-separated chains.
- BindCraft: target PDB, target chains/hotspots, length range and final design count.
  Filters and advanced stage settings come from the server profile.
- BoltzGen: target PDB, target/binder chains, length range and optional binding positions.
- Vina: prepared receptor PDBQT **and** ligand PDBQT, docking box and search settings.
  Receptor/ligand preparation is required; the adapter does not rename PDB to PDBQT.
- DiffDock: protein PDB plus ligand SDF/MOL or SMILES. SMILES is parsed into a
  prepared SDF inside the configured environment, never interpreted as a file path.

Uploaded secondary files are checked against the same job owner before enqueueing
and again before execution. Model outputs have an owner-protected ZIP download;
archives cannot include paths outside that job's work directory. Vina uses the
CPU queue; the other five use the GPU queue. Device selection can be fixed with
`cuda_visible_devices` in a runtime entry. BoltzGen is limited to one device.

The current queue is single-node. Local and online queues have separate state;
hardware-wide GPU allocation across installations is still an operational concern.

## Compatibility notes from the installed previews

The research preview keeps each new environment below
`runtime/local/model-runtimes`; it does not install packages into an existing
application environment. Upstream commits, checkpoint SHA-256 values, original
source backups and test artifacts are recorded under
`research_runs/rsi/20261101_model_parameters`.

- LigandMPNN's bundled OpenFold used removed `np.int`, `np.float` and related
  aliases. The installed copy replaces these aliases with their equivalent Python
  builtins. Its upstream source and the compatibility changes are retained.
- AntiBMPNN's current entry point defines `--path_to_model_weights` and `--seed`
  but ignores both in `main`. Before enabling a fresh installation, replace
  `path_to_model_weights='../antibmpnn_model_weights'` with
  `path_to_model_weights=args.path_to_model_weights or '../antibmpnn_model_weights'`,
  and use `seed=args.seed or int(np.random.randint(0, high=999, size=1, dtype=int)[0])`
  for the seed assignment. The installed preview preserves the original file.
- External Python runtimes do not inherit the application's `PYTHONPATH` or
  provider credentials. This prevents application dependency paths from overriding
  each model's isolated environment.

Small example runs verify integration and outputs; they do not establish model
accuracy, binding affinity, or scientific reproduction of the original papers.

Official interfaces: [LigandMPNN](https://github.com/dauparas/LigandMPNN),
[AntiBMPNN](https://github.com/zeysun/AntiBMPNN),
[BindCraft](https://github.com/martinpacesa/BindCraft),
[BoltzGen](https://github.com/HannesStark/boltzgen),
[AutoDock Vina](https://autodock-vina.readthedocs.io/en/latest/docking_basic.html),
[DiffDock](https://github.com/gcorso/DiffDock).

## Built-in mutation models: ProSST, ESM-1v, ESM-IF1 and ProtSSN

Their maintained implementations remain in `mcp_tools/models/pretrained/mutation/models/`.
Dispatch is in `mcp_tools/models/pretrained/runners/mutation.py`; both API/MCP zero-shot predictions
and ProSST/ESM-IF1/ProtSSN calls inside the mutation ensemble use it. Only the four known
model adapters are allowed: requests cannot supply an executable or script.

Profiles use keys `prosst`, `esm1v`, `esmif1`, `protssn`. Each supports `python`, `enabled`,
`model_id`, `weights`, `cache`, `cuda_visible_devices`, `timeout_seconds`.
`weights` is an installed checkpoint path (HF-format directory for ProSST/ESM-1v,
an ESM checkpoint file for ESM-IF1) and takes precedence over the upstream `model_id`.
`cache` separates Hugging Face and Torch caches. ProSST additionally supports
`structure_weights`, a directory containing `AE.pt` and `2048.joblib`.

The adapter code is shared, but the selected Python interpreter and its packages
are not imported into the API process. Provider credentials and arbitrary parent
`PYTHONPATH` are not inherited. This is trusted model execution, not an untrusted
code sandbox; generated code continues to use the session sandbox.

A missing profile preserves the existing environment and checkpoint caches, so
reorganizing source does not silently disable installed tools or redownload large
weights. An explicitly disabled or incomplete profile fails without falling back
to another environment. The template is deliberately disabled: install and verify
the environment/weights before enabling it. Configuration support alone is not
proof that a new isolated environment has been installed or scientifically validated.

Paths in all profiles are repository-relative; `model_id` remains an upstream
identifier, not a filesystem path. Local and online service owners can select
different profile files with `VENUS_MODEL_RUNTIMES_CONFIG`; keep environments in
their respective `runtime/local` and `runtime/online` trees. Queue and administrator
tool policies remain the authority for resource admission and online availability.

## Model identity: engine, artifact and scientific task

An execution environment is not a model identity. The same runtime can execute
many checkpoints; the same backbone can yield many independently trained models.
Model Plaza, workflow components and the shared model catalog preserve existing
IDs and distinguish these model types:

| Type | Example | Identity and role |
| --- | --- | --- |
| `foundation` | ESM2-650M | Backbone available for training |
| `pretrained` | ProteinMPNN, ProSST, ProtSSN | Published prediction/design method and pretrained weights |
| `finetuned` | DeepSol · ESM2-650M | Task checkpoint trained with a named backbone and dataset |
| `trained` | User regression model, reviewed Scientist model | Task artifact that need not use a PLM |
| `tool` | AutoDock Vina, file conversion | Computational tool rather than a learned checkpoint |

Scientific categories (solubility, design, docking, etc.) and source
(`platform`, `scientist`, `community`) remain separate filters. The shared public
metadata includes `model_type`, `runtime_id`, `backbone`, `training_dataset`,
`training_method` and `provenance`. New registered checkpoints also record the
weight SHA-256 and a content-derived `version`. A dataset/backbone name is a display
label, not a replacement for the model ID. A Factory checkpoint trained on DeepSol
is not represented as the original DeepSol architecture.

Existing records gain only metadata that can be derived from their existing
configuration; unknown training data is left blank. This does not change visibility,
ownership, review status or claim independent validation. Scientist candidates
still need independent approval before publication. Installation paths and secrets
are not part of public model cards.

## ProteinMPNN

The `proteinmpnn` runtime serves both design and scoring through the same fixed
entry point. It supports `python`, `enabled`, `weights`, `cache`,
`cuda_visible_devices` and `timeout_seconds`. Its `weights` directory contains:

```
weights/
  vanilla_model_weights/v_48_020.pt
  soluble_model_weights/v_48_020.pt
  ca_model_weights/v_48_020.pt
```

Other checkpoint names matching `v_<neighbors>_<noise>` are supported. The form
lists checkpoints from the configured directories. Chain constraints, fixed/tied
positions, temperature, random seed, batch size, sequence count, probability and
scoring options are forwarded intact. A configured root takes precedence over
legacy paths; missing configured weights do not trigger a fallback download.
Without a profile the existing `ckpt/ProteinMPNN` layout remains supported.

ProtSSN uses its `weights` directory for GNN checkpoints, `model_id` for its PLM,
optional `config` for the GNN configuration and `use_ensemble: false` for a single
model. Normalization assets are resolved relative to the adapter, not the job's
working directory. Missing explicitly configured weights fail instead of downloading
a different checkpoint set. The default cache remains compatible with existing
installations.

## Factory-trained artifacts

The `finetuned` runtime executes bundled function/residue models (ESM2, Ankh,
ProtT5, ProtBert). `plm_checkpoint` executes registered Custom Model prediction
and evaluation. Both support `python`, `enabled`, `cache`,
`cuda_visible_devices`, `timeout_seconds`. Neither accepts a global `weights` or
`model_id`: each task artifact selects its own training configuration and checkpoint.
For example, DeepSol and SoluE can share a runtime and ESM2 backbone while retaining
different trained weights and registry IDs. Training itself retains its existing
queue, environment and online ESM2-8M restriction.

Legacy bundled task weights remain under `ckpt/<dataset>/<backbone>/`, and
registered artifacts retain their authoritative registry location. This migration
does not move or duplicate large weight directories. New dedicated interpreters
can be provisioned below `runtime/<mode>/model-runtimes/` and selected by profile.
The managed children inherit compute/cache settings but not API-provider keys.
Queue admission, cancellation and administrator tool policies remain in force;
the runtime adapter is not a second scheduler or a generated-code sandbox.


## Sequence annotation: S4PRED and InterProScan

Both tools use the existing shared MCP service and administrator tool controls.
They appear in Model Plaza and Workflows, and can be discovered by Agent,
Scientist and Researcher. They are available in both local and online mode when
configured; BindCraft and BoltzGen remain local-only.

| Tool | MCP name | Input | Output | Scheduling |
| --- | --- | --- | --- | --- |
| S4PRED | `predict_s4pred` | Protein sequence | Per-residue C/H/E probabilities, SS2 and JSON | CPU queue, 2 threads |
| InterProScan | `query_interproscan` | Protein sequence, then returned job ID | Families, domains, sites, InterPro/GO/pathway annotations | EBI API submission and polling |

Install [S4PRED](https://github.com/psipred/s4pred) and its five official weights
in the configured root. The published weights archive MD5 is
`e04ad7d10b61551f7e07a86b65bb88dc`. The preview uses upstream commit
`7f309dfc9a8155efdf30113fb4cb0affb40aeb7e`, a dedicated Python launcher, and CPU
inference. Its installation receipt contains `verified: true` only after a real
prediction. Online requests accept at most 2,000 residues and run for at most
180 seconds; local requests accept up to 5,000 residues and 600 seconds.

For InterProScan, set `interproscan.enabled` and an administrator contact email
in the runtime profile (`contact_email`), or supply `EBI_EMAIL`. No API key is
required. The adapter uses the official
[EMBL-EBI Job Dispatcher API](https://www.ebi.ac.uk/Tools/services/rest/iprscan5).
The submitted sequence is processed by EBI. MCP submission returns a platform
job ID immediately; call the same tool with that ID to retrieve its status and
results. IDs are owner-bound, persisted in the mode's queue database, and cannot
be used to read another owner's results. Identical active requests reuse a job;
an uncertain network submission is not automatically repeated. The current
limit is four active EBI requests per mode, with at least 15 seconds between
polls. Report/workflow adapters wait within their existing bounded task budget.

SignalP 6 is not installed. Its GitHub development source is not a replacement
for the DTU-distributed package and fast weights. Add it only after the required
distribution has been obtained; S4PRED predicts secondary structure, not signal
peptides.

## GPU runtime provisioning

Keep code, environments and caches below `runtime/local/model-runtimes/` and
refer to them with repository-relative paths in the profile. Do not install
these dependencies into the web/API environment. The integration versions used
in this research checkout are:

- BindCraft: the official repository, Python 3.10, JAX 0.4.35, ColabDesign and
  PyRosetta, with the official AlphaFold 2022-12-06 parameter archive. JAX 0.4.35
  needs the older `nvidia-cuda-nvcc-cu12==12.6.85` package layout. Follow upstream
  dependency licensing and size the GPU for the target and binder length.
  When animation saving is enabled, put ffmpeg on this runtime's PATH.
  The tested environment uses imageio-ffmpeg==0.6.0 and a runtime-local
  bin/ffmpeg symlink to its bundled executable; no system-wide install is needed.
  The health check reports a missing encoder before a long design job starts.
- BoltzGen: `boltzgen==0.3.2`, Torch 2.7.1 with CUDA 12.6 and cuequivariance
  packages pinned to 0.5.1. Download its inference assets before enabling:
  `runtime/local/model-runtimes/boltzgen-env/bin/boltzgen download all --cache runtime/local/model-runtimes/boltzgen-cache`.
  The configured cache is passed to every run; inference uses offline cache
  access. Newer upstream releases can change CUDA and GPU requirements.
- DiffDock: the official v1.1 model bundle, Torch 2.2.2 with CUDA 12.1,
  matching PyG wheels and the ESM2-650M checkpoint. The example profile points
  to `workdir/v1.1/score_model` and `confidence_model`; the latter uses
  `best_model_epoch75.pt`. Its first import creates SO(3)/SO(2) numerical caches
  and can take several minutes. Initialize once in the configured repository
  root before enabling concurrent requests. Request-specific step counts are
  capped consistently in the generated inference configuration.

There is no permanent model process: a queued job starts the configured CLI,
captures its log and output files, then releases its resource slot. Timeouts
terminate the child process tree. Installed profiles and weights are not tracked
by Git. Installation and bounded inference evidence for this setup is kept in
`research_runs/rsi/20261102_protein_services`; consult the result files rather than
treating a successful package install or `--help` as an inference test.


These three GPU runtimes also require an `installation` receipt with the matching
`runtime` name and `verified: true`, written after a real inference smoke test.
Keep the test settings and output with that receipt. A BindCraft trajectory may
be rejected by scientific filters while still demonstrating successful runtime
execution; that is not an accepted binder. The installed BindCraft copy counts
rejected as well as relaxed trajectories towards `max_trajectories` to bound
attempts. Its original source and compatibility patch hash are preserved with
the installation evidence.


The configured preview caps BindCraft at 20 attempted trajectories and 3,600
seconds, using `BindCraft/factory_advanced.json`. DiffDock's installed entry point
also caps its requested file-descriptor limit at the host hard limit. BoltzGen's
five checkpoint SHA-256 values were checked against the official Hugging Face
release. Its bounded six-stage test produced CIF, CSV and PDF artifacts; none
of that tiny test's candidates passed scientific quality filters. DiffDock's
single-pose test used 20 diffusion steps. Both are integration evidence only.
Installed dependency snapshots live beside each runtime's installation receipt
as `requirements.installed.txt`.

## Initialization

Use `python3 initialize.py plan`, `python3 initialize.py install` or `python3 initialize.py doctor` from the repository root.
The declarative catalogue is `configs/bootstrap.json`; implementation lives in
`scripts/bootstrap/`. See [installation and health checks](installation.md).

## RFD4-Proteina, RFantibody and ESMFold runtimes

Install optional environments and public assets without changing the Web Python:

```bash
python3 initialize.py install --tools rfd4_proteina,rfantibody,esmfold,esmfold2 --stages tools
python3 initialize.py doctor --tools rfd4_proteina,rfantibody,esmfold,esmfold2 --stages tools
```

| Runtime ID | Implementation | Environment and assets |
| --- | --- | --- |
| `rfd4_proteina` | Official RFD4-Proteina | Python 3.12; CUDA 12 profile; pinned native flow/autoencoder bundle |
| `rfantibody` | Official RFantibody | Upstream uv lockfile; RFdiffusion-Ab, ProteinMPNN and RF2 weights |
| `esmfold` | Original ESMFold | Hugging Face `facebook/esmfold_v1`; independent subprocess; PDB output |
| `esmfold2` | Biohub ESMFold2 | Python 3.12; Torch 2.11; defaults to ESMFold2-Fast; bundled ESMC and CCD data |

These are separate software/model versions. The original local implementation loaded
ESMFold v1; renaming it to ESMFold2 would not upgrade its weights. ESMFold2-Fast
downloads approximately 26 GB of checkpoint data plus its CCD dictionary.
The CCD asset is pinned and checksum-verified along with the model files.

Source revisions and weight revisions are recorded in `configs/bootstrap.json`.
Environments, clones, weights, dependency snapshots and inference receipts belong in
`runtime/local/model-runtimes/<runtime>/`, which is excluded from Git.
A public Hugging Face mirror can be selected through `HF_ENDPOINT` for
installation; the default remains Hugging Face. Do not pass API credentials into
scientific model subprocesses.

Installing a package does not publish a new Chat tool or enable an online service.
The initializer reports `needs_validation` until the configured runtime has
real inference evidence. Keep administrator switches, session ownership and queue
limits in place. Run optional native CLIs through a GPU task slot, with an explicit
interpreter, working directory, output folder and timeout. RFantibody also needs its
virtual environment's `bin` directory on PATH because its CLI launches child
Python processes.

A small generated structure verifies software execution and file delivery. It does
not establish a useful binder, model accuracy or experimental validity. Keep rejected
designs distinguished from candidates accepted by scientific filters.

### Previously published Scientist models

Publication metadata remains readable when a frozen model interface no longer
matches the installed runtime. Such entries are marked **Needs revalidation** and
are excluded from callable model search. Describing and predicting still validate
the full evidence; matching a code fingerprint alone is not a new scientific review.
Use the operator's explicit numerical migration/review process before restoring
execution. Never rewrite a historical policy fingerprint simply to remove an error.

LigandMPNN design selections must identify protein residues with complete backbone atoms. The preflight rejects missing chains, DNA-only selections, invalid residue labels and masks with no designable residues. A completed subprocess with zero designed residues is treated as a failed prediction. Downloadable FASTA headers retain the checkpoint filename without exposing its server path.


## Install independent structure predictors

Install one tool, or several with a comma-separated list. Installation runs on
Linux without sudo and shows five stages, download sizes, speed, elapsed time
and a final report:

```bash
python3 initialize.py install --tools alphafold2,protenix,mol1 --manager uv
python3 initialize.py doctor --tools alphafold2,protenix,mol1 --verify-hashes
```

Each tool lives under `runtime/local/model-runtimes/<id>/`: `source/` is a pinned
upstream checkout, `env/` is its private Python, `bioenv/` contains native search
programs, and `data/`, `cache/`, and `installation.json` hold its assets and
installation record. These downloaded files stay outside Git. Native search
programs use a private Conda prefix, even when Python packages use uv. The
installer can prepare the manager itself; it never installs into system Python.

| Tool ID | Official implementation | Default assets |
| --- | --- | --- |
| `alphafold2` | [DeepMind AlphaFold](https://github.com/google-deepmind/alphafold) | Monomer, pTM and multimer-v3 parameters; databases are separate |
| `protenix` | [ByteDance Protenix](https://github.com/bytedance/Protenix) | Public `protenix_base_default_v1.0.0` checkpoint and inference reference data |
| `mol1` | [Molence Mol-1](https://github.com/molence/Mol-1) | `mol-1-base` checkpoint, common data and template sequence database |

`Proteinix` is accepted as an alias for `protenix`; `Mol-1` is an alias for `mol1`.
The Protenix code includes newer model support, but the installer does not select
or redistribute its restricted v2 weights. Protenix template-search databases, RNA MSA databases and training
datasets are not part of a default inference installation.

AlphaFold2's full sequence pipeline needs large databases. Download them explicitly
only when required; the full platform installation also leaves them out:

```bash
# Budget at least 650 GiB free for reduced databases, or 3,000 GiB for full.
python3 initialize.py install --tools alphafold2 --alphafold-databases reduced
```

Rerun the same command to reuse completed environments and downloads. Existing
administrator profiles are preserved. `dependencies_and_files` means installation
checks passed; it does not mean a scientific benchmark or the full MSA pipeline
has passed. New structure profiles remain disabled for platform use until a
bounded prediction has been checked and their service is enabled.


Use the native CLI after installation; the helper selects the private interpreter,
reference-data root, CUDA compiler and cache without activating an environment:

```bash
python3 scripts/run_structure.py protenix pred --help
python3 scripts/run_structure.py mol1 doctor
```

Installing software does not publish an MCP tool or change an online/local
administrator switch. Existing working services keep their current interpreters.
New ProSST, ESM-1v, ESM-IF1 and ProtSSN installations also use their own environments.
