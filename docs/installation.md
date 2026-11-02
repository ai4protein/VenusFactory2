# Installation and health checks

`initialize.py` is the supported entry point. It delegates to `scripts/bootstrap/`;
`configs/bootstrap.json` describes tool sources, versions, dependencies and profiles.
Factory artifacts are indexed separately in `ckpt/manifest.json`.

## One-command full installation

```bash
python3 install_full.py              # full installation and health checks
python3 install_full.py plan         # inspect the complete plan without installing
python3 install_full.py doctor       # audit the same complete selection
```

This prepares the platform environment, built-in capabilities, frontend, scientific
skills, all four Harnesses, the external-tool catalogue, Factory `predict-all` weights
and their five required backbones, and the RSI model-library integration. It uses the
same progress, checksums, rerun and environment detection as `initialize.py`.
Factory heads and backbone snapshots are checked separately; a partial Hub cache
is not accepted as a complete backbone. `doctor --verify-hashes` also checks the
published SHA256 values of cached backbone files where provided by the Hub.
Licensed tools or missing GPU prerequisites are listed separately and produce an
incomplete result; the installer continues preparing independent components.
It does not start a new service automatically.

Database retrieval, web search, public protein servers, session files, basic analysis,
the sandbox interface and the manual are **built in**, not optional model plugins.
Core setup also installs the isolated gget dependency environment. Tavily/SerpAPI keys
and the InterProScan contact email are configured separately in Administration.
Installing an adapter does not establish remote-service credentials or availability.

| Optional family | What it supplies |
| --- | --- |
| `external` | Independent pretrained model/tool runtimes, including ProteinMPNN, LigandMPNN, ProSST, ProtSSN, docking and ADMET |
| `factory` | Factory-trained task heads plus their foundation-model weights |
| `rsi` | Discovery and execution of reviewed Scientist models in a selected library |
| `harnesses` | Kimi Code, DeepSeek Harness, Codex and Claude Code |

```bash
# Built-in platform only; no optional model/Harness families.
python3 install_full.py --components none
# Core + selected families and selected members of each family.
python3 install_full.py --components external,harnesses --tools proteinmpnn,ligandmpnn --harnesses codex
# Core + Factory and a previously populated RSI library.
python3 install_full.py --components factory,rsi --rsi-library /path/to/reviewed-library
```

`--components` chooses installation contents; Administration controls runtime enable/
disable permissions for all modes. Installing a package does not override disabled
models. A fresh RSI library can be empty. Scientist publishes models through the
existing independent review/evidence checks; attachment does not auto-publish entries.
The default new library is `runtime/local/models/rsi/library`; existing research
libraries and `VENUS_SCIENTIST_MODEL_LIBRARY` configuration retain precedence.
An explicit `--rsi-library` attaches an existing library without copying model assets.

## Start here

Use Linux (or WSL2), Git, an initial Python 3.10+ interpreter and outbound HTTPS.
The installer creates Python 3.12 for the platform. It does not need `sudo`, modify
shell startup files, write API keys, start services, or reset an existing checkout.

```bash
python3 initialize.py plan
python3 initialize.py install
.venv/bin/python initialize.py doctor --verify-hashes
```

The default **minimal** profile prepares only a managed Python environment. It
does not install Torch, model weights, external repositories, Harnesses, Node or
vendor skills. This is a bootstrap environment, not yet a runnable Web service.
Install each component when needed:

```bash
# Prepare the Web/API/MCP environment and build the frontend (includes ML dependencies).
python3 initialize.py install --dependencies platform --stages environment,frontend
# Independent additions; each flag selects only its own stage.
python3 initialize.py install --harnesses deepseek-harness,codex
python3 initialize.py install --tools proteinmpnn,ligandmpnn
python3 initialize.py install --factory-preset demo
python3 initialize.py install --stages skills
```

The complete Web dependency set still includes Torch/PyG; select it explicitly.
Large models may require tens of GiB. The plan reports exact Factory artifact
bytes, not an invented total for all upstream dependencies. Source remains
separate from ignored runtime assets. For a new Git checkout use `--depth 1`
without `--recurse-submodules`; download vendor skills only when requested.
Existing Git history is not rewritten by this installer.

Automatic manager selection reuses **uv**, then **Conda**, when available;
otherwise it bootstraps uv. You can override it with `--manager uv/conda`.
When uv is unavailable,
uv is installed inside `runtime/local/bootstrap/uv-manager` from its official,
checksum-verified PyPI wheel; this also works when the initial Python has no pip
or venv module. Choose Conda explicitly:

```bash
python3 initialize.py install --manager conda
```

An existing Conda executable is reused. Otherwise the installer obtains Miniforge
from its official release, verifies its published SHA256, and installs it inside
the project. Both managers create separate environments for incompatible tools.
The platform environment defaults to `.venv`; `--environment` selects another
project-owned prefix. No global/system package installation is performed.

`--torch-type auto` chooses CUDA 12.8 wheels only when the NVIDIA driver advertises
CUDA 12.8 or newer; otherwise it selects CPU. Override with `--torch-type cpu` or
`--torch-type cu128`. Tool-specific GPU packages retain their tested CUDA versions.
macOS/native Windows can inspect a plan, but the complete multi-process/sandbox
service is provisioned on Linux; this command does not pretend to provision a
Linux sandbox on another operating system.

## Choose components

| Selection | What it prepares |
| --- | --- |
| `--profile minimal` | Default: managed Python only |
| `--profile standard` | Platform + built-ins + frontend + skills + RSI integration + ProteinMPNN, LigandMPNN, S4PRED + `predict-core` |
| `--profile extended` | Standard + AntiBMPNN, ProSST-2048, ESM-1v, ESM-IF1, ProtSSN, ADMET-AI and Vina |
| `--tools a,b` | Install only the named tools when `--stages` is omitted |
| `--harnesses deepseek-harness,codex` | Install only the named Harness clients |
| `--profile full` / `install_full.py` | Complete platform and all four optional families |
| `--dependencies platform --stages environment,frontend` | Prepare the full Web service without model collections |
| `--stages environment,builtins,frontend,harnesses,tools,factory,rsi,skills` | Explicit stage selection; runnable platform always includes built-ins |
| `--factory-preset none/demo/predict-core/predict-all/all` | Select Factory weights; `proteinmpnn` also remains a legacy checkpoint preset |

Examples:

```bash
# Only CPU-compatible MPNN tools; use their own fresh environments.
python3 initialize.py install --stages tools --tools proteinmpnn,ligandmpnn --torch-type cpu --factory-preset none

# Preserve an existing platform environment without installing/upgrading packages in it.
python3 initialize.py install --environment /absolute/path/to/env --use-existing-environment

# Download all Factory task heads. Prefetch additional backbone models explicitly.
.venv/bin/python initialize.py install --stages factory --factory-preset predict-all --backbone facebook/esm2_t33_650M_UR50D

# Install optional GPU packages and weights; scientific validation is a separate step.
.venv/bin/python initialize.py install --stages tools --tools boltzgen,diffdock --factory-preset none

# Check existing local/online services, without starting or restarting either one.
.venv/bin/python initialize.py doctor --service-url http://127.0.0.1:7862/api/runtime-config --service-url http://127.0.0.1:8084/health
```

`--tools all` includes GPU tools and entries with manual prerequisites. It is useful
for a complete plan; installation returns an incomplete status if prerequisites
remain. It does not silently skip those entries and report success.

The frontend uses Node.js **24 LTS** and its npm 11 toolchain. Python compatibility
versions are recorded in `requirements/bootstrap-constraints.txt`; in particular,
the MCP/FastMCP pair and Torch/PyG ABI are kept compatible. The installer runs both
`pip check` and dependency import checks.

If the official Hugging Face endpoint is unreachable, explicitly choose a mirror:
`--hf-endpoint https://hf-mirror.com`. The same Factory SHA256 manifest is still
checked. There is no silent mirror fallback, and no Hugging Face credential is sent
by these public-model downloads.

## Runtime scope

Harness clients are optional and pinned in `scripts/bootstrap/harnesses.py`.
DeepSeek uses the official `deepseek-harness-sdk` (not the unrelated PyPI package
named `deepseek-harness`); Codex/Claude use their official npm packages with native
executables; Kimi uses its official release archive. Each gets its own
`runtime/local/harnesses/<id>/` directory and receipt. The platform discovers these
receipts automatically. `VENUS_KIMI_BIN`, `VENUS_CODEX_BIN`, `VENUS_CLAUDE_BIN`,
`VENUS_DSH_BIN` and `VENUS_DSH_SITE` retain precedence over managed defaults.
An install/version check does not make an LLM request, configure credentials or
override the administrator's permitted model routes.

Official installation references: [Codex CLI](https://developers.openai.com/codex/cli/),
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness),
[Kimi Code](https://github.com/MoonshotAI/kimi-code),
[Claude Code](https://code.claude.com/docs/en/setup).

| Runtime | Provisioning and verification |
| --- | --- |
| ProteinMPNN | Official source and vanilla/soluble/CA weight collections; isolated environment; one-sequence CPU inference |
| LigandMPNN | Pinned official source, design and side-chain packing checkpoints; isolated Python 3.11; design inference |
| AntiBMPNN | Pinned official source and Zenodo weights; guarded path/seed compatibility patches; sequence inference |
| S4PRED | Pinned source, five model weights with published archive checksum; 76-residue CPU inference |
| ProSST | Official ProSST-2048 snapshot and all structure-tokenizer assets; model loading check; other vocabularies remain separate configured variants |
| ESM-1v / ESM-IF1 / ProtSSN | Official weights, pretrained-runtime settings and model loading; ProtSSN includes nine GNN artifacts |
| ADMET-AI | Independent pinned package environment; ethanol prediction with the official ensemble |
| AutoDock Vina | Official Linux x86_64 release binary; CLI check; remains disabled until administrator validation |
| BoltzGen / DiffDock | Independent GPU environments and official weights; CLI check, **not** a binding/docking quality claim; remain disabled pending real workload validation |
| BindCraft | Existing installations are audited; fresh installation needs separately licensed PyRosetta and a bounded GPU test. See [runtime guide](model-runtimes.md) |
| SignalP 6 | DTU package/fast weights are required; no unofficial substitute is downloaded |
| InterProScan / BLAST / Foldseek | Remote services, not large locally installed databases. InterProScan needs an administrator contact email |

Source revision, download origin, asset sizes/digests and the verification level
are recorded. Small MPNN tests establish that inference executes, not that newly
designed proteins meet a wet-lab objective. A CLI `--help`/`--version` result never
becomes an `inference` receipt. Existing verified runtime receipts are preserved.

For Factory weights, the initializer reuses `scripts/download_ckpts.py` and then
checks every selected artifact against `ckpt/manifest.json`. `predict-all` downloads
task-specific heads/adapters across backbones; add the foundation models required
by those heads with repeatable `--backbone` options. Backbones are cached inside the
configured `finetuned` runtime, not mixed into the Factory task-head directory.

## Files, reruns and recovery

```text
initialize.py                    public command
scripts/bootstrap/               implementation and health checks
configs/bootstrap.json           provisioning catalogue, no secrets
configs/model_runtimes.json       administrator paths and enable/disable switches
ckpt/manifest.json               Factory artifact hashes and collection presets
.venv/                           managed platform environment
runtime/local/bootstrap/         manager bootstrap, plans, logs, results and lock
runtime/local/model-runtimes/     upstream source, dedicated environments and weights
runtime/local/harnesses/          optional Harness clients, SDKs and version receipts
runtime/local/builtin-tools/      isolated dependencies for built-in services (gget)
runtime/local/components/         component receipts and RSI library selection
runtime/local/models/rsi/library/ default library for a fresh installation
runtime/local/                   local sessions/queues/results
runtime/online/                  online sessions/queues/results
```

Local and online services may read the same installed model assets. Their mutable
session/queue/output stores remain separate. Installing a tool does not override
online policy, administrative disable switches or scientific publication review.

Rerunning the same command reuses completed verified assets and managed environments.
The terminal shows numbered stages, tool/Harness counts, elapsed package-manager
time and download byte progress/speed/ETA when the server supplies a total size.
SSH/CI output uses occasional plain lines; `--json` keeps stdout machine-readable,
and `--no-progress` suppresses progress. Full package output stays in `install.log`.
Checksum-backed HTTP partial downloads resume on retry when the server supports
Range; otherwise they restart. A checksum failure is never promoted to a final file.
Partial downloads are not installed as final weights. Existing corrupt files,
unexpected Git origins/revisions and unowned environments produce an actionable
error instead of being overwritten. The administrator can move the problematic
asset aside and rerun the same command. No `git reset`, `git clean` or recursive
deletion of installed environments is used.

The installer preserves custom profiles. A copied, untouched example profile can
be populated on a fresh machine, while its disabled switch remains disabled.
An already configured custom profile is audited rather than replaced. If it is
incomplete, fix its configured paths or remove that one profile after backing it up.

Each run writes `runtime/local/bootstrap/<timestamp>/plan.json`, `install.log` and
`result.json`; `latest.json` points at the latest report. `doctor --report path.json`
writes a machine-readable report. Exit codes: **0** requested checks/stages passed,
**1** incomplete/failed checks, **2** invalid invocation or an interrupted run.
`doctor` is offline unless explicit `--service-url` checks are requested; it does
not issue LLM calls, download models or start jobs. Import checks, file checks,
previous inference evidence and currently enabled status are separate fields.

API credentials are configured separately in the administrator interface or `.env`.
The installer neither creates credentials nor enables paid-model fallback.
Bubblewrap availability is reported, but host namespace/cgroup/sandbox operational
checks still determine whether generated analysis code can safely execute.

## Start the services

```bash
.venv/bin/python scripts/serve.py --mode local --port 7862 --mcp-port 8084
.venv/bin/python scripts/serve.py --mode online --port 7863 --mcp-port 8085 --api-workers 2
```

For separate API, MCP and worker processes, use `--component api`, `--component mcp`
and `--component workers`; see [service startup and recovery](architecture/services.md).
The initializer provisions files; `scripts/serve.py` owns runtime startup.

## Upstream references

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [Miniforge distribution](https://github.com/conda-forge/miniforge)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [LigandMPNN checkpoints](https://github.com/dauparas/LigandMPNN/blob/main/get_model_params.sh)
- [S4PRED source and weight checksums](https://github.com/psipred/s4pred)
- [Factory checkpoints](https://huggingface.co/AI4Protein/VenusFactory2-ckpts)


Source release archives omit optional Git submodule contents. `python3 initialize.py install --stages skills` also works from an unpacked release: it installs the exact upstream revision pinned in `configs/bootstrap.json`, preserving its license. In a Git checkout the same command initializes the pinned submodule. Existing modified reference libraries are not overwritten.
