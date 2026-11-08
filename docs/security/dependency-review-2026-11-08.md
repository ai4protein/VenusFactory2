# Dependency security review — 2026-11-08

## Changes

The platform initializer now applies both the model compatibility constraints and
the service security constraints. Its installation receipt includes both files,
so changing either triggers dependency verification and installation again.
The compatibility venv/current-environment installers now share one implementation
and apply both constraints at every installation step. Torchvision is pinned with
Torch, the PyG wheel index is checked against that ABI, and dependency/import checks
must pass before success. Commands use argument arrays instead of shell strings.
The main initializer and compatibility commands now read the same Torch/torchvision
pins and validate the matching PyG wheel index before creating an environment.
All three installation stages apply both constraints files; changing the wheel
policy invalidates the installation receipt.
Automatic cu128 selection requires a known driver advertising CUDA 12.8 or newer;
unknown or older drivers select CPU, as in the supported initializer.

Linux and macOS declarations use patched minimum versions for the Web/MCP,
agent, document-processing and scientific utility packages. The macOS PEFT pin
also matches the existing platform compatibility constraint.

The frontend uses npm and `frontend/package-lock.json` throughout initialization,
development and CI. The obsolete pnpm lockfile contained different dependency
versions and has been removed. DOMPurify supplies its own types; the redundant
`@types/dompurify` package has been removed. The install policy permits the locked
esbuild installer and disables Scarf installation telemetry.

GitHub Actions now use pinned Node 24 action revisions. CI checks npm advisories
and validates skill routing descriptions strictly. An uninstalled optional skill
library is a valid minimal installation; an incomplete populated library still
fails strict validation.

## Initial service and frontend verification

These regression counts precede the Torch compatibility update below.

| Check | Result |
| --- | --- |
| Declared Python exact/minimum versions, OSV | 22 affected package/version pairs before, 2 after |
| Resolved Linux environment, both constraints | 260 packages; findings in Torch and Transformers |
| Resolved Apple Silicon macOS environment, both constraints | 224 packages; findings in Torch and Transformers |
| Frontend npm audit, production and development dependencies | 0 known vulnerabilities, 326 dependency entries |
| Backend coverage, segmented run and correction reruns | 5,474 passed; 16 skipped |
| Installer regression | 69 passed |
| CI boundary, skill and installation regression | 113 passed |
| Frontend tests | 114 passed |
| Frontend type check and production build | Passed |
| Source credentials/paths, repository layout and documentation | Passed |
| Strict skill validation, including reference paths | 36 skills; 0 errors and 0 warnings |

The broad backend run was completed in segments after an execution time limit.
Two environment-sensitive assertions were corrected and their modules retested:
the reference check now validates frozen source bytes across Biopython upgrades,
and the simulated download test explicitly controls its offline flags. Reference
tables retain their recorded extraction version and source hashes.

The Python inventory counts affected package/version pairs rather than individual
advisories; GHSA and PYSEC identifiers can describe the same vulnerability.
[OSV query documentation](https://google.github.io/osv.dev/post-v1-querybatch/).

Isolated CPU checks exercised Biopython 1.88 FASTA parsing, Accelerate 1.15.0
optimization, and an ESM LoRA forward/backward pass using PEFT 0.21.2,
Transformers 4.48.1 and Torch 2.8.0. macOS validation covers dependency resolution;
a macOS device and GPU inference were not used in these checks.

## Torch compatibility update

The platform installation pair is now Torch 2.10.0 and torchvision 0.25.0, with
the matching PyG 2.10 wheel index and Triton 3.6.0. ESM and Transformers retain
their existing compatibility constraints. Independently installed external model
runtimes keep their own versions.

An isolated Python 3.12 CPU environment passed dependency consistency checks and:

- Loaded the real ESM2 8M checkpoint and reproduced embeddings within a maximum
  absolute difference of 1.44e-6 from the Torch 2.8 environment.
- Completed a LoRA update through the platform training head, then saved and
  reloaded the adapter and head with identical predictions.
- Generated a 141-residue design for the A chain of the bundled 4HHB structure
  using real ProteinMPNN weights, with finite score and probability outputs.
- Executed scatter, neighbor-search, sparse-matrix and graph-convolution operations
  using the matching compiled PyG extensions.

These are software compatibility checks, not new scientific models. The CUDA 12.8
wheel metadata includes the required Linux/Python 3.12 extensions; GPU execution
and macOS device testing have not been performed in this update.

The updated resolution covers 263 packages on Linux and 224 on macOS.
Each profile reports 38 OSV records, down from 45.
Torch contributes two remaining records and Transformers 36. Aliases can describe
the same vulnerability, so this reduction is not a count of GitHub alerts closed.

## Remaining model-runtime advisories

| Package | Current constraint | Upgrade dependency |
| --- | --- | --- |
| Transformers | 4.48.1 | ESM 3.2.3 requires Transformers below 4.48.2 |
| Torch | 2.10.0 | Later patched releases require replacing or rebuilding the remaining compiled PyG extensions |

These advisories remain open. Even ESM 3.4.2 requires Transformers below 5 and
Torch below 2.12, while the reviewed fix targets include Transformers 5.10 and
Torch 2.13. A compatible migration therefore needs separate ESM/model environments
or a validated upstream update, followed by checkpoint and inference tests.

Torch 2.10 fixes the checkpoint-loader vulnerability that also affected
weights_only=True. This does not make arbitrary model code safe or resolve the
remaining Torch advisories. See the
[PyTorch loader advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
[remaining TorchScript advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47),
[ESM release metadata](https://pypi.org/pypi/esm/3.4.2/json) and
[Transformers advisory](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w).

## Continuous audit

The Python dependency audit workflow resolves Linux and Apple Silicon macOS profiles
with both constraints files, then queries OSV for every resolved package. It runs
when dependency declarations change, weekly, and on manual dispatch. Resolution
does not install GPU libraries or download model weights.

The audit prints every finding, including the remaining model-runtime risks.
`open-model-advisories.json` records the reviewed package, exact version,
advisory identifier, affected profile, migration reason and review deadline.
Only these specific records can remain non-blocking until 8 December 2026.
A new advisory, different vulnerable version, overdue review, incomplete response
or unavailable audit service fails the check. These entries do not dismiss
GitHub alerts or indicate that the underlying vulnerabilities are fixed.

For a strict check that also fails on the recorded risks:

```sh
uv pip compile requirements.txt -c requirements/bootstrap-constraints.txt \
  -c requirements/service-security.txt --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --no-header --no-annotate \
  -o /tmp/venus-resolved.txt
python scripts/audit_dependencies.py /tmp/venus-resolved.txt --profile linux --strict
```

Optional CubeSandbox and E2B client manifests apply the service constraints even
when installed directly. Their resolved Linux dependencies are audited separately
in strict mode; platform model-risk entries cannot exempt these clients. The
initial scans covered 11 CubeSandbox and 27 E2B packages with no findings.

The scope is the resolved platform environment and these optional clients. Independently installed external
model runtimes still require their own audit and inference validation. GHSA and
PYSEC identifiers may describe the same issue; the number of OSV records is not
the number of GitHub Dependabot alerts.

## Rollout

This revision updates source, installation metadata and CI. Running services keep
their existing Python environments until a separately verified environment rollout.
Their older dependency audit is recorded in
[deployment validation](deployment-validation.md). Recheck each optional external
model runtime independently when installing or upgrading it.

## GitHub maintenance

Dependabot checks the Python manifests, frontend lockfile and GitHub Actions weekly.
Security updates for the scientific runtime are grouped separately from service
updates so Torch, Transformers and ESM compatibility is reviewed together. Other
patch updates are grouped to reduce review overhead. Major updates still require
compatibility checks; no update is automatically merged or deployed. No advisory
is ignored or dismissed by this configuration.

Each Python audit run retains its exact resolved inventory and JSON findings as a
GitHub Actions artifact for 14 days. The existing audit gate continues to fail on
new or overdue findings and reports known open model risks explicitly.
