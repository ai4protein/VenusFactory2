# Dependency security review — 2026-11-08

## Changes

The platform initializer now applies both the model compatibility constraints and
the service security constraints. Its installation receipt includes both files,
so changing either triggers dependency verification and installation again.
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

## Verified scope

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

## Remaining model-runtime advisories

| Package | Current constraint | Upgrade dependency |
| --- | --- | --- |
| Transformers | 4.48.1 | ESM 3.2.3 requires Transformers below 4.48.2 |
| Torch | 2.8.0 | The initializer aligns Torch, torchvision and compiled PyG wheels; these must move together |

These advisories remain open. Even ESM 3.4.2 requires Transformers below 5 and
Torch below 2.12, while the reviewed fix targets include Transformers 5.10 and
Torch 2.13. A compatible migration therefore needs separate ESM/model environments
or a validated upstream update, followed by checkpoint and inference tests.

The checkpoint-loader advisory also affects `weights_only=True`; that flag alone
does not resolve it. See the
[PyTorch advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
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

The scope is the resolved platform environment. Independently installed external
model runtimes still require their own audit and inference validation. GHSA and
PYSEC identifiers may describe the same issue; the number of OSV records is not
the number of GitHub Dependabot alerts.

## Rollout

This revision updates source, installation metadata and CI. Running services keep
their existing Python environments until a separately verified environment rollout.
Their older dependency audit is recorded in
[deployment validation](deployment-validation.md). Recheck each optional external
model runtime independently when installing or upgrading it.
