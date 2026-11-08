# Dependency security follow-up — 2026-11-09

## Platform and ESM3 separation

Factory prediction, training and the API environment now use Transformers 5.10.4,
Tokenizers 0.22.2 and Hugging Face Hub 1.5.0. The obsolete ESM 3.2.3 dependency
has been removed from the platform profiles. Torch/PyG remains on the validated
2.10 ABI.

Transformers 5.10.0 is a withdrawn PyPI release; it was rejected during candidate
resolution. The selected 5.10.4 release is not withdrawn and had no OSV findings
in the complete platform resolution. The upstream fixes include
[model-configuration code execution](https://github.com/advisories/GHSA-29pf-2h5f-8g72)
and [tokenizer output path traversal](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w).

ESM3 structure preprocessing uses a separate managed runtime, with explicit
Python and checkpoint paths, a disabled default, tensor-only loading, bounded
process execution and validated result records. Existing dataset-merger and
CSV/JSON interfaces dispatch to it. An absent, disabled or unverified runtime
does not fall back to the platform interpreter.

Its profile uses ESM 3.4.2, Torch 2.11.0 and Transformers 4.57.6, matching
[ESM's declared dependencies](https://pypi.org/pypi/esm/3.4.2/json).
The dependency resolver is allowed to enforce upstream constraints; no
`--no-deps` workaround or patched package metadata is used.

## Audit results

| Resolved profile | Packages | Open OSV records |
| --- | ---: | ---: |
| Linux platform | 248 | 2 |
| Apple Silicon macOS platform | 190 | 2 |
| Optional Linux ESM3 runtime | 106 | 15 |

Each platform profile previously had 38 records. The remaining two platform
records describe Torch issues; Transformers contributes none in these profiles.
The optional runtime reports one Torch, eleven Transformers and three setuptools
records. GHSA and PYSEC identifiers can refer to the same vulnerability; these
numbers are not GitHub Dependabot alert counts.

ESM requires Torch below 2.12 and Transformers below 5. Torch 2.11 also requires
setuptools below 82. The setuptools findings concern
[macOS source-distribution exclusions](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
and [legacy Windows wheel installation](https://github.com/advisories/GHSA-xmj3-9gf7-h52w);
the optional runtime is reviewed for Linux preprocessing and does neither.
These version findings still appear in its audit. They have not been dismissed
as fixed.

The inventory binds every remaining finding to its exact package version and
profile, with a review deadline of 8 December 2026. ESM3 records cannot exempt
the platform or sandbox clients. New findings, incomplete scans and overdue
reviews fail CI. The independent ESM3 audit does not apply the incompatible
platform constraints.

## Compatibility verification

- A real ESM2 8M checkpoint produced finite embeddings with a maximum absolute
  difference of 2.15e-6 from the previous Torch 2.10 / Transformers 4.48 environment.
- Factory LoRA completed an optimizer update; adapter and head predictions were
  identical after saving and reloading.
- Compiled PyG scatter, neighbor search, sparse multiplication and graph
  convolution passed in the candidate CPU environment.
- ESM, Bert, T5, ESMFold, VenusPLM and Factory prediction modules imported with
  the new stack.
- An untrained encoder fixture produced identical 141-residue structure tokens
  with ESM 3.2.3 and 3.4.2. This checks API/numerical compatibility only, not
  pretrained model quality.
- The real ESM3 installation command created an independent environment and
  passed its dependency/import checks. No pretrained ESM3 encoder checkpoint was
  present; installation correctly remained disabled and unverified.

These are software compatibility checks, not newly trained scientific models.
macOS was resolved and audited, not tested on a macOS device. GPU execution was
not exercised in this update.

## Deployment scope

The source change does not replace a running service's environment, restart PBS
jobs or alter deployment directories. Roll out a separately verified environment
before treating the running installation as updated. Independently provisioned
third-party runtimes retain their own versions and require separate audits.
