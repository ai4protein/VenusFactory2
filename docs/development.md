# Developer guide

Start with the [user tutorials](wiki/Home.md) to understand the research workflows.
Implementation and deployment details are organised below:

- [Repository layout](architecture/repository-layout.md): source ownership, runtime assets and skills.
- [Model integration](architecture/model-integration.md): pretrained methods, Factory-trained models and RSI artifacts.
- [Services and operations](architecture/services.md): local/online deployment, API/MCP processes, worker pools and job recovery.
- [Installation](installation.md): dependencies, optional components, model weights and health checks.

## Maintain the tutorials

The in-app Manual and Agent use the canonical tutorials in `docs/manual/en/` and
`docs/manual/zh/`, indexed by `docs/manual/catalog.json`. Update both languages
when behavior changes, then run this from the repository root:

```bash
python scripts/sync_platform_docs.py
python scripts/sync_platform_docs.py --check
```

Wiki pages, legacy manual files and `docs/README.md` are generated views. Edit
the source tutorials or generator instead of those views.

## Maintainer checks

From the repository root, use the installed platform Python environment:

```bash
python scripts/check_repository.py
python scripts/check_layout.py
python scripts/validate_skills.py
python scripts/sync_platform_docs.py --check
PYTHONPATH=src python -m pytest -q tests/test_model_family_boundaries.py tests/test_model_identity.py tests/test_managed_model_runtimes.py tests/test_tool_runtime.py
```

These are targeted layout, documentation and model-integration checks; they do
not replace regression tests for the areas you change. See the
[service guide](architecture/services.md) for API, MCP and worker startup, and
the [directory guide](architecture/repository-layout.md) for source and runtime ownership.
Current UI screenshots and capture details are listed in [images/README.md](images/README.md).

## Development checkpoints

Complete one coherent behavior change, then review its implementation before
starting another feature or preparing a commit. Also stop for this review when a
fix introduces a second path for the same behavior, another compatibility
exception, or a dependency across package boundaries.

1. Trace the affected user action through API, worker, MCP and model code. Keep
   each policy and state transition in one owning module; callers use its public
   interface.
2. Inspect the diff and existing implementations. Consolidate repeated validation,
   configuration and lifecycle logic. Remove a replaced path only after checking
   callers, saved jobs, CLI entry points and tests.
3. Keep compatibility at an explicit boundary. New code imports the owning package;
   modules in the same RSI engine use relative imports. Preserve standalone bundle
   execution, stable tool names and frozen research evidence. Do not relax source
   hashes or numerical verification to make historical results pass.
4. Run the maintainer checks above and focused behavioral tests. Changes to a
   bundled runtime also need a standalone execution check; changes to research
   validation need replay and rejection checks.
5. Update the existing architecture or user documentation when behavior changes.
   Keep machine-specific profiles, experiments and verification logs in the ignored
   runtime or research directories. Commit refactoring separately from unrelated
   features.

Remove investigation-only traces and their configuration switches when closing a
fix. Operational diagnostics belong in the existing job or service logs, tied to
the actual task identity; do not keep hard-coded investigation sessions in model
or request code.

Keep experimental fixes provisional until the intended behavior passes its
acceptance check. A passing protocol or regression suite is not evidence that a
model-quality workaround helped. Retain the failed experiment and its source
binding separately instead of adding another special case to the main path.

Review the affected subsystem as a whole after every two or three feature batches: identify
cycles, duplicated state, obsolete adapters and oversized modules with unrelated
responsibilities. Refactor around those responsibilities, rather than moving files
only to make directories look smaller. Record concrete unresolved issues with an
owner and next verification step; do not create a parallel checklist for every fix.

## Consolidation priorities

Address these responsibilities when next changing their subsystem. Keep each
refactor separate from new product behavior, with a bounded regression target.

| Subsystem owner | Consolidation target | Verification before merging |
| --- | --- | --- |
| Chat UI (`ChatPage.tsx`) | Separate session navigation, composer actions and history presentation from page coordination | Switching mode or session must preserve running jobs, drafts and streamed answers |
| Model API (`custom_model_api.py`) | Keep training, evaluation and prediction submission separate; share ownership and admission policy | Owner isolation, online limits, idempotent queue submission and existing route contracts |
| RSI engine (`generated_context*.py`) | Share repeated execution primitives while retaining version-specific scientific contracts | Standalone bundles, strict scope checks, unchanged numerical results and complete source bindings |

The layout check rejects imports from shared Harness contracts or worker adapters
back into worker entry points. The dispatcher owns worker selection; reusable
events and session argument validation belong in their owning modules.

## Before a commit

Use Python 3.12 and Node 24. Keep credentials, runtime installations, session data,
training weights and research evidence outside source control. Optional scientific
skills stay in their Git submodule; do not copy the vendor checkout into Git.

Run `python scripts/check_repository.py --staged` and `git diff --cached --check`
against the proposed commit. The repository check detects private paths, large
blobs and known credential patterns (including example ZIPs); it is a guardrail,
not a complete secret scanner or security audit.

For a release, use a clean source checkout without `.env` or model credentials:

```bash
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=src python -m pytest -q
cd frontend
npm ci --no-audit --no-fund
npm test
npm run build
```

Run service/worker integration tests on Linux with bubblewrap available. Real GPU
inference, public services and live model-provider probes are separate acceptance
checks, not implied by offline regression. CI checks source boundaries, tutorial
consistency and the frontend; full backend validation uses the installed platform
environment. Refresh README screenshots when navigation changes, recording their
capture details in `docs/images/manifest.json`.


## Publishing a release

Update the Python, frontend, lockfile and container-default versions together and add `docs/releases/vX.Y.Z.md`. Run `python scripts/check_release.py`. Merge the reviewed source into `main` before creating the corresponding annotated tag.

The tag workflow reruns repository/frontend checks, verifies that the tagged commit belongs to `main`, scans the source archive with the checksum-pinned Gitleaks binary, and publishes the archive with SHA-256 checksums. It uses the repository's ephemeral Actions token; no publishing credential belongs in `.env` or the source tree.

Scan Git history separately before a release. Removing a secret from the current tree does not revoke a historical credential; credential rotation and any coordinated history rewrite are separate operations. Gitleaks exceptions must be narrowly scoped to reviewed synthetic fixtures or public data identifiers, never whole test directories or historical secret commits.
